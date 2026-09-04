"""RLResidualAgent -- run a PPO-trained residual policy on the real NAVSIM benchmark.

Why this file exists
--------------------
rl/eval.py answers "did the policy beat RAP on held-out navtrain scenes, scored
through the same precomputed cache it trained on". That is the right question while
training, and the wrong one for a paper: it reuses the frozen scene latents, the
frozen base trajectories and the train-split metric cache. The number everyone else
reports comes from the devkit's run_pdm_score.py, which drives an AbstractAgent live
over navtest/navhard from raw sensors.

Nothing in the RL package could do that. This agent is the bridge: it is a drop-in
replacement for RAPAgent that runs the identical RAP model, then adds the learned
residual on top of RAP's selected trajectory. Point run_pdm_score.py at it with
agent=rl_agent and the RL policy is measured by exactly the code that produced the
baseline's number.

The one thing that must not drift
---------------------------------
The observation handed to the policy here is rebuilt live, but it has to be the same
vector rl/precompute.py wrote and the policy trained on -- same fields, same order,
same flattening, same dtype rounding -- and then normalised with the *training* run's
running mean/std. Every one of those is a silent failure if wrong: the policy still
emits actions, the trajectories still score, and the benchmark just reports that RL
did not help. _build_observation() mirrors precompute.encode_batch() field for field,
and the residual itself comes from rl/residual.py, shared with the training env.
"""

import inspect
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.rap_dino.rap_features import RAPFeatureBuilder, RAPTargetBuilder
from navsim.agents.rap_dino.rap_model import RAPModel
from navsim.common.dataclasses import SensorConfig, Trajectory

# The PDM scorer samples 4 s (default_scoring_parameters.yaml: 40 poses at 0.1 s), but
# this runs at time_horizon=5 -- 10 poses at 2 Hz -- because that is the horizon the
# checkpoint and the policy were trained at. Handing the scorer all 10 would score two
# seconds the metric cache holds no observations for. The devkit's RAPAgent truncates
# at exactly this number; the baseline column depends on it matching.
_SCORER_NUM_POSES = 8

from rl.residual import apply_residual, check_action_dim, delta_scale


class RLResidualAgent(AbstractAgent):
    """RAP, plus a learned bounded residual on the trajectory it selected.

    :param config: RAPConfig -- must match the checkpoint (time_horizon in particular)
    :param checkpoint_path: pretrained RAP weights, the same file precompute.py used
    :param policy_path: PPO .zip from rl/train.py
    :param vecnormalize_path: observation statistics saved alongside it. Defaults to
        vecnormalize.pkl next to policy_path.
    :param residual_scale: must match RLConfig.residual_scale at training time
    :param deterministic: use the action distribution's mean (yes, for evaluation)
    :param disable_residual: run as plain RAP. The honest A/B: identical code path,
        identical weights, residual forced to zero.
    """

    def __init__(
        self,
        config,
        checkpoint_path: str = "",
        policy_path: str = "",
        vecnormalize_path: str = "",
        residual_scale: Tuple[float, float, float] = (1.0, 1.0, 0.1),
        deterministic: bool = True,
        disable_residual: bool = False,
        lr: float = 1e-4,
    ):
        # RAP's AbstractAgent takes no arguments; the devkit's requires
        # trajectory_sampling positionally. This class is imported under both trees --
        # the devkit's during benchmark evaluation, RAP's from the RL package -- and a
        # bare super().__init__() is a TypeError under the devkit. The devkit's own
        # RAPAgent carries the same shim, hardcoded; binding whichever signature is
        # actually present keeps this file working in either checkout.
        if "trajectory_sampling" in inspect.signature(AbstractAgent.__init__).parameters:
            super().__init__(trajectory_sampling=config.trajectory_sampling)
        else:
            super().__init__()
        self._config = config
        self._lr = lr
        self._checkpoint_path = checkpoint_path
        self._policy_path = policy_path
        self._vecnormalize_path = vecnormalize_path
        self._residual_scale = tuple(residual_scale)
        self._deterministic = deterministic
        self._disable_residual = disable_residual

        self._rap_model = RAPModel(config)
        # Inference only. Matches RAPAgent, which freezes the backbone at construction.
        for p in self._rap_model.parameters():
            p.requires_grad = False

        self._policy = None
        self._obs_norm = None
        self._delta_scale: Optional[np.ndarray] = None
        self.device = torch.device("cpu")

    # ------------------------------------------------------------------ interface

    def name(self) -> str:
        return self.__class__.__name__

    def get_sensor_config(self) -> SensorConfig:
        """Identical to RAPAgent: the four cameras RAP's ImgEncoder consumes.
        The policy sees a pooled latent of exactly these, so they cannot differ."""
        return SensorConfig(
            cam_f0=[3], cam_l0=[3], cam_l1=[], cam_l2=[],
            cam_r0=[3], cam_r1=[], cam_r2=[], cam_b0=[3], lidar_pc=[],
        )

    def get_feature_builders(self):
        return [RAPFeatureBuilder(config=self._config)]

    def get_target_builders(self):
        return [RAPTargetBuilder(config=self._config)]

    def initialize(self) -> None:
        """Load RAP weights, then the policy. Called once per evaluation worker."""
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
        else:
            self.device = torch.device("cpu")

        self._load_rap_checkpoint()
        if not self._disable_residual:
            self._load_policy()
        else:
            print("[rl_agent] disable_residual=True -- running as plain RAP")

        num_poses = self._config.trajectory_sampling.num_poses
        self._delta_scale = delta_scale(num_poses, self._residual_scale)
        self.to(self.device)
        self.eval()

    # -------------------------------------------------------------------- loading

    def _load_rap_checkpoint(self) -> None:
        if not self._checkpoint_path:
            raise ValueError("rl_agent needs agent.checkpoint_path (the RAP weights).")
        state_dict: Dict[str, Any] = torch.load(self._checkpoint_path, map_location="cpu")
        state_dict = state_dict.get("state_dict", state_dict)
        # Lightning saved these as `agent._rap_model.*`; this class holds the model at
        # `_rap_model`, so the same rewrite RAPAgent.initialize does applies here.
        state_dict = {k.replace("agent._rap_model", "_rap_model"): v
                      for k, v in state_dict.items()}

        # The residual is meaningless if it is stacked on a different base trajectory
        # than the policy trained against, and a horizon mismatch is the way that
        # happens: init_feature is (num_poses * proposal_num, d_model), so strict=False
        # would skip it and leave a random embedding driving every proposal.
        expected = self._rap_model.init_feature.weight.shape
        found = state_dict.get("_rap_model.init_feature.weight")
        if found is not None and tuple(found.shape) != tuple(expected):
            raise ValueError(
                f"Checkpoint/config mismatch: init_feature is {tuple(found.shape)} in the "
                f"checkpoint but {tuple(expected)} here -- "
                f"{found.shape[0] // self._config.proposal_num} poses vs "
                f"{expected[0] // self._config.proposal_num}. Set "
                f"agent.config.trajectory_sampling.time_horizon to the value the "
                f"checkpoint was trained with."
            )

        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        # Every RAP weight that can affect a trajectory must land: strict=False otherwise
        # leaves part of the planner randomly initialised and still runs.
        #
        # domain_classifier is the one legitimate exception. It is the sim2real GRL head,
        # trained only in RAP's own training loop and absent from the released checkpoint
        # (RAP_DINO_navsimv2.ckpt is missing exactly these 6 tensors). RAPModel.forward
        # does call it, but its output goes only to output["domain_logits"] -- never to
        # trajectory, score, pred_logit or bev_feature -- so a random init here cannot
        # change what this agent emits. precompute.py tolerates the same 6 keys, which is
        # what makes the two sides comparable.
        stranded = [k for k in missing
                    if k.startswith("_rap_model.") and "domain_classifier" not in k]
        if stranded:
            raise ValueError(
                f"{len(stranded)} RAP weights missing from {self._checkpoint_path}, "
                f"first few: {stranded[:5]}. strict=False would leave these randomly "
                f"initialised -- the planner would run and quietly emit wrong trajectories."
            )
        if unexpected:
            print(f"[rl_agent] {len(unexpected)} unexpected keys (expected: "
                  f"domain_classifier, shared-refiner duplicates), first few: {unexpected[:5]}")

    def _load_policy(self) -> None:
        from stable_baselines3 import PPO

        if not self._policy_path:
            raise ValueError(
                "rl_agent needs agent.policy_path (a PPO .zip from rl/train.py), or "
                "agent.disable_residual=True to run the plain-RAP baseline."
            )
        policy_path = Path(self._policy_path)
        if not policy_path.exists():
            raise FileNotFoundError(f"No policy at {policy_path}")

        self._policy = PPO.load(str(policy_path), device=self.device)
        check_action_dim(
            int(np.prod(self._policy.action_space.shape)),
            self._config.trajectory_sampling.num_poses,
        )

        stats_path = Path(self._vecnormalize_path or policy_path.parent / "vecnormalize.pkl")
        if not stats_path.exists():
            # Not a warning. scene_latent is 5120 raw DINOv3 activations; feeding those
            # unnormalised to a policy trained on normalised ones produces a plausible
            # but wrong trajectory for every scene in the benchmark.
            raise FileNotFoundError(
                f"Missing observation statistics at {stats_path}. train.py writes "
                "vecnormalize.pkl next to final_model.zip and it is part of the trained "
                "model -- evaluating without it is meaningless. Pass "
                "agent.vecnormalize_path explicitly if it lives elsewhere."
            )
        import pickle

        with open(stats_path, "rb") as f:
            vec_normalize = pickle.load(f)
        # Only the running statistics are wanted, not the wrapper: normalize_obs() reads
        # obs_rms/clip_obs/epsilon and touches no venv, so the object works unattached.
        vec_normalize.training = False
        self._obs_norm = vec_normalize
        print(f"[rl_agent] policy {policy_path.name} + statistics {stats_path.name}")

        self._check_checkpoint_provenance(policy_path.parent / "obs_meta.json")

    def _check_checkpoint_provenance(self, meta_path: Path) -> None:
        """Refuse to stack this policy's residual on a different RAP checkpoint.

        The action is a delta on whatever trajectory the RAP model emits, so the base has
        to be the model the observations were built from. Point this at a different
        checkpoint -- easy, since the supervised pipeline's eval script uses a different
        one -- and every number is wrong with nothing raised: the residual lands on a
        trajectory it was never fitted to, and the benchmark reports "RL did not help".

        precompute.py writes meta.json into the observation cache and train.py copies it
        next to the model, so this can be checked rather than remembered.
        """
        import json

        if not meta_path.exists():
            print(f"[rl_agent] no {meta_path.name}; cannot verify this is the checkpoint "
                  f"the policy was trained against. Re-run rl/precompute.py to record it.")
            return

        meta = json.loads(meta_path.read_text())
        recorded, actual = meta.get("checkpoint"), str(self._checkpoint_path)
        recorded_bytes = meta.get("checkpoint_bytes")
        actual_bytes = Path(actual).stat().st_size if Path(actual).exists() else None

        # Size is the test, not the path: a moved or renamed but identical checkpoint is
        # fine, a genuinely different file is not.
        if recorded_bytes is not None and actual_bytes != recorded_bytes:
            raise ValueError(
                f"This policy was trained on observations from {recorded} "
                f"({recorded_bytes} bytes), but agent.checkpoint_path is {actual} "
                f"({actual_bytes} bytes). The residual is defined relative to the "
                f"trajectory that checkpoint emits -- evaluating on a different one is "
                f"silently meaningless. Point agent.checkpoint_path at the checkpoint "
                f"rl/precompute.py used, or re-run precompute + train for this one."
            )
        if recorded != actual:
            print(f"[rl_agent] checkpoint path differs from the recorded {recorded}, but "
                  f"the file is the same size -- treating it as the same weights.")

    # ------------------------------------------------------------------- inference

    def _build_observation(self, output: Dict[str, torch.Tensor],
                           features: Dict[str, torch.Tensor],
                           best: torch.Tensor, idx: torch.Tensor) -> Dict[str, np.ndarray]:
        """Rebuild precompute.encode_batch's observation live, field for field.

        Any divergence from precompute.py here is silent, so the mapping is spelled out
        rather than inferred:

          scene_latent  bev_feature (B, cam, patch, D) mean-pooled over patches
          ego_status    the full (B, 4, 11) history, flattened
          base_traj     the proposal RAP's own scorer ranked first
          base_scores   sigmoid(pred_logit) for that proposal
        """
        # .half() is not a typo. precompute.py stores this fp16 and RLDataset promotes it
        # back to fp32, so the policy -- and the running mean/std -- only ever saw
        # fp16-rounded latents. Rounding here too keeps eval on the training distribution.
        scene_latent = output["bev_feature"].mean(dim=2).half().float()

        return {
            "scene_latent": scene_latent.flatten(1).cpu().numpy().astype(np.float32),
            "ego_status": features["ego_status"].float().flatten(1).cpu().numpy().astype(np.float32),
            "base_traj": output["trajectory"][idx, best].float().flatten(1).cpu().numpy().astype(np.float32),
            "base_scores": torch.sigmoid(output["pred_logit"])[idx, best].float().cpu().numpy().astype(np.float32),
        }

    def compute_trajectory(self, agent_input) -> Trajectory:
        """Two deltas from the devkit's AbstractAgent, both matching its RAPAgent.

        Overridden rather than editing the devkit's shared base class, which every other
        agent there also uses:

          * ``.to(self.device)`` -- RAPModel lives on the GPU, and the devkit's base
            leaves features on the CPU.
          * ``[:8]`` -- truncate to the horizon the PDM scorer actually samples; see
            _SCORER_NUM_POSES.

        Keeping these identical to RAPAgent is what makes disable_residual=True a true
        baseline: the two columns then differ only by the residual.
        """
        self.eval()
        features: Dict[str, torch.Tensor] = {}
        for builder in self.get_feature_builders():
            features.update(builder.compute_features(agent_input))

        features = {k: v.unsqueeze(0).to(self.device) for k, v in features.items()}

        with torch.no_grad():
            predictions = self.forward(features)
            poses = predictions["trajectory"].squeeze(0).cpu().numpy()[:_SCORER_NUM_POSES]

        return Trajectory(poses)

    def forward(self, features: Dict[str, torch.Tensor], targets=None,
                return_score: bool = False) -> Dict[str, torch.Tensor]:
        """RAP's selected trajectory, plus the policy's residual.

        Returns {"trajectory": (B, P, 3)} over the FULL horizon; compute_trajectory above
        truncates to what the scorer samples. Signature matches RAPAgent.forward so the
        two are interchangeable, though targets/return_score are inference no-ops here.
        """
        if not self._disable_residual and self._policy is None:
            raise RuntimeError(
                "initialize() has not run -- no policy is loaded. run_pdm_score.py calls "
                "it for you; a direct caller must."
            )
        # return_score=True gives the proposals *and* their scores. RAP's own
        # return_score=False branch picks argmax of the same tensor, so selecting here
        # reproduces plain RAP exactly -- and reproduces precompute.py's base_traj.
        output = self._rap_model(features, None, return_score=True)

        proposals = output["trajectory"]            # (B, num_proposals, P, 3)
        best = output["score"].argmax(dim=1)
        idx = torch.arange(proposals.shape[0], device=proposals.device)
        base_traj = proposals[idx, best]            # (B, P, 3)

        if self._disable_residual:
            return {"trajectory": base_traj}

        obs = self._build_observation(output, features, best, idx)
        # normalize_obs applies the training run's running mean/std and clip_obs. It does
        # not update them -- training=False above, and this path never calls step_wait.
        obs = self._obs_norm.normalize_obs(obs)
        action, _ = self._policy.predict(obs, deterministic=self._deterministic)

        base_np = base_traj.float().cpu().numpy()
        trajectory = np.stack([
            apply_residual(base_np[b], action[b], self._delta_scale)
            for b in range(base_np.shape[0])
        ])
        return {
            "trajectory": torch.as_tensor(trajectory, dtype=torch.float32, device=proposals.device)
        }
