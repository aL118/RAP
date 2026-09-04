"""Distil the RAP feature cache into compact RL observations. Run once, on a GPU.

Why this step exists
--------------------
A cached RAP sample carries 4 camera images of 3x448x768 -- 16.5 MB of float32.
An on-policy rollout buffer of even a few thousand of those is hundreds of GB, so
feeding raw images to stable-baselines3 is a non-starter. The DINOv3 backbone is
frozen in RAP training anyway, so running it inside the RL loop would recompute an
identical tensor every epoch for nothing.

So we run the pretrained RAP model over the cache exactly once and keep, per token:

  scene_latent  (4, 1280) fp16  frozen ImgEncoder output, mean-pooled over the
                                28x48 patch grid of each of the 4 cameras
  ego_status    (4, 11)   fp32  ego pose/velocity/acceleration/command history
  base_traj     (P, 3)    fp32  RAP's own selected trajectory -- the residual base
  base_scores   (6,)      fp32  RAP's *predicted* PDM sub-scores for that trajectory
  base_pdm      (6,)      fp32  the TRUE PDM sub-scores of base_traj, from the metric
                                cache -- the per-scene baseline the reward is centred on

That is ~21 KB per token instead of 16.5 MB: the whole 14675-token navtrain subset
fits in ~300 MB of RAM, and the env becomes pure array indexing plus PDM scoring.

Output is sharded .npz files so a run is resumable at shard granularity.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl.config import RLConfig


def build_model(config, checkpoint_path: Path, device: torch.device):
    """Instantiate RAPModel and load pretrained weights.

    RAPAgent is bypassed on purpose: its constructor builds metric-cache loaders,
    loss modules and (optionally) a ray worker pool, none of which inference needs.
    """
    from navsim.agents.rap_dino.rap_model import RAPModel

    model = RAPModel(config)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    # Lightning saves the agent as `agent._rap_model.*`; RAPModel wants bare keys.
    state_dict = {
        k.replace("agent._rap_model.", "").replace("_rap_model.", ""): v
        for k, v in state_dict.items()
    }
    # strict=False is needed for the training-only domain_classifier, but it also means a
    # horizon mismatch would pass silently: init_feature is (num_poses * proposal_num,
    # d_model), so loading a 10-pose checkpoint into an 8-pose model would skip that key
    # and leave a randomly initialised embedding driving every trajectory. Check it.
    expected = model.init_feature.weight.shape
    found = state_dict.get("init_feature.weight")
    if found is not None and tuple(found.shape) != tuple(expected):
        raise ValueError(
            f"Checkpoint/config mismatch: init_feature is {tuple(found.shape)} in "
            f"{checkpoint_path.name} but {tuple(expected)} in this config. That is "
            f"{found.shape[0] // config.proposal_num} poses vs "
            f"{expected[0] // config.proposal_num} -- pass the --time-horizon the "
            f"checkpoint was trained with (RAP_DINO_navsimv2.ckpt used 5)."
        )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    # The trajectory head is shared across `ref_num` refiners via one nn.Module
    # instance, so a handful of duplicate-path keys are expected to be unexpected.
    if missing:
        print(f"[precompute] {len(missing)} missing keys, first few: {missing[:5]}")
    if unexpected:
        print(f"[precompute] {len(unexpected)} unexpected keys, first few: {unexpected[:5]}")

    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad = False
    return model


def build_dataset(config, feature_cache_path: Path):
    """The same CacheOnlyDataset the supervised trainer uses, same builders."""
    from navsim.planning.training.dataset import CacheOnlyDataset
    from navsim.agents.rap_dino.rap_features import RAPFeatureBuilder, RAPTargetBuilder

    return CacheOnlyDataset(
        cache_path=str(feature_cache_path),
        feature_builders=[RAPFeatureBuilder(config)],
        target_builders=[RAPTargetBuilder(config)],
    )


@torch.no_grad()
def encode_batch(model, features: Dict[str, torch.Tensor], device: torch.device):
    """Run RAP once and pull out the four arrays we keep per sample."""
    features = {k: v.to(device) for k, v in features.items() if torch.is_tensor(v)}

    output = model(features, targets=None, return_score=True)

    proposals = output["trajectory"]            # (B, num_proposals, P, 3)
    proposal_scores = output["score"]           # (B, num_proposals)
    pred_logit = output["pred_logit"]           # (B, num_proposals, 6)
    bev = output["bev_feature"]                 # (B, num_cam, num_patch, D)

    best = proposal_scores.argmax(dim=1)
    idx = torch.arange(proposals.shape[0], device=proposals.device)

    return {
        # Mean over the patch dimension: one 1280-d descriptor per camera. Spatial
        # detail is dropped deliberately -- keeping the 28x48 grid would be 1344x
        # larger and would swamp an MLP policy's first layer.
        "scene_latent": bev.mean(dim=2).half().cpu().numpy(),
        "ego_status": features["ego_status"].float().cpu().numpy(),
        "base_traj": proposals[idx, best].float().cpu().numpy(),
        "base_scores": torch.sigmoid(pred_logit)[idx, best].float().cpu().numpy(),
    }


def score_base_trajectories(tokens, base_traj, metric_cache_path, num_workers):
    """True PDM sub-scores for every base trajectory, in parallel.

    Done here, once, rather than in the env: the baseline never changes, so paying
    ~0.2 s per scene during a rollout would double the cost of every single step.

    Processes, not threads -- the PDM scorer is a stateful module-level singleton
    (see rl/reward.py).
    """
    import multiprocessing
    from navsim.common.dataloader import MetricCacheLoader
    # Imported by name so spawn workers pickle rl.reward.score_row -- a light module --
    # rather than __main__._score_one, which would re-import this whole script.
    from rl.reward import score_row

    paths = MetricCacheLoader(Path(metric_cache_path)).metric_cache_paths

    # A token with no metric cache gets a NaN row rather than aborting the shard --
    # a multi-hour run should not die on one missing scenario. rl/env.py drops rows
    # with no finite baseline when it builds its index.
    scorable = [i for i, t in enumerate(tokens) if t in paths]
    if len(scorable) < len(tokens):
        print(f"[precompute] {len(tokens) - len(scorable)} of {len(tokens)} tokens have no "
              f"metric cache; their baselines are NaN and rl/env.py will skip them")

    jobs = [(paths[tokens[i]], base_traj[i]) for i in scorable]
    # "spawn", not the Linux default "fork". By the time this runs, the RAP model has
    # already run on the GPU, so the parent holds an initialised CUDA context -- and
    # forking one is unsafe. Observed directly (job 7435020): all 8 workers sat in S
    # state at 0.0% CPU with the parent's 2.7 GB image each, never scoring a single
    # trajectory, and the job had to be cancelled. Nothing raised; it simply hung.
    # spawn gives each worker a clean interpreter, at the cost of re-importing the
    # nuplan stack once per worker. train.py's SubprocVecEnv uses spawn for the same
    # reason.
    context = multiprocessing.get_context("spawn")
    with context.Pool(num_workers) as pool:
        scored = list(tqdm(pool.imap(score_row, jobs, chunksize=8),
                           total=len(jobs), desc="scoring base trajectories"))

    rows = np.full((len(tokens), len(scored[0]) if scored else 6), np.nan, np.float32)
    for slot, row in zip(scorable, scored):
        rows[slot] = row
    return rows


def _shard_is_readable(path: Path) -> bool:
    """True if an existing shard can actually be opened and carries its arrays.

    np.load on a .npz is lazy about member data, so touching one array is what forces
    the zip central directory and that member to be read.
    """
    try:
        with np.load(path, allow_pickle=True) as data:
            required = ("tokens", "scene_latent", "ego_status", "base_traj",
                        "base_scores", "base_pdm")
            if any(key not in data.files for key in required):
                return False
            return len(data["base_traj"]) == len(data["tokens"])
    except Exception:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Images per forward = 4x this. 8 fits a 24 GB A5000 at inference.")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--shard-size", type=int, default=2000,
                        help="Tokens per output .npz; a run resumes at this granularity.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Encode only the first N tokens (smoke tests).")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--score-workers", type=int, default=12,
                        help="Processes used to PDM-score the base trajectories (CPU bound).")
    parser.add_argument("--time-horizon", type=float, default=5.0,
                        help="Must match the checkpoint. RAP_DINO_navsimv2.ckpt used 5 s.")
    parser.add_argument("--rl-cache-path", type=Path, default=None,
                        help="Override the output directory. Use it for smoke runs: a "
                             "partially populated cache/rl_obs would be silently reused by "
                             "a later full run, which skips shards that already exist.")
    args = parser.parse_args()

    cfg = RLConfig()
    if args.rl_cache_path is not None:
        cfg.rl_cache_path = args.rl_cache_path
    cfg.validate(require_rl_cache=False)
    if not cfg.checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing pretrained checkpoint {cfg.checkpoint_path}")

    from navsim.agents.rap_dino.navsim_config import RAPConfig
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

    rap_config = RAPConfig()
    rap_config.trajectory_sampling = TrajectorySampling(
        time_horizon=args.time_horizon, interval_length=0.5
    )

    device = torch.device(args.device)
    print(f"[precompute] device={device} checkpoint={cfg.checkpoint_path}")

    dataset = build_dataset(rap_config, cfg.feature_cache_path)
    tokens: List[str] = dataset.tokens
    if args.limit:
        tokens = tokens[: args.limit]
    print(f"[precompute] {len(tokens)} tokens in {cfg.feature_cache_path}")

    model = build_model(rap_config, cfg.checkpoint_path, device)
    cfg.rl_cache_path.mkdir(parents=True, exist_ok=True)

    shards = [tokens[i:i + args.shard_size] for i in range(0, len(tokens), args.shard_size)]
    for shard_idx, shard_tokens in enumerate(shards):
        out_path = cfg.rl_cache_path / f"shard_{shard_idx:04d}.npz"
        if out_path.exists():
            # Verify, don't trust. This run is resumable precisely because it gets
            # killed -- a 4 h wall clock over ~34 min/shard means the last shard is
            # routinely cut off mid-write -- and a truncated .npz still "exists". Left
            # unchecked, the resume would skip it and every later stage would train on
            # a cache with a silently missing chunk.
            if _shard_is_readable(out_path):
                print(f"[precompute] shard {shard_idx} exists, skipping")
                continue
            print(f"[precompute] shard {shard_idx} is unreadable (truncated by an "
                  f"interrupted run?) -- regenerating")
            out_path.unlink()

        # A Subset over the parent dataset keeps CacheOnlyDataset's loading path intact.
        token_to_index = {t: i for i, t in enumerate(dataset.tokens)}
        subset = torch.utils.data.Subset(dataset, [token_to_index[t] for t in shard_tokens])
        loader = DataLoader(
            subset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=_collate,
        )

        buffers: Dict[str, List[np.ndarray]] = {}
        for features, _targets in tqdm(loader, desc=f"shard {shard_idx}/{len(shards) - 1}"):
            for key, value in encode_batch(model, features, device).items():
                buffers.setdefault(key, []).append(value)

        arrays = {k: np.concatenate(v, axis=0) for k, v in buffers.items()}
        arrays["base_pdm"] = score_base_trajectories(
            shard_tokens, arrays["base_traj"], cfg.metric_cache_path, args.score_workers
        )
        base_pdm = arrays["base_pdm"]
        print(f"[precompute] shard {shard_idx} base PDMS {np.nanmean(base_pdm[:, -1]):.4f} "
              f"collision rate {1.0 - np.nanmean(base_pdm[:, 0]):.4f}")
        # Write then rename: os.replace is atomic within a filesystem, so a shard file
        # either does not exist or is complete. Without this, being killed inside
        # savez_compressed leaves a half-written file that looks finished.
        # The tmp name has to satisfy two constraints at once: it must end in .npz,
        # because savez_compressed silently appends .npz to any path that does not
        # (so a ".npz.tmp" target lands at ".npz.tmp.npz" and the replace below fails
        # with FileNotFoundError), and it must not match the shard_*.npz glob that
        # rl/env.py:RLDataset uses, or a run killed mid-write leaves a partial shard
        # that training picks up as real.
        tmp_path = out_path.with_name(f".tmp_{out_path.name}")
        np.savez_compressed(tmp_path, tokens=np.array(shard_tokens, dtype=object), **arrays)
        os.replace(tmp_path, out_path)
        shapes = {k: v.shape for k, v in arrays.items()}
        print(f"[precompute] wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB) {shapes}")

    # Record which model produced these observations. The policy's residual is defined
    # relative to the trajectory THIS checkpoint emits, so evaluating it on top of a
    # different one is silently wrong -- the base moves under the policy and the
    # benchmark just reports that RL did not help. train.py copies this next to the
    # saved model and rl/agent.py checks it. See _write_meta.
    _write_meta(cfg, args)
    print(f"[precompute] done -> {cfg.rl_cache_path}")
    return 0


def _write_meta(cfg, args) -> None:
    """Provenance for the observation cache: what made it, at what horizon."""
    import json

    meta = {
        "checkpoint": str(cfg.checkpoint_path),
        # Size, not a hash: hashing 3.9 GB on every run costs more than it is worth, and
        # size alone reliably separates two different checkpoints. A moved-but-identical
        # file is tolerated deliberately; a different file is not.
        "checkpoint_bytes": cfg.checkpoint_path.stat().st_size,
        "time_horizon": args.time_horizon,
        "feature_cache": str(cfg.feature_cache_path),
    }
    meta_path = cfg.rl_cache_path / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[precompute] wrote {meta_path}: {meta}")


def _collate(samples):
    """Stack feature tensors; drop targets. `token` is a str and would break
    default_collate's tensor path, and RL never reads the ground-truth trajectory."""
    features = {
        key: torch.stack([s[0][key] for s in samples])
        for key in samples[0][0]
        if torch.is_tensor(samples[0][0][key])
    }
    return features, None


if __name__ == "__main__":
    raise SystemExit(main())
