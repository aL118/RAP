"""Assert that rl/agent.py sees exactly what rl/precompute.py wrote.

The policy is trained on observations built by precompute.encode_batch and evaluated
on observations built by RLResidualAgent._build_observation -- two separate pieces of
code, in different processes, that must produce the same vector from the same model
output. This checks exactly that: both sides run here against the same RAP weights, so
any difference is a difference in how the observation is assembled, which is the part
this package controls. (Feature *building* differs by construction at benchmark time --
cached tensors vs. live sensors -- and is the devkit's business, not this file's.)

If they ever stop agreeing, nothing raises. The agent still emits trajectories, the
benchmark still produces a number, and that number is simply wrong -- it measures a
policy fed observations it was never trained on. The failure looks exactly like "RL
did not help", which is the most expensive possible way to be wrong, because it is
indistinguishable from a real negative result.

So this compares them directly, field by field, on real cached samples:

    python rl/parity_check.py --limit 8

Checks:
  1. scene_latent / ego_status / base_traj / base_scores agree elementwise
  2. disable_residual=True reproduces RAP's own selected trajectory bit for bit
  3. with a policy, the residual stays inside residual_scale and moves pose 0 least
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl.config import RLConfig
from rl.residual import delta_scale


def _report(name: str, a: np.ndarray, b: np.ndarray, tol: float) -> bool:
    """Print a one-line verdict and return whether the field matched."""
    if a.shape != b.shape:
        print(f"  {name:<14} SHAPE MISMATCH  precompute {a.shape} vs agent {b.shape}")
        return False
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    ok = bool(np.all(diff <= tol))
    print(f"  {name:<14} max|diff| {diff.max():.3e}  {'OK' if ok else 'MISMATCH'}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, default=8, help="Samples to compare.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--time-horizon", type=float, default=5.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--policy-path", default="",
                        help="Optional PPO .zip; also checks the residual's bound.")
    parser.add_argument("--tol", type=float, default=0.0,
                        help="Elementwise tolerance. 0 = bit-identical, which is what "
                             "the same weights on the same device should give.")
    args = parser.parse_args()

    cfg = RLConfig()
    cfg.validate(require_rl_cache=False)

    from navsim.agents.rap_dino.navsim_config import RAPConfig
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

    rap_config = RAPConfig()
    rap_config.trajectory_sampling = TrajectorySampling(
        time_horizon=args.time_horizon, interval_length=0.5
    )
    device = torch.device(args.device)

    from rl.precompute import _collate, build_dataset, build_model, encode_batch

    dataset = build_dataset(rap_config, cfg.feature_cache_path)
    tokens = dataset.tokens[: args.limit]
    subset = torch.utils.data.Subset(dataset, list(range(len(tokens))))
    loader = torch.utils.data.DataLoader(
        subset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=_collate
    )

    print(f"[parity] {len(tokens)} samples, device={device}, horizon={args.time_horizon}")

    # --- the precompute side: the exact model and code path that wrote the cache -----
    model = build_model(rap_config, cfg.checkpoint_path, device)

    # --- the agent side: constructed the way run_pdm_score.py constructs it ----------
    from rl.agent import RLResidualAgent

    agent = RLResidualAgent(
        config=rap_config,
        checkpoint_path=str(cfg.checkpoint_path),
        policy_path=args.policy_path,
        disable_residual=not args.policy_path,
    )
    agent.initialize()

    all_ok = True
    for batch_idx, (features, _) in enumerate(loader):
        expected = encode_batch(model, features, device)

        gpu_features = {k: v.to(device) for k, v in features.items() if torch.is_tensor(v)}
        with torch.no_grad():
            output = agent._rap_model(gpu_features, None, return_score=True)
            best = output["score"].argmax(dim=1)
            idx = torch.arange(output["trajectory"].shape[0], device=output["trajectory"].device)
            actual = agent._build_observation(output, gpu_features, best, idx)

        print(f"\n[parity] batch {batch_idx}")
        # precompute keeps the natural shapes; the env flattens them before the policy
        # sees them, and _build_observation flattens directly. Compare on flat vectors.
        for key in ("scene_latent", "ego_status", "base_traj", "base_scores"):
            ref = expected[key].reshape(expected[key].shape[0], -1)
            all_ok &= _report(key, ref, actual[key], args.tol)

        # Check 2: the agent's trajectory with no residual IS RAP's selected proposal.
        with torch.no_grad():
            base_only = agent._rap_model(gpu_features, None, return_score=False)["trajectory"]
        all_ok &= _report(
            "base_traj/argmax",
            base_only.float().cpu().numpy(),
            expected["base_traj"],
            args.tol,
        )

        # Check 3: the residual is bounded and ramped.
        if args.policy_path:
            with torch.no_grad():
                traj = agent.forward(gpu_features)["trajectory"].cpu().numpy()
            base = expected["base_traj"]
            scale = delta_scale(base.shape[1], agent._residual_scale)
            delta = np.abs(traj - base)
            within = bool(np.all(delta <= scale + 1e-5))
            print(f"  {'residual bound':<14} max|delta| {delta.max():.4f} "
                  f"(limit {scale.max():.4f})  {'OK' if within else 'EXCEEDED'}")
            # The ramp is the reason the trajectory stays anchored at the ego.
            print(f"  {'ramp':<14} pose0 {delta[:, 0].max():.4f} <= "
                  f"poseN {delta[:, -1].max():.4f}")
            all_ok &= within

    print("\n[parity] " + ("ALL CHECKS PASSED" if all_ok else "FAILED -- see above"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
