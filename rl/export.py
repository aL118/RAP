"""Turn a round checkpoint into one anything else can load.

Round checkpoints are deltas by default: they hold only the tensors that round
could change and name the pretrained base they sit on (see `RAPScorer.save`). That
is right for this pipeline, which always has the base to hand, and wrong for
everything else -- the devkit's evaluation loads a checkpoint with no idea that a
base exists, and `strict=False` in every loader on that path means a delta would
be accepted in silence with a randomly initialised DINOv3 tower behind it.

So this merges the two and writes a self-contained file, in the Lightning key
layout (`agent._rap_model.*`) that `RAPAgent.initialize` expects, so the result
drops into the devkit the same way `weights/RAP_DINO_navsimv2.ckpt` does.

    python rl/export.py --round 4 --out weights/rap_round4.ckpt
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl.config import RLConfig


def export(config: RLConfig, checkpoint_path: Path, out_path: Path) -> Path:
    """Merge a delta round checkpoint with its base and write a full one."""
    from rl.model import RAPScorer

    # Building the model and letting `from_checkpoint` do the merge means the
    # export is produced by exactly the loader the rounds themselves use -- there
    # is no second copy of the base-plus-delta logic to drift out of step.
    model = RAPScorer.from_checkpoint(config, checkpoint_path, torch.device("cpu"))

    state_dict = {
        f"agent._rap_model.{key}": value for key, value in model.rap.state_dict().items()
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(f".tmp_{out_path.name}")
    torch.save({"state_dict": state_dict}, tmp_path)
    tmp_path.replace(out_path)

    size_gb = out_path.stat().st_size / 1e9
    print(f"[export] wrote {out_path} ({size_gb:.2f} GB, {len(state_dict)} tensors)")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--round", type=int, default=None, help="Round to export.")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Export this file instead of a round's.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--experiment-name", default=None)
    args = parser.parse_args()

    config = RLConfig()
    if args.experiment_name:
        config.experiment_name = args.experiment_name

    checkpoint = args.checkpoint or config.round_checkpoint(
        args.round if args.round is not None else config.num_rounds
    )
    if not Path(checkpoint).is_file():
        raise FileNotFoundError(checkpoint)

    export(config, Path(checkpoint), args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
