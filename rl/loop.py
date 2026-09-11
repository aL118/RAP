"""The whole thing: collect, train, repeat.

    round 1   sample from the pretrained planner + the human trajectory
              -> score everything -> buffer -> train the scorer on the buffer
    round r   sample from round r-1's model
              -> score -> append to the buffer -> retrain on ALL of it

Resumable at round granularity, which matters because a round is hours: the buffer
file and the checkpoint are each written once, atomically, at the end of their
half, so a killed job restarts at the first half that did not finish. Nothing is
recomputed that already landed on disk unless `--force` says so.
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl import buffer as buffer_module
from rl.collect import collect_round
from rl.config import RLConfig
from rl.train import train_round


def run(config: RLConfig, device: torch.device, force: bool = False, first_round: int = 1) -> None:
    config.output_path.mkdir(parents=True, exist_ok=True)
    print(f"[loop] {config.num_rounds} rounds -> {config.output_path}")
    if config.log_tensorboard:
        print(f"[loop] curves: tensorboard --logdir {config.output_path / 'tb'}")

    for round_index in range(first_round, config.num_rounds + 1):
        collected = round_index in buffer_module.rounds_present(config.buffer_path)
        if collected and not force:
            print(f"[loop] round {round_index}: buffer exists, skipping collection")
        else:
            collect_round(config, round_index, device)

        checkpoint = config.round_checkpoint(round_index)
        if checkpoint.is_file() and not force:
            print(f"[loop] round {round_index}: {checkpoint} exists, skipping training")
        else:
            train_round(config, round_index, device)

    print(f"[loop] done. Final model: {config.round_checkpoint(config.num_rounds)}")
    print(f"[loop] per-batch scalars: {config.output_path / 'logs' / 'train.jsonl'}")
    print("[loop] measure it against the pretrained planner with:")
    print(f"       python rl/eval.py --baseline --round {config.num_rounds} "
          f"--experiment-name {config.experiment_name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--num-rounds", type=int, default=None)
    parser.add_argument("--first-round", type=int, default=1,
                        help="Start here instead of round 1. Rounds before it must "
                             "already have a buffer file and a checkpoint.")
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--samples-per-scene", type=int, default=None)
    parser.add_argument("--regular-ratio", type=float, default=None)
    parser.add_argument("--epochs-per-round", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--score-workers", type=int, default=None)
    parser.add_argument("--restart-from-pretrained", action="store_true",
                        help="Re-fit the pretrained checkpoint on the whole buffer "
                             "each round instead of continuing from the previous one.")
    parser.add_argument("--no-tensorboard", action="store_true",
                        help="Skip the event files under <run>/tb/. The JSONL in "
                             "<run>/logs/ is written either way.")
    parser.add_argument("--force", action="store_true",
                        help="Redo rounds that already have output on disk.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = RLConfig()
    for field in ("num_rounds", "experiment_name", "samples_per_scene", "regular_ratio",
                  "epochs_per_round", "batch_size", "score_workers"):
        value = getattr(args, field)
        if value is not None:
            setattr(config, field, value)
    if args.restart_from_pretrained:
        config.restart_from_pretrained = True
    if args.no_tensorboard:
        config.log_tensorboard = False
    config.validate()

    run(config, torch.device(args.device), args.force, args.first_round)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
