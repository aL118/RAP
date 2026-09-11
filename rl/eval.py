"""Did the round help? Three numbers, on scenes no round ever trained on.

The training loss says the scorer fits its buffer. That is not the question. The
question is whether the model now *picks* better trajectories, so this evaluates
the thing inference actually does: propose, rank, take the argmax, and score what
came out with the true scorer.

  selected   true PDMS of the proposal the model's own scorer ranked first.
             The headline. It moves for two reasons at once -- better proposals
             and better ranking of them -- which is fine, because that is also the
             only thing that moves at inference.
  pool_best  true PDMS of the best proposal in a uniform sample of the proposal
             set -- what the planner has to offer, independent of how it ranks.
             `pool_best` rising means the refiners improved. It is NOT a ceiling
             and `selected` is normally well above it: a good scorer beats a
             handful of random draws, which is the whole job. At
             `--proposals-per-scene 64` the sample is the entire proposal set and
             this becomes the exact oracle, with `selected <= pool_best` again.
  spearman   rank correlation between predicted and true score over that same
             uniform sample, averaged over scenes. The direct measure of the
             scorer, independent of how good the proposals happen to be.

Why the sample is uniform
-------------------------
Only `selected` may depend on the scorer being evaluated.
`pool_best` and `spearman` must not, or they cannot be compared across rounds --
which is the only thing either number is for.

Scoring all 64 proposals on every val scene is the most expensive thing in the
pipeline, so a subset gets truly scored. If that subset is the top-k under the
checkpoint's *own* predicted score, then a better scorer surfaces different
trajectories into it, and both `pool_best` and `spearman` move for that reason
alone: `pool_best` rises because the sampled set is better, with the proposals
unchanged, and `spearman` is measured over a differently-restricted range. Round 4
would then beat round 0 on both without a single proposal having improved.

So the budget is spent as the model's own argmax -- always scored, which is what
keeps `selected` exact -- plus a uniform draw over the proposal slots, seeded by
the scene token. The draw is over query-slot indices, which the
architecture fixes and the scorer does not touch, so every checkpoint is measured
through the same selection rule. `pool_best` and `spearman` are computed over the
uniform part alone.

The cost is that `spearman` no longer measures discrimination *near the top*,
which is the comparison inference actually makes -- most of a uniform draw is
proposals no scorer would consider. That signal cannot be had comparably without a
fixed reference model to pick the subset, so it is not reported rather than
reported misleadingly.

Run it against round 0 first. Every number here is only meaningful as a delta
against the pretrained planner measured through this same code path, and none of
them is comparable to a published NAVSIM number: the split is a held-out slice of
the training sources, and half of it is CARE footage that has no drivable-area
label at all.
"""

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl import data as data_module
from rl.collect import score_jobs
from rl.config import RLConfig
from rl.scoring import PDM_INDEX
from rl.tracking import RunLog


def _spearman(predicted: np.ndarray, true: np.ndarray) -> float:
    """Rank correlation of two 1-D arrays, ties averaged.

    Written out rather than pulled from scipy: this package's environment is the
    training env, scipy is present but its import costs a second in every worker,
    and this is twenty lines of argsort.
    """
    def rank(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values)
        ranks = np.empty(len(values), dtype=np.float64)
        ranks[order] = np.arange(len(values), dtype=np.float64)
        # Average the ranks of tied values, which matters here: most proposals in
        # a safe scene score identically at 1.0 on four of the six sub-scores.
        unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
        for index in np.flatnonzero(counts > 1):
            tied = inverse == index
            ranks[tied] = ranks[tied].mean()
        return ranks

    if len(predicted) < 2:
        return float("nan")
    a, b = rank(predicted), rank(true)
    a = a - a.mean()
    b = b - b.mean()
    denominator = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denominator) if denominator > 0 else float("nan")


def _scored_indices(
    predicted: np.ndarray, budget: int, token: str
) -> Tuple[np.ndarray, int]:
    """Which proposal slots to truly score for one scene.

    Index 0 of the result is the model's own argmax, so `selected` is exact no
    matter how small the budget is. The rest is a uniform draw over the proposal
    slots, seeded by the token: the same slots for every checkpoint, so
    `pool_best` and `spearman` measured over them are comparable across rounds.
    Slot indices are fixed by the architecture and untouched by the scorer, which
    is what makes a uniform draw over them scorer-independent -- see the module
    docstring.

    :return: ``(indices, num_uniform)`` -- ``indices[1:1 + num_uniform]`` is the
        uniform part, which is what the comparable metrics are computed on.
    """
    num_proposals = len(predicted)
    budget = max(1, min(budget, num_proposals))

    best = int(np.argmax(predicted))

    # Seeded by the token, not by a run-level counter: a scene draws the same
    # slots whichever order the loader happened to visit it in, and whichever
    # checkpoint is being evaluated.
    rng = np.random.default_rng(
        int.from_bytes(hashlib.md5(token.encode()).digest()[:8], "little")
    )
    num_uniform = budget - 1
    uniform = rng.permutation(num_proposals)[:num_uniform]

    # The draw is over *all* slots, deliberately including `best`. Excluding it
    # would be the obvious thing and is wrong: the pool would then depend on which
    # slot the scorer ranked first, so two checkpoints would draw different
    # uniform parts on the same scene and the whole point of seeding would be
    # lost. The cost is that `best` lands in the uniform part about
    # (budget - 1) / num_proposals of the time and is scored twice -- one extra
    # trajectory in an already-batched call. Row 0 is still the argmax and rows
    # 1.. are still a uniform sample; nothing double-counts, because `selected`
    # reads only row 0 and the comparable metrics read only rows 1...
    return np.concatenate([[best], uniform]).astype(np.int64), num_uniform


def evaluate(
    config: RLConfig,
    checkpoint: Path,
    device: torch.device,
    limit: int = 0,
    proposals_per_scene: int = 16,
) -> Dict[str, Dict[str, float]]:
    """Propose, rank and truly score, on the held-out split of every source.

    :param proposals_per_scene: how many of the model's proposals to truly score
        per scene -- the argmax plus `proposals_per_scene - 1` drawn uniformly.
        Scoring all 64 on every val scene is the most expensive thing in the
        pipeline. Raising this tightens `pool_best` and `spearman` (both are
        estimates over the uniform part) and does nothing to `selected`, which is
        exact at any budget. At 64 the uniform part is the whole proposal set and
        `pool_best` stops being an estimate.
    """
    from rl.model import RAPScorer

    sources = data_module.build_sources(config)
    model = RAPScorer.from_checkpoint(config, checkpoint, device)
    model.eval()

    results: Dict[str, Dict[str, float]] = {}
    for name, source in sources.items():
        tokens = source.tokens
        if name == "navtrain":
            tokens = source.scorable_tokens()
        _, val_tokens = data_module.split_tokens(tokens, config.val_fraction)
        if limit:
            val_tokens = val_tokens[:limit]
        if not val_tokens:
            continue

        batches = data_module.loader(source, val_tokens, config.batch_size, config.num_workers)
        jobs, predicted_by_token, uniform_count = [], {}, {}

        with torch.no_grad():
            for batch_tokens, features in tqdm(batches, desc=f"proposing ({name})"):
                features = {
                    key: value.to(device)
                    for key, value in features.items()
                    if torch.is_tensor(value)
                }
                proposals, predicted = model.propose(features)
                proposals = proposals.float().cpu().numpy()
                predicted = predicted.float().cpu().numpy()

                for index, token in enumerate(batch_tokens):
                    order, num_uniform = _scored_indices(
                        predicted[index], proposals_per_scene, token
                    )
                    jobs.append(
                        (token, source.name, proposals[index][order].astype(np.float32),
                         source.scoring_payload(token))
                    )
                    predicted_by_token[token] = predicted[index][order]
                    uniform_count[token] = num_uniform

        scored = score_jobs(jobs, config.score_workers)

        selected, pool_best, correlations = [], [], []
        for token, _source, _trajectories, _payload in jobs:
            if token not in scored:
                continue
            true = scored[token][0][:, PDM_INDEX]
            if not np.isfinite(true).any():
                continue

            # Row 0 is the model's own argmax, rows 1.. the uniform draw. The two
            # are kept apart on purpose: `selected` is the only number allowed to
            # depend on the scorer being evaluated.
            if np.isfinite(true[0]):
                selected.append(true[0])

            sample = true[1 : 1 + uniform_count[token]]
            predicted_sample = predicted_by_token[token][1 : 1 + uniform_count[token]]
            finite = np.isfinite(sample)
            if finite.any():
                pool_best.append(np.nanmax(sample))
                correlations.append(_spearman(predicted_sample[finite], sample[finite]))

        results[name] = {
            "scenes": float(len(selected)),  # scenes whose argmax scored finitely
            "selected": float(np.mean(selected)) if selected else float("nan"),
            "pool_best": float(np.mean(pool_best)) if pool_best else float("nan"),
            "spearman": float(np.nanmean(correlations)) if correlations else float("nan"),
        }

    return results


def _report(
    title: str,
    results: Dict[str, Dict[str, float]],
    log: Optional[RunLog] = None,
    round_index: Optional[int] = None,
    checkpoint: Optional[Path] = None,
) -> None:
    """Print the table, and put the same three numbers on the run's timeline.

    Stepped by round, not by the training step counter, and written to its own
    sink so TensorBoard shows it as a separate run: these are the numbers that
    decide whether a round helped, and they are worth reading against each other
    across rounds rather than against a batch axis they have no place on.
    """
    print(f"\n{title}")
    print(f"  {'source':<10}{'scenes':>8}{'selected':>11}{'pool_best':>11}{'spearman':>11}")
    for name, values in results.items():
        print(
            f"  {name:<10}{int(values['scenes']):>8}{values['selected']:>11.4f}"
            f"{values['pool_best']:>11.4f}{values['spearman']:>11.4f}"
        )
        if log is not None and round_index is not None:
            log.record(values, round_index, f"{name}/", kind="eval",
                       round=round_index, source=name,
                       checkpoint=str(checkpoint) if checkpoint else None)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--round", type=int, default=None,
                        help="Evaluate this round's checkpoint. 0 is the pretrained model.")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Evaluate an explicit checkpoint instead.")
    parser.add_argument("--baseline", action="store_true",
                        help="Also evaluate round 0, so the delta comes from one run.")
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--limit", type=int, default=0, help="Cap val scenes per source.")
    parser.add_argument("--proposals-per-scene", type=int, default=16,
                        help="Proposals truly scored per scene: the model's argmax "
                             "plus this many minus one drawn uniformly.")
    parser.add_argument("--regular-ratio", type=float, default=None,
                        help="0 skips the navtrain source entirely, which is what "
                             "an experiment collected with --regular-ratio 0 needs "
                             "-- otherwise this asks for a metric cache the run "
                             "never used.")
    parser.add_argument("--score-workers", type=int, default=None)
    parser.add_argument("--no-tensorboard", action="store_true",
                        help="Skip the event files under <run>/tb/. The JSONL in "
                             "<run>/logs/ is written either way.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = RLConfig()
    if args.experiment_name:
        config.experiment_name = args.experiment_name
    if args.score_workers is not None:
        config.score_workers = args.score_workers
    if args.regular_ratio is not None:
        config.regular_ratio = args.regular_ratio
    if args.no_tensorboard:
        config.log_tensorboard = False
    config.validate()

    device = torch.device(args.device)
    log = RunLog(config.output_path, "eval", config.log_tensorboard)

    if args.baseline:
        _report("round 0 (pretrained)", evaluate(
            config, config.round_checkpoint(0), device, args.limit, args.proposals_per_scene
        ), log, 0, config.round_checkpoint(0))

    # The round the logged point belongs to. With an explicit --checkpoint this is
    # a guess, which is why every record also carries the checkpoint path.
    round_index = args.round if args.round is not None else config.num_rounds
    checkpoint = args.checkpoint or config.round_checkpoint(round_index)
    _report(str(checkpoint), evaluate(
        config, checkpoint, device, args.limit, args.proposals_per_scene
    ), log, round_index, checkpoint)

    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
