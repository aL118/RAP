"""One round of collection: sample trajectories, score them, append to the buffer.

Round 1 samples from the pretrained planner. Round r samples from the model round
r-1 produced, which is the whole reason the rounds exist: as the scorer improves,
the model ranks its proposals differently, so the trajectories it would actually
drive change, and the scorer has to be right about *those* rather than about the
ones its predecessor liked. Sampling fresh every round and keeping the old rows is
DAgger's bargain -- correct on the current distribution, still calibrated on the
old one.

What "sampling 20-50 trajectories" means here
---------------------------------------------
RAP is not a stochastic policy. One forward pass emits `proposal_num` (64)
trajectories and a predicted score for each, and inference takes the argmax. So a
round does not sample repeatedly; it takes one forward pass and *subsamples* those
64, splitting the budget between:

  top-k    the highest-ranked proposals under the model's current scorer -- the
           on-policy part, the ones that would actually be driven
  uniform  drawn from the rest -- the coverage part

Both halves are load-bearing. Train only on the top-k and the scorer never learns
why the proposals it rejects are bad, which is exactly the comparison it has to get
right at inference. Train only uniformly and the rows that matter most are a
vanishing fraction of the buffer.

The human trajectory is added once, in round 1. It does not depend on the model, so
re-adding it every round would upweight it by a factor of `num_rounds` for nothing.
"""

import argparse
import multiprocessing
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl import buffer as buffer_module
from rl import data as data_module
from rl.buffer import Record
from rl.config import RLConfig
from rl.scoring import PDM_INDEX, score_job


def sample_proposal_indices(
    predicted_scores: np.ndarray, count: int, top_k_fraction: float, rng: np.random.Generator
) -> np.ndarray:
    """Which of the model's proposals to score for one scene.

    :param predicted_scores: (N,) the model's own ranking of its proposals
    :param count: how many to take, <= N
    :param top_k_fraction: share of `count` taken from the top of that ranking
    :return: (count,) proposal indices, top-k first
    """
    num_proposals = len(predicted_scores)
    count = min(count, num_proposals)
    num_top = min(int(round(count * top_k_fraction)), count)

    order = np.argsort(-predicted_scores)
    top = order[:num_top]
    remainder = order[num_top:]
    if count - num_top > 0:
        sampled = rng.choice(remainder, size=count - num_top, replace=False)
    else:
        sampled = np.empty(0, dtype=order.dtype)
    return np.concatenate([top, sampled])


def propose_for_tokens(
    model, source, tokens: Sequence[str], config: RLConfig, device: torch.device, rng
) -> Dict[str, np.ndarray]:
    """Run the model over a source's tokens, returning the sampled candidates.

    :return: token -> (K, P, 3) float32 candidate trajectories
    """
    candidates: Dict[str, np.ndarray] = {}
    batches = data_module.loader(
        source, tokens, config.batch_size, config.num_workers, shuffle=False
    )

    model.eval()
    with torch.no_grad():
        for batch_tokens, features in tqdm(batches, desc=f"proposing ({source.name})"):
            features = {
                key: value.to(device) for key, value in features.items() if torch.is_tensor(value)
            }
            proposals, predicted = model.propose(features)
            proposals = proposals.float().cpu().numpy()
            predicted = predicted.float().cpu().numpy()

            for index, token in enumerate(batch_tokens):
                chosen = sample_proposal_indices(
                    predicted[index], config.samples_per_scene, config.top_k_fraction, rng
                )
                candidates[token] = proposals[index][chosen].astype(np.float32)
    return candidates


def build_jobs(
    candidates: Dict[str, np.ndarray],
    sources: Dict[str, object],
    token_source: Dict[str, str],
    include_human: bool,
) -> List:
    """Pack one scoring job per scene, human trajectory first when included.

    Position matters: the human row is prepended, so `origins` below can label rows
    by index without carrying a parallel list through the pool.
    """
    jobs = []
    for token, trajectories in candidates.items():
        source = sources[token_source[token]]
        if include_human:
            human = np.asarray(source.human_trajectory(token), dtype=np.float32)
            if human.shape != trajectories.shape[1:]:
                raise ValueError(
                    f"human trajectory for {token} is {human.shape}, candidates are "
                    f"{trajectories.shape[1:]} -- check config.num_future_frames "
                    f"against config.time_horizon / interval_length"
                )
            trajectories = np.concatenate([human[None], trajectories], axis=0)
        jobs.append(
            (token, source.name, trajectories, source.scoring_payload(token))
        )
    return jobs


def score_jobs(jobs: Sequence, workers: int) -> Dict[str, tuple]:
    """Score every scene's candidates in a process pool.

    Processes, never threads: `compute_navsim_score` keeps per-call state on a
    module-level `PDMScorer` singleton, and concurrent calls in one process
    interleave and corrupt each other's results with no exception -- wrong labels,
    silently, which is the one failure this pipeline cannot detect downstream.

    "spawn", not the Linux default "fork". By the time this runs, the RAP model has
    been on the GPU, so the parent holds an initialised CUDA context, and forking
    one is unsafe -- it has been observed in this project as a pool of workers
    sitting at 0% CPU forever, holding the parent's image, having scored nothing.
    spawn costs one nuplan import per worker and is correct.
    """
    if workers <= 1:
        return {
            token: (scores, mask)
            for token, scores, mask in tqdm(map(score_job, jobs), total=len(jobs), desc="scoring")
        }

    context = multiprocessing.get_context("spawn")
    with context.Pool(workers) as pool:
        results = list(
            tqdm(pool.imap_unordered(score_job, jobs, chunksize=4), total=len(jobs), desc="scoring")
        )
    return {token: (scores, mask) for token, scores, mask in results}


def collect_round(config: RLConfig, round_index: int, device: torch.device) -> Path:
    """Sample, score and append one round. Returns the buffer file written."""
    from rl.model import RAPScorer

    config.output_path.mkdir(parents=True, exist_ok=True)
    sources = data_module.build_sources(config)

    # Which scenes this round touches. CARE contributes its whole training split
    # every round -- 50 clips is a small pool and every window is worth revisiting
    # under the new model. navtrain contributes a fresh seeded subsample, so across
    # rounds the buffer accumulates coverage of ordinary driving rather than
    # re-scoring the same thousand scenes four times.
    rng = np.random.default_rng(config.seed + 1000 * round_index)
    token_source: Dict[str, str] = {}
    per_source: Dict[str, List[str]] = {}

    care_train, _ = data_module.split_tokens(sources["care"].tokens, config.val_fraction)
    per_source["care"] = care_train
    for token in care_train:
        token_source[token] = "care"

    if "navtrain" in sources:
        navtrain_train, _ = data_module.split_tokens(
            sources["navtrain"].scorable_tokens(), config.val_fraction
        )
        wanted = min(len(navtrain_train), int(round(config.regular_ratio * len(care_train))))
        chosen = rng.choice(len(navtrain_train), size=wanted, replace=False)
        selected = [navtrain_train[i] for i in sorted(chosen)]
        per_source["navtrain"] = selected
        for token in selected:
            token_source[token] = "navtrain"

    print(f"[collect] round {round_index}: " + ", ".join(
        f"{name} {len(tokens)} scenes" for name, tokens in per_source.items()
    ))

    checkpoint = config.round_checkpoint(round_index - 1)
    print(f"[collect] sampling from {checkpoint}")
    model = RAPScorer.from_checkpoint(config, checkpoint, device)

    candidates: Dict[str, np.ndarray] = {}
    for name, tokens in per_source.items():
        candidates.update(
            propose_for_tokens(model, sources[name], tokens, config, device, rng)
        )

    # Release the GPU before the scoring pool starts: scoring is CPU-bound and can
    # run for an hour, and a 3.9 GB model sitting idle on the card for that hour is
    # a card another job could have had.
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    include_human = config.include_human and round_index == 1
    jobs = build_jobs(candidates, sources, token_source, include_human)
    scored = score_jobs(jobs, config.score_workers)

    records: List[Record] = []
    for token, source_name, trajectories, _payload in jobs:
        if token not in scored:
            continue
        scores, mask = scored[token]
        for row, (trajectory, score) in enumerate(zip(trajectories, scores)):
            is_human = include_human and row == 0
            records.append(
                Record(
                    token=token,
                    source=source_name,
                    origin="human" if is_human else "proposal",
                    # The human trajectory is not something a round produced, so it
                    # is filed under round 0 and stays distinguishable in the
                    # buffer summary from the rows this round actually sampled.
                    round=0 if is_human else round_index,
                    traj=trajectory,
                    scores=score,
                    mask=mask,
                )
            )

    existing = None
    if buffer_module.rounds_present(config.buffer_path):
        existing = buffer_module.load(config.buffer_path)
    kept = buffer_module.deduplicate(records, existing)
    if len(kept) < len(records):
        print(f"[collect] dropped {len(records) - len(kept)} duplicate trajectories")

    written = buffer_module.write_round(config.buffer_path, round_index, kept)
    finite = np.isfinite(np.stack([r.scores for r in kept]))
    print(f"[collect] wrote {written} ({len(kept)} rows)")
    print(f"[collect] mean true PDMS this round: "
          f"{np.nanmean([r.scores[PDM_INDEX] for r in kept]):.4f}")
    if not finite.all():
        unscored = int((~finite.all(axis=1)).sum())
        print(f"[collect] {unscored} rows have no finite score and will be dropped on load")
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--round", type=int, required=True, help="1-based round index")
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--samples-per-scene", type=int, default=None)
    parser.add_argument("--regular-ratio", type=float, default=None)
    parser.add_argument("--score-workers", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--limit-scenes", type=int, default=0,
                        help="Cap scenes per source. Smoke tests only -- a capped "
                             "round writes a real buffer file that a later full run "
                             "would train on, so point it at its own experiment name.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = RLConfig()
    for field in ("experiment_name", "samples_per_scene", "regular_ratio",
                  "score_workers", "batch_size"):
        value = getattr(args, field)
        if value is not None:
            setattr(config, field, value)
    config.validate()

    if args.limit_scenes:
        _install_scene_cap(args.limit_scenes)

    collect_round(config, args.round, torch.device(args.device))
    return 0


def _install_scene_cap(limit: int) -> None:
    """Truncate every source's token list. Smoke tests only."""
    original = data_module.split_tokens

    def capped(tokens, val_fraction):
        train, val = original(tokens, val_fraction)
        return train[:limit], val[: max(1, limit // 10)]

    data_module.split_tokens = capped
    print(f"[collect] SMOKE TEST: capped at {limit} scenes per source")


if __name__ == "__main__":
    raise SystemExit(main())
