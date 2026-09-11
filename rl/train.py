"""One round of training: fit the scorer to everything the buffer holds.

What is being trained
---------------------
The `Scorer` head and the `Traj_refiner` stack that feeds it. The DINOv3 tower is
frozen (`config.freeze_backbone`), as it is in RAP's own supervised training.

Where the score loss can and cannot reach
-----------------------------------------
Worth being exact about, because it is not what the shape of the code suggests.
`Bev_refiner` takes the trajectory in as `ref_2d = pose.detach()`
(`bevformer/bev_refiner.py`), so **no gradient from the score loss ever reaches a
`traj_decoder`**. The score loss moves the Bev_refiner stack, `hist_encoding`,
`init_feature` and the scorer head; the decoders that actually emit the proposals
are reachable only through the imitation term.

Measured on round 1 of the smoke run, which had no navtrain rows and therefore no
imitation loss: all four stages' `Bev_refiner` weights moved, and all four stages'
`traj_decoder` weights were bit-identical to the pretrained checkpoint afterwards.
(20 of the scorer's 30 tensors are also untouched, but that is deliberate and
unrelated -- they are the agent/area/BEV-semantic heads that `_score_head` skips.)

Two consequences:

  * Proposals still change between rounds, which is what the rounds need. They
    change *indirectly*: the score loss moves the features the decoders read, not
    the decoders. The DAgger premise holds.
  * `regular_ratio = 0` is more degenerate than "the model goes timid". With no
    navtrain rows there is no imitation loss, so the proposal decoders receive
    exactly zero gradient and this becomes scorer-only distillation. That is fine
    for a smoke test and is not a training configuration.

Nothing here pulls proposal geometry towards higher true PDMS on CARE scenes; the
scorer learns to rank crash-avoidance and inference picks the best of whatever was
proposed. That is what "distil the true score into the Scorer head" means, and it
is worth knowing before reading a flat `pool_best` as a failure.

Two losses, on two different halves of the batch
------------------------------------------------
**Score loss, on every row.** Binary cross-entropy from the model's predicted
sub-scores to the true ones the buffer stores, masked twice over: by which columns
that row actually has labels for (CARE clips have no drivable-area ground truth --
see rl/scoring.py) and by which rows are real rather than padding. The aggregate
PDMS column gets a second, separate term because that is the single number
inference ranks proposals on, and averaging it in with five others under-weights
the thing being optimised.

**Imitation loss, on navtrain rows only.** The winner-take-all L1 from RAP's own
`rap_loss`: the closest proposal to the human trajectory is pulled towards it. It
is what stops the planner from drifting somewhere the scorer happens to like while
nothing holds it to plausible driving.

It is deliberately *not* applied to CARE rows, and that is the sharpest asymmetry
in this pipeline. A CARE clip is a crash: its human trajectory is the one that hit
something. Training the planner to reproduce it is the exact opposite of the goal.
The CARE human trajectory still enters the buffer as a *scored candidate* -- it is
a real trajectory with a real score, usually the most informative row in its scene
-- but it is never a target. This is also the other half of what `regular_ratio`
buys: without navtrain rows there is no imitation signal at all, and the cheapest
way to score well on crash footage is to stop moving.

Buffer rows per scene, and the 64-slot ceiling
----------------------------------------------
Four rounds at 33 rows a scene is ~130 rows, and the refiner has exactly
`proposal_num` (64) query slots. So a step samples up to `max_candidates` of a
scene's rows rather than taking all of them; over epochs every row is seen. The
sampling is uniform over the scene's rows regardless of which round produced them,
which is what "without replacing old samples" means in practice -- an old row is
exactly as likely to be trained on as a new one.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl import buffer as buffer_module
from rl import data as data_module
from rl.config import RLConfig
from rl.scoring import PDM_INDEX
from rl.tracking import RunLog


def gather_candidates(
    buffer, tokens: Sequence[str], max_candidates: int, rng: np.random.Generator
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack one batch of scenes' buffered rows into padded tensors.

    :return: ``(traj (B, K, P, 3), scores (B, K, 6), column_mask (B, K, 6),
        row_valid (B, K))`` where K is the largest row count in this batch, capped
        at `max_candidates`. Scenes with fewer rows are padded by repeating their
        first row, and `row_valid` is what keeps the padding out of the loss --
        the padded *values* are real rows rather than zeros only so the tensor is
        well-formed, never because they are meant to contribute.
    """
    per_scene = [buffer.rows_for(token) for token in tokens]
    width = min(max_candidates, max(len(rows) for rows in per_scene))

    trajectories, scores, column_mask, row_valid = [], [], [], []
    for rows in per_scene:
        if len(rows) > width:
            rows = rng.choice(rows, size=width, replace=False)
        count = len(rows)
        padding = width - count
        indices = np.concatenate([rows, np.repeat(rows[:1], padding)]) if padding else rows

        trajectories.append(buffer.traj[indices])
        scores.append(buffer.scores[indices])
        column_mask.append(buffer.mask[indices])
        valid = np.zeros(width, dtype=bool)
        valid[:count] = True
        row_valid.append(valid)

    return (
        torch.from_numpy(np.stack(trajectories)).float(),
        torch.from_numpy(np.stack(scores)).float(),
        torch.from_numpy(np.stack(column_mask)),
        torch.from_numpy(np.stack(row_valid)),
    )


def score_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    column_mask: torch.Tensor,
    row_valid: torch.Tensor,
    config: RLConfig,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Masked BCE from predicted sub-scores to the buffer's true ones.

    The mean is taken per scene and then over scenes, not over all elements at once.
    A scene that has accumulated rows over four rounds would otherwise carry four
    times the gradient of one collected this round, purely for having been sampled
    more often -- which is a property of the collection schedule, not of how much
    that scene has to teach.
    """
    weights = (column_mask & row_valid[..., None]).float()
    per_element = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

    scene_totals = (per_element * weights).sum(dim=(1, 2))
    scene_counts = weights.sum(dim=(1, 2)).clamp(min=1.0)
    sub_score = (scene_totals / scene_counts).mean()

    final_weights = row_valid.float()
    final_element = F.binary_cross_entropy_with_logits(
        logits[..., PDM_INDEX], targets[..., PDM_INDEX], reduction="none"
    )
    final_totals = (final_element * final_weights).sum(dim=1)
    final_counts = final_weights.sum(dim=1).clamp(min=1.0)
    final_score = (final_totals / final_counts).mean()

    loss = config.sub_score_weight * sub_score + config.final_score_weight * final_score
    return loss, {"sub_score_loss": float(sub_score), "final_score_loss": float(final_score)}


def imitation_loss(
    proposal_list: Sequence[torch.Tensor], human: torch.Tensor, prev_weight: float
) -> torch.Tensor:
    """Winner-take-all L1 to the human trajectory, staged, as in RAP's `rap_loss`.

    Only the closest proposal is pulled, so the other 63 stay free to cover the
    rest of the space -- pulling all of them would collapse the proposal set, and a
    collapsed set is a scorer with nothing to rank.

    Every refinement stage is supervised, not just the last, with RAP's own
    recursion `loss = prev_weight * loss + min_loss`. The four `Traj_refiner`
    stages share one module instance, so the final stage alone would still reach
    every weight -- but it would only ever ask them to be right after four
    applications, and the pretrained model was fit with each stage's output held to
    the target on its own. Keeping the recursion is what makes this a continuation
    of that training rather than a differently-shaped objective.
    """
    loss = torch.zeros((), device=human.device)
    for proposals in proposal_list:
        distance = torch.linalg.norm(proposals - human[:, None], dim=-1, ord=1).mean(-1)
        loss = prev_weight * loss + distance.amin(dim=1).mean()
    return loss


def train_round(
    config: RLConfig, round_index: int, device: torch.device, resume_from: Optional[Path] = None
) -> Path:
    """Train one round on rounds 1..round_index of the buffer. Returns the checkpoint."""
    from rl.model import RAPScorer

    buffer = buffer_module.load(config.buffer_path, up_to_round=round_index)
    print(f"[train] round {round_index} buffer:\n{buffer.summary()}")

    sources = data_module.build_sources(config)
    max_candidates = config.max_candidates

    checkpoint = resume_from or config.round_checkpoint(
        0 if config.restart_from_pretrained else round_index - 1
    )
    print(f"[train] starting from {checkpoint}")
    model = RAPScorer.from_checkpoint(config, checkpoint, device)
    if max_candidates is None:
        max_candidates = model.num_proposals
    max_candidates = min(max_candidates, model.num_proposals)

    parameters = model.trainable_parameters()
    print(f"[train] {sum(p.numel() for p in parameters) / 1e6:.1f}M trainable parameters")
    optimizer = torch.optim.AdamW(
        parameters, lr=config.learning_rate, weight_decay=config.weight_decay
    )

    # Only scenes the buffer actually has rows for. A scene whose every row failed
    # to score is not in the buffer at all (rl/buffer.py drops non-finite rows on
    # load), so this is what keeps the loaders and the buffer in step.
    loaders = {}
    for name, source in sources.items():
        tokens = [token for token in source.tokens if token in buffer.by_token]
        if not tokens:
            continue
        loaders[name] = data_module.loader(
            source,
            tokens,
            config.batch_size,
            config.num_workers,
            shuffle=True,
            include_human=(name == "navtrain"),
        )
        print(f"[train] {name}: {len(tokens)} scenes")
    if not loaders:
        raise ValueError("no scene in the buffer is loadable from any source")

    rng = np.random.default_rng(config.seed + round_index)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    log = RunLog(config.output_path, "train", config.log_tensorboard)

    model.train()
    for epoch in range(config.epochs_per_round):
        totals: Dict[str, List[float]] = {}
        by_source: Dict[str, Dict[str, List[float]]] = {}
        started = time.time()
        batches = data_module.iterate_batches(loaders, seed=config.seed + epoch)
        steps = sum(len(value) for value in loaders.values())

        for source_name, tokens, features in tqdm(
            batches, total=steps, desc=f"round {round_index} epoch {epoch}"
        ):
            human = features.pop("human_trajectory", None)
            features = {
                key: value.to(device) for key, value in features.items() if torch.is_tensor(value)
            }
            trajectories, targets, column_mask, row_valid = gather_candidates(
                buffer, tokens, max_candidates, rng
            )
            trajectories = trajectories.to(device)
            targets = targets.to(device)
            column_mask = column_mask.to(device)
            row_valid = row_valid.to(device)

            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                proposal_list, logits = model.forward_train(features, trajectories)
                loss, parts = score_loss(logits, targets, column_mask, row_valid, config)

                if human is not None and config.trajectory_weight > 0:
                    trajectory_term = imitation_loss(
                        [p.float() for p in proposal_list],
                        human.to(device),
                        model.rap_config.prev_weight,
                    )
                    loss = loss + config.trajectory_weight * trajectory_term
                    parts["trajectory_loss"] = float(trajectory_term)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            # Returned before clipping, and unscaled by the line above. Free, and
            # it separates "this head is being pushed hard" from "the loss happens
            # to be flat" -- a mean loss can sit still while one source's gradient
            # dominates every step.
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            scaler.step(optimizer)
            scaler.update()

            parts["loss"] = float(loss)
            parts["grad_norm"] = float(grad_norm)
            parts[f"batches_{source_name}"] = 1.0
            for key, value in parts.items():
                totals.setdefault(key, []).append(value)

            # Per batch and split by source, which is the whole reason this is
            # here: the epoch mean averages the two sources together, and the two
            # sources are exactly what needs looking at -- CARE supplies no
            # imitation loss and no drivable-area column, so the two halves of an
            # epoch are not measuring the same thing.
            scalars = {key: value for key, value in parts.items()
                       if not key.startswith("batches_")}
            for key, value in scalars.items():
                by_source.setdefault(source_name, {}).setdefault(key, []).append(value)
            log.record(scalars, log.step, "batch/", kind="batch",
                       round=round_index, epoch=epoch, source=source_name)
            log.scalars(scalars, log.step, f"batch_{source_name}/")
            log.step += 1

        elapsed = time.time() - started
        summary = " ".join(
            f"{key}={np.mean(value):.4f}" if not key.startswith("batches_")
            else f"{key}={int(np.sum(value))}"
            for key, value in sorted(totals.items())
        )
        print(f"[train] round {round_index} epoch {epoch} ({elapsed:.0f}s) {summary}")

        # Stamped at the epoch's last global step so the means overlay the
        # per-batch curves on one axis instead of needing a second one. Note
        # `trajectory_loss` averages over navtrain batches only -- CARE rows never
        # produce the term at all, so it is not a mean over the epoch's batches.
        means = {key: float(np.mean(value)) for key, value in totals.items()
                 if not key.startswith("batches_")}
        log.record(means, log.step - 1, "epoch/", kind="epoch", round=round_index,
                   epoch=epoch, seconds=round(elapsed, 1))
        for name, values in by_source.items():
            log.scalars({key: float(np.mean(value)) for key, value in values.items()},
                        log.step - 1, f"epoch_{name}/")

    log.close()

    out_path = config.round_checkpoint(round_index)
    model.save(out_path, full=config.save_full_checkpoints)
    print(f"[train] wrote {out_path}")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--round", type=int, required=True, help="1-based round index")
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--epochs-per-round", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--trajectory-weight", type=float, default=None)
    parser.add_argument("--regular-ratio", type=float, default=None,
                        help="0 skips building the navtrain source entirely. Only "
                             "meaningful when the buffer holds no navtrain rows.")
    parser.add_argument("--save-full", action="store_true",
                        help="Write a self-contained checkpoint instead of a delta "
                             "against the pretrained one. 3.5 GB rather than ~190 MB.")
    parser.add_argument("--restart-from-pretrained", action="store_true",
                        help="Re-fit the pretrained checkpoint on the whole buffer "
                             "instead of continuing from the previous round.")
    parser.add_argument("--no-tensorboard", action="store_true",
                        help="Skip the event files under <run>/tb/. The JSONL in "
                             "<run>/logs/ is written either way.")
    parser.add_argument("--resume-from", type=Path, default=None,
                        help="Weights to start this round from, overriding the "
                             "previous round's. For re-running a round after "
                             "changing a loss weight.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = RLConfig()
    for field in ("experiment_name", "epochs_per_round", "batch_size",
                  "learning_rate", "trajectory_weight", "regular_ratio"):
        value = getattr(args, field)
        if value is not None:
            setattr(config, field, value)
    if args.restart_from_pretrained:
        config.restart_from_pretrained = True
    if args.save_full:
        config.save_full_checkpoints = True
    if args.no_tensorboard:
        config.log_tensorboard = False
    config.validate(require_buffer=True)

    train_round(config, args.round, torch.device(args.device), args.resume_from)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
