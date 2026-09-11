"""The append-only buffer of scored trajectories.

Shape of the thing
------------------
One row is *one trajectory on one scene, with the score the true scorer gave it*:

    token   str    the scene it belongs to
    source  str    "care" or "navtrain" -- which loader can rebuild that scene
    origin  str    "human" or "proposal"
    round   int    which round produced it (0 = the human, added once)
    traj    (P, 3) the trajectory itself, ego frame
    scores  (6,)   true sub-scores, rl.scoring.SCORE_KEYS order
    mask    (6,)   which of those six columns are actually labelled

Rounds append; they never replace. That is the point of the design rather than an
implementation convenience. Round 2 samples from a model that round 1 already
moved, so its trajectories are drawn from a different distribution; keeping round
1's rows means the scorer stays calibrated on the trajectories it used to propose
instead of forgetting them the moment it stops proposing them, which is the classic
way an iteratively retrained ranker walks off a cliff. It also means every round is
cheaper than a from-scratch relabel: the true score of a trajectory does not change
when the model does, so a row scored once is scored forever.

Only trajectories and scores are stored, never features. The refiners train, so a
round-1 trajectory has to be re-featurised by the round-3 model before round 3 can
learn from it -- see the module docstring in rl/model.py.

On disk
-------
`round_XX.npz` per round, written once and never rewritten, so an interrupted round
loses that round and nothing before it. Written to a temporary name and renamed:
`os.replace` is atomic within a filesystem, so a file either does not exist or is
complete. The temporary name ends in `.npz` because `np.savez_compressed` appends
`.npz` to any path that does not, and starts with a dot so it does not match the
`round_*.npz` glob a concurrent reader would use.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence
import os

import numpy as np

from rl.scoring import PDM_INDEX


@dataclass
class Record:
    """One scored trajectory, before it is packed into arrays."""

    token: str
    source: str
    origin: str
    round: int
    traj: np.ndarray      # (P, 3)
    scores: np.ndarray    # (6,)
    mask: np.ndarray      # (6,) bool


class Buffer:
    """Every round's rows, loaded flat and indexed by scene."""

    def __init__(
        self,
        tokens: np.ndarray,
        sources: np.ndarray,
        origins: np.ndarray,
        rounds: np.ndarray,
        traj: np.ndarray,
        scores: np.ndarray,
        mask: np.ndarray,
    ):
        self.tokens = tokens
        self.sources = sources
        self.origins = origins
        self.rounds = rounds
        self.traj = traj
        self.scores = scores
        self.mask = mask

        self.by_token: Dict[str, List[int]] = {}
        for index, token in enumerate(tokens):
            self.by_token.setdefault(str(token), []).append(index)

    def __len__(self) -> int:
        return len(self.tokens)

    def source_of(self, token: str) -> str:
        return str(self.sources[self.by_token[token][0]])

    def rows_for(self, token: str) -> np.ndarray:
        return np.array(self.by_token[token], dtype=np.int64)

    def summary(self) -> str:
        """One line per (source, round): rows, scenes, and the mean true PDMS.

        The mean is the number to watch across rounds. It should *rise* as the
        planner improves and the trajectories it proposes get better -- but only on
        the rows that round added, which is why this splits by round rather than
        reporting a single average over a buffer that mixes four generations.
        """
        lines = []
        for source in sorted(set(map(str, self.sources))):
            for round_index in sorted(set(self.rounds.tolist())):
                selected = (self.sources == source) & (self.rounds == round_index)
                if not selected.any():
                    continue
                pdm = self.scores[selected, PDM_INDEX]
                lines.append(
                    f"  {source:<9} round {round_index}: {int(selected.sum()):6d} rows, "
                    f"{len(set(map(str, self.tokens[selected]))):5d} scenes, "
                    f"mean PDMS {np.nanmean(pdm):.4f}"
                )
        return "\n".join(lines)


def _round_path(buffer_path: Path, round_index: int) -> Path:
    return Path(buffer_path) / f"round_{round_index:02d}.npz"


def rounds_present(buffer_path: Path) -> List[int]:
    """Round indices already on disk, so `loop.py` can resume."""
    buffer_path = Path(buffer_path)
    if not buffer_path.is_dir():
        return []
    return sorted(
        int(path.stem.split("_")[1]) for path in buffer_path.glob("round_*.npz")
    )


def write_round(buffer_path: Path, round_index: int, records: Sequence[Record]) -> Path:
    """Pack one round's records and write them atomically."""
    if not records:
        raise ValueError(f"round {round_index} produced no records")

    buffer_path = Path(buffer_path)
    buffer_path.mkdir(parents=True, exist_ok=True)
    out_path = _round_path(buffer_path, round_index)

    arrays = {
        "tokens": np.array([r.token for r in records], dtype=object),
        "sources": np.array([r.source for r in records], dtype=object),
        "origins": np.array([r.origin for r in records], dtype=object),
        "rounds": np.array([r.round for r in records], dtype=np.int32),
        "traj": np.stack([np.asarray(r.traj, np.float32) for r in records]),
        "scores": np.stack([np.asarray(r.scores, np.float32) for r in records]),
        "mask": np.stack([np.asarray(r.mask, bool) for r in records]),
    }

    tmp_path = out_path.with_name(f".tmp_{out_path.name}")
    np.savez_compressed(tmp_path, **arrays)
    os.replace(tmp_path, out_path)
    return out_path


def load(buffer_path: Path, up_to_round: Optional[int] = None) -> Buffer:
    """Every round on disk, concatenated.

    Rows whose scores are not finite are dropped here rather than at training time.
    A scene the true scorer failed on is written with a NaN row (see
    rl/scoring.py) so that one bad scenario cannot abort a round that has already
    spent an hour scoring, and this is the other half of that: the row exists as a
    record of the failure and never reaches a loss.

    :param up_to_round: load rounds <= this. Round r trains on rounds 1..r, and
        passing r explicitly is what makes a re-run of round r reproducible after
        round r+1 has already been collected.
    """
    paths = sorted(Path(buffer_path).glob("round_*.npz"))
    if not paths:
        raise FileNotFoundError(
            f"No round_*.npz in {buffer_path}. Run rl/collect.py first."
        )

    collected: Dict[str, List[np.ndarray]] = {}
    for path in paths:
        if up_to_round is not None and int(path.stem.split("_")[1]) > up_to_round:
            continue
        with np.load(path, allow_pickle=True) as data:
            for key in ("tokens", "sources", "origins", "rounds", "traj", "scores", "mask"):
                collected.setdefault(key, []).append(data[key])

    packed = {key: np.concatenate(value) for key, value in collected.items()}

    finite = np.isfinite(packed["scores"]).all(axis=1)
    dropped = int((~finite).sum())
    if dropped:
        print(f"[buffer] dropped {dropped} of {len(finite)} rows with no finite score")
    packed = {key: value[finite] for key, value in packed.items()}

    return Buffer(**{
        "tokens": packed["tokens"],
        "sources": packed["sources"],
        "origins": packed["origins"],
        "rounds": packed["rounds"],
        "traj": packed["traj"],
        "scores": packed["scores"],
        "mask": packed["mask"],
    })


def deduplicate(
    records: Sequence[Record], existing: Optional[Buffer], tolerance: float = 1e-3
) -> List[Record]:
    """Drop trajectories a previous round already scored on the same scene.

    A round samples from a model that has moved, but not necessarily on every
    scene: an easy scene can hand back proposals almost identical to last round's,
    and every duplicate that lands in the buffer silently upweights that scene's
    loss without adding information. Keys on the trajectory rounded to
    `tolerance` metres, keeping the first occurrence -- which is the one whose
    round number honestly records when it was first proposed.
    """
    seen = set()
    if existing is not None:
        for index in range(len(existing)):
            key = (
                str(existing.tokens[index]),
                np.round(existing.traj[index] / tolerance).astype(np.int64).tobytes(),
            )
            seen.add(key)

    kept: List[Record] = []
    for record in records:
        key = (
            record.token,
            np.round(np.asarray(record.traj) / tolerance).astype(np.int64).tobytes(),
        )
        if key in seen:
            continue
        seen.add(key)
        kept.append(record)
    return kept
