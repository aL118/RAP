"""Iterative distillation of the true PDM score into RAP's own scorer head.

RAP already trains its `Scorer` against the true PDM score, but only ever on the
proposals the model happened to emit at that moment, scored inline, once. This
package turns that into rounds with a memory:

    round 1   for every scene, take the human trajectory and a subsample of the
              pretrained model's 64 proposals. Score all of them with the true
              scorer. Put them in a buffer. Train the scorer on the buffer.
    round r   sample fresh trajectories from the model round r-1 produced. Score
              them. APPEND to the buffer -- nothing is replaced. Retrain on
              everything. Repeat for 3-5 rounds.

Two sources of scenes, mixed deliberately: the CARE clips this is aimed at, and
enough ordinary navtrain to keep the planner from concluding that the safest thing
to do in a crash is to stop moving.

Layout::

    rl/config.py    paths, round schedule, loss weights (one dataclass)
    rl/scoring.py   true sub-scores; a nuPlan backend and a map-free CARE one
    rl/care.py      CARE clip pickles -> scenes the model and the scorer can read
    rl/data.py      both sources behind one interface, plus batching
    rl/model.py     RAP wrapped so arbitrary trajectories can be scored
    rl/buffer.py    the append-only buffer of scored trajectories
    rl/collect.py   one round: sample, score, append
    rl/train.py     one round: fit the scorer to the whole buffer
    rl/loop.py      collect -> train -> repeat, resumable
    rl/eval.py      selected / pool_best / rank correlation on held-out scenes
    rl/tracking.py  per-batch scalars to JSONL and TensorBoard
    rl/export.py    round checkpoint (a delta) -> a self-contained one

See rl/README.md for the design rationale and the run order.
"""

from rl.config import RLConfig

__all__ = ["RLConfig"]
