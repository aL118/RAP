"""Invariants of which proposals `rl/eval.py` truly scores.

There is one property here worth a test, and it is not obvious enough to survive
on inspection: **only `selected` may depend on the scorer being evaluated.**

`pool_best` and `spearman` exist to be compared across rounds. They are measured on a
subset of the 64 proposals, because scoring all of them on every val scene is the
most expensive thing in the pipeline. If that subset is chosen using the
checkpoint's own predicted scores, both numbers move when the scorer improves even
with the proposals untouched, and the comparison they exist for is worthless.

The first attempt at this drew uniformly from the slots *other than* the argmax,
which looks scorer-independent and is not: the pool itself then depends on which
slot the scorer ranked first, so two checkpoints draw different uniform parts on
the same scene. `test_uniform_part_is_scorer_independent` is that bug.

    python -m pytest rl/test_eval_sampling.py     (or run this file directly)
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl.eval import _scored_indices

NUM_PROPOSALS = 64
BUDGET = 16


def test_argmax_is_row_zero():
    """`selected` reads row 0 and must be exact at any budget."""
    rng = np.random.default_rng(0)
    for trial in range(100):
        predicted = rng.random(NUM_PROPOSALS)
        indices, num_uniform = _scored_indices(predicted, BUDGET, f"t{trial}")
        assert indices[0] == np.argmax(predicted)
        assert len(indices) == BUDGET
        assert num_uniform == BUDGET - 1


def test_uniform_part_is_scorer_independent():
    """The regression that matters: the same scene draws the same slots whichever
    checkpoint is being evaluated, so `pool_best` and `spearman` are comparable."""
    rng = np.random.default_rng(1)
    for token in ("tok-a", "tok-b", "care_clip_07@31", "0f3e2a1b"):
        a, _ = _scored_indices(rng.random(NUM_PROPOSALS), BUDGET, token)
        b, _ = _scored_indices(rng.random(NUM_PROPOSALS), BUDGET, token)
        assert np.array_equal(a[1:], b[1:]), f"uniform part moved with the scorer on {token}"


def test_uniform_part_varies_by_scene():
    """Seeded, but not constant -- every scene drawing the same slots would make
    `pool_best` an average over one fixed corner of each proposal set."""
    rng = np.random.default_rng(2)
    draws = {
        token: tuple(sorted(_scored_indices(rng.random(NUM_PROPOSALS), BUDGET, token)[0][1:]))
        for token in (f"scene-{i}" for i in range(20))
    }
    assert len(set(draws.values())) == len(draws)


def test_no_duplicates_within_the_uniform_part():
    """A repeated slot would weight one trajectory twice in `pool_best` and put a tie
    into `spearman` that is an artefact of sampling."""
    rng = np.random.default_rng(3)
    for trial in range(500):
        indices, num_uniform = _scored_indices(rng.random(NUM_PROPOSALS), BUDGET, f"t{trial}")
        assert len(set(indices[1:].tolist())) == num_uniform


def test_draw_is_uniform_over_slots():
    """`pool_best` is an estimate of the proposal pool, so the draw behind it
    has to actually be uniform -- a draw biased towards low slot indices would
    report a corner of the set as if it were the whole."""
    rng = np.random.default_rng(4)
    counts = np.zeros(NUM_PROPOSALS)
    trials = 20000
    for trial in range(trials):
        counts[_scored_indices(rng.random(NUM_PROPOSALS), BUDGET, f"s{trial}")[0][1:]] += 1

    expected = trials * (BUDGET - 1) / NUM_PROPOSALS
    assert abs(counts.mean() - expected) < 1e-6
    assert counts.std() / expected < 0.05


def test_budget_edge_cases():
    rng = np.random.default_rng(5)
    # budget 1 (and 0, which clamps up) is the argmax alone: `selected` still
    # works, `pool_best` and `spearman` have nothing to report.
    assert _scored_indices(rng.random(NUM_PROPOSALS), 1, "t")[1] == 0
    assert _scored_indices(rng.random(NUM_PROPOSALS), 0, "t")[1] == 0
    # a budget past the proposal count is capped, never an out-of-range index
    indices, _ = _scored_indices(rng.random(NUM_PROPOSALS), 999, "t")
    assert len(indices) == NUM_PROPOSALS and indices.max() < NUM_PROPOSALS
    # fewer proposals than the budget asks for
    assert len(_scored_indices(rng.random(3), BUDGET, "t")[0]) == 3


def test_tied_predictions():
    """Every proposal predicted identically -- a real case on an easy scene, where
    most proposals score 1.0 on four of the six sub-scores."""
    indices, num_uniform = _scored_indices(np.ones(NUM_PROPOSALS), BUDGET, "t")
    assert len(set(indices[1:].tolist())) == num_uniform
    assert 0 <= indices[0] < NUM_PROPOSALS


if __name__ == "__main__":
    failures = 0
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            try:
                function()
                print(f"  ok   {name}")
            except AssertionError as error:
                failures += 1
                print(f"  FAIL {name}: {error}")
    raise SystemExit(failures)
