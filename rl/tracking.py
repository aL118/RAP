"""Where a run's numbers go, so a curve exists after the fact.

Everything here writes; nothing here decides anything. Two sinks, because they
answer two different questions on a cluster:

  ``logs/<name>.jsonl``  one line per record, appended, line-buffered. Greppable
                         from a login node with `jq` and no server, which is the
                         only way to look at a running job's numbers over ssh.
                         This is the durable one -- the event files are a view.
  ``tb/<name>/``         TensorBoard event files, one subdirectory per sink so
                         `tensorboard --logdir exp/<run>/tb` shows `train` and
                         `eval` as separate runs with their own step axes. That
                         separation is the point: train steps are batches and
                         eval steps are round indices, and putting both on one
                         axis would make the eval points land at step 3740.

Why this exists
---------------
Round 1 of `rap_iterative_quicktest` (job 7479555) regressed the planner on both
sources -- `selected` fell 0.54 -> 0.41 on CARE and 0.96 -> 0.65 on navtrain --
while every loss the run printed went down monotonically. The five epoch lines in
the .out file were the entire record, and per-batch values were averaged and
dropped. Two things would have shown it and neither was recoverable afterwards:
the per-source split (the two sources pull the shared head in different
directions, and an epoch mean over both averages exactly that away) and
`trajectory_loss` at batch resolution (it never got below its epoch-0 value).

So: per batch, tagged by source, plus grad norm, plus the eval metrics on the same
timeline. `record` is the durable path and `scalars` is TensorBoard-only, used
where the JSONL line already carries the field being split on -- the per-source
curves are recoverable from the flat `source` field, so duplicating those lines
would only make the file bigger.

TensorBoard is optional. It is imported lazily and a missing install degrades to
JSONL rather than failing a training run hours in.
"""

import json
import math
import time
from pathlib import Path
from typing import Mapping, Optional


def _finite(value) -> Optional[float]:
    """A float JSON can hold, or None. `json.dumps` happily emits bare `NaN`,
    which Python reads back and nothing else does."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _last_step(path: Path) -> int:
    """Highest step already in the file, or -1.

    Read rather than tracked in memory so the step axis survives the process
    boundary: `loop.py` calls `train_round` once per round, a resumed run starts a
    fresh process at round 3, and a curve that restarted at zero each time would
    be unreadable. The tail is enough -- records are appended in step order.
    """
    if not path.is_file():
        return -1
    with open(path, "rb") as handle:
        handle.seek(0, 2)
        handle.seek(max(0, handle.tell() - 65536))
        lines = handle.read().decode("utf-8", "replace").splitlines()
    for line in reversed(lines):
        if not line.strip():
            continue
        try:
            return int(json.loads(line)["step"])
        except (ValueError, KeyError, TypeError):
            continue  # a truncated last line from a killed job is not fatal
    return -1


class RunLog:
    """One sink: a JSONL file and, optionally, a TensorBoard event directory.

    :param directory: the run directory (`config.output_path`).
    :param name: sink name -- `train` or `eval`. Names both files.
    :param tensorboard: False writes JSONL only.
    """

    def __init__(self, directory: Path, name: str, tensorboard: bool = True):
        self.name = name
        self.path = Path(directory) / "logs" / f"{name}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # Read before opening: append does not truncate, but the ordering makes it
        # obvious that the resume point is the *previous* run's, not this one's.
        self.step = _last_step(self.path) + 1

        self._file = open(self.path, "a", buffering=1)  # line-buffered: tail -f works
        self._writer = self._open_writer(Path(directory) / "tb" / name) if tensorboard else None

    @staticmethod
    def _open_writer(path: Path):
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            print("[log] tensorboard not installed -- writing JSONL only")
            return None
        path.mkdir(parents=True, exist_ok=True)
        # flush_secs so a job's curves are watchable while it runs, which on an
        # 8-hour walltime is most of the value.
        return SummaryWriter(log_dir=str(path), flush_secs=30)

    def record(self, values: Mapping[str, float], step: int, prefix: str = "", **fields) -> None:
        """One JSONL line, and one TensorBoard scalar per entry of `values`.

        JSONL keys stay flat (`loss`), TensorBoard tags get `prefix` (`batch/loss`):
        the file is for `jq`, the tags are for grouping in the UI.

        :param fields: flat context columns -- `kind`, `round`, `epoch`, `source`.
            These are what make the per-source split recoverable from the file, so
            it never needs a second line for it.
        """
        line = {"step": int(step), "wall": round(time.time(), 3), **fields}
        line.update({key: _finite(value) for key, value in values.items()})
        self._file.write(json.dumps(line) + "\n")
        self.scalars(values, step, prefix)

    def scalars(self, values: Mapping[str, float], step: int, prefix: str = "") -> None:
        """TensorBoard only. For a cut of numbers already in the JSONL under a
        context field -- the per-source curves, which `record`'s `source` column
        already describes."""
        if self._writer is None:
            return
        for key, value in values.items():
            number = _finite(value)
            if number is not None:
                self._writer.add_scalar(f"{prefix}{key}", number, int(step))

    def close(self) -> None:
        self._file.close()
        if self._writer is not None:
            self._writer.close()

    def __enter__(self) -> "RunLog":
        return self

    def __exit__(self, *_) -> None:
        self.close()
