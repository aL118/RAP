"""One-shot: convert the aug checkpoint from positional indices to log names.

The old format stored positions in sorted(os.listdir(NUPLAN_DB_PATH)) over ALL
raw .db files. create_openscene_metadata_aug.py now filters to navtrain and
records names, so the old indices have to be resolved against the unfiltered
listing they were written against. Run with the job stopped.
"""
import os
import sys
import yaml

here = os.path.dirname(os.path.abspath(__file__))
db_path = os.environ.get(
    "NUPLAN_DB_PATH",
    "/fs/nexus-projects/sim2real/aliu/nuplan/dataset/nuplan-v1.1/splits/trainval",
)
old_ckpt = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "checkpoint.txt")
new_ckpt = sys.argv[2] if len(sys.argv) > 2 else os.path.join(here, "checkpoint_aug.txt")

unfiltered = sorted(
    f[:-3] for f in os.listdir(db_path) if os.path.isfile(os.path.join(db_path, f))
)

with open(old_ckpt) as f:
    entries = [line.strip() for line in f if line.strip()]

if not all(e.isdigit() for e in entries):
    sys.exit(f"{old_ckpt} already holds names, not indices -- nothing to migrate.")

done_names = {unfiltered[int(i)] for i in entries}

navtrain = set(yaml.safe_load(open(os.path.join(here, "navtrain.yaml")))["log_names"])
val_logs = set(
    yaml.safe_load(open(os.path.join(here, "default_train_val_test_log_split.yaml")))["val_logs"]
)
kept = sorted(n for n in done_names if n in navtrain and n not in val_logs)

with open(new_ckpt, "w") as f:
    f.write("".join(f"{n}\n" for n in kept))

print(f"{len(entries)} indices -> {len(kept)} navtrain logs kept -> {new_ckpt}")
print(f"({len(done_names) - len(kept)} completed logs dropped: not in navtrain)")
