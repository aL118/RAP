#!/usr/bin/env python3
"""Print the dataset's composition -- scenario against setting -- as a table.

    python dataset_composition.py                    # data/CARE_YTB
    python dataset_composition.py --dataset CARE_YTB --markdown

Read straight out of each clip's info.json, so the table is whatever the corpus
currently is rather than a number written down once and left to rot.

BOTH AXES PARTITION THE CLIPS, which is the only way the Total column and the
Total row can both be arithmetic rather than decoration. info.json does not
partition on its own -- `event_type` is a list, and a clip is routinely both a
collision and a near miss -- so each axis is a list of (label, test) pairs
applied in order and the first match wins. The order below is therefore the
claim being made about the corpus, and it is: an outcome beats a near-outcome,
and a road class beats the land use around it.

The cost of that is real and is reported under the table rather than buried: a
collision on an icy road counts once, under collision, so the "Adverse weather"
row undercounts how much bad weather the corpus actually holds. The footnote
prints the true figure alongside.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# The two axes. Order is precedence: first match wins, so read them top down.
# ---------------------------------------------------------------------------

# `ego_involved` is the split that matters for planning: a collision the ego is
# part of is a label about the ego's own trajectory, one it merely films is a
# label about somebody else's. One clip has it null, which falls to third-party
# -- the ego plainly did not hit anything in it, and guessing True would put a
# crash in the ego's own record on the strength of a missing field.
SCENARIOS = [
    ("Ego collision",
     lambda i: "collision" in i["event_type"] and i.get("ego_involved") is True),
    ("Third-party collision",
     lambda i: "collision" in i["event_type"]),
    ("Near-miss",
     lambda i: "near_miss" in i["event_type"]),
    ("Adverse weather",
     lambda i: i["weather"] != "clear" or i["road_surface"] != "dry"),
    # cut_in, blocking and ped_in_roadway clips that reached neither a collision
    # nor a near miss. Not a leftover bin to be embarrassed about: "someone
    # blocked the lane and nothing happened" is a real scenario and the corpus
    # is meant to hold some.
    ("Other conflict", lambda i: True),
]

# "Hwy." is a road class and the other two are land use, which is not an
# oversight: a freeway through a suburb is a freeway scenario, and the road
# class is what decides the speeds and the geometry. So road_type is tested
# first and land_use only separates what is left. Rural joins Sub. because
# three columns cannot hold four values and rural roads are the nearer of the
# two in everything the label is standing in for -- speed, density, sight lines.
SETTINGS = [
    ("Hwy.", lambda i: i["road_type"] == "freeway"),
    ("Urban", lambda i: i["land_use"] == "urban"),
    ("Sub.", lambda i: True),
]

# Lighting. dawn_dusk counts as Night: the point of the column is whether the
# detector and the camera had daylight to work with, and at dawn or dusk they
# did not. Folding it into Day would flatter the corpus by five clips.
LIGHTING = [
    ("Day", lambda i: i["time"] == "day"),
    ("Night", lambda i: True),
]

REQUIRED = ("time", "weather", "road_surface", "land_use", "road_type", "event_type")


def classify(info: dict, axis: list) -> str:
    for label, test in axis:
        if test(info):
            return label
    raise AssertionError("every axis ends in a catch-all")     # unreachable


def load(dataset: str) -> dict:
    """{clip: info} for every clip in the dataset that has an info.json."""
    root = BASE / "data" / dataset
    if not root.is_dir():
        raise SystemExit(f"no such dataset: {root}")
    infos, missing = {}, []
    for path in sorted(root.glob("*/info.json")):
        clip = path.parent.name
        info = json.loads(path.read_text())
        absent = [key for key in REQUIRED if key not in info]
        if absent:
            # Skipped rather than defaulted. A clip missing `road_type` would
            # otherwise land in whichever column the catch-all happens to be and
            # be counted as a fact.
            missing.append((clip, absent))
            continue
        infos[clip] = info
    for path in sorted(root.iterdir()):
        if path.is_dir() and not (path / "info.json").exists():
            missing.append((path.name, ["info.json"]))
    return infos, missing


def render(infos: dict, markdown: bool) -> str:
    columns = [label for label, _ in LIGHTING] + [label for label, _ in SETTINGS]
    rows = [label for label, _ in SCENARIOS]

    counts = {row: dict.fromkeys(columns, 0) for row in rows}
    totals = dict.fromkeys(rows, 0)
    for info in infos.values():
        row = classify(info, SCENARIOS)
        counts[row][classify(info, LIGHTING)] += 1
        counts[row][classify(info, SETTINGS)] += 1
        totals[row] += 1
    # Dropped rather than printed empty: a scenario nothing in the corpus
    # matches is noise in a composition table, and the row order is a
    # precedence rule, not a promised set of headings.
    rows = [row for row in rows if totals[row]]

    width = max([len("Scenario")] + [len(row) for row in rows] + [len("Total")])
    head = ["Scenario".ljust(width)] + columns + ["Total"]
    body = [[row.ljust(width)] + [str(counts[row][c]) for c in columns] + [str(totals[row])]
            for row in rows]
    body.append(["Total".ljust(width)]
                + [str(sum(counts[r][c] for r in rows)) for c in columns]
                + [str(sum(totals.values()))])

    cells = [head] + body
    widths = [max(len(cell[i]) for cell in cells) for i in range(len(head))]
    lines = []
    for index, cell in enumerate(cells):
        padded = [cell[0].ljust(widths[0])] + [c.rjust(widths[i + 1])
                                               for i, c in enumerate(cell[1:])]
        lines.append(("| " + " | ".join(padded) + " |") if markdown
                     else "  ".join(padded).rstrip())
        if markdown and index == 0:
            lines.append("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    return "\n".join(lines)


def footnotes(infos: dict) -> list:
    """The counts the precedence rule hides, stated rather than left implicit."""
    notes = []
    adverse = [c for c, i in infos.items()
               if i["weather"] != "clear" or i["road_surface"] != "dry"]
    shown = [c for c in adverse if classify(infos[c], SCENARIOS) == "Adverse weather"]
    if len(adverse) != len(shown):
        notes.append(f"{len(adverse)} clips are wet, icy or overcast; "
                     f"{len(adverse) - len(shown)} of them count under a collision or "
                     "near-miss row instead, which outranks weather.")
    low = [c for c, i in infos.items() if i["time"] != "day"]
    dusk = [c for c in low if infos[c]["time"] == "dawn_dusk"]
    if dusk:
        notes.append(f"Night includes {len(dusk)} dawn/dusk clips "
                     f"({len(low) - len(dusk)} are true night).")
    rural = [c for c, i in infos.items()
             if i["road_type"] != "freeway" and i["land_use"] == "rural"]
    if rural:
        notes.append(f"Sub. includes {len(rural)} rural clips.")
    glare = [c for c, i in infos.items() if i.get("sun_glare")]
    if glare:
        notes.append(f"{len(glare)} clips are flagged sun_glare (cuts across every row).")
    return notes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="CARE_YTB",
                        help="directory under data/ (default: CARE_YTB)")
    parser.add_argument("--markdown", action="store_true",
                        help="pipe-delimited, for pasting into a doc or a paper")
    args = parser.parse_args()

    infos, missing = load(args.dataset)
    if not infos:
        raise SystemExit(f"no usable info.json under data/{args.dataset}")

    print(f"{args.dataset}: {len(infos)} clips\n")
    print(render(infos, args.markdown))
    notes = footnotes(infos)
    if notes:
        print()
        for note in notes:
            print(f"  * {note}")
    if missing:
        print(f"\n  ! {len(missing)} clip(s) left out of the table:")
        for clip, fields in missing:
            print(f"      {clip}: no {', '.join(fields)}")


if __name__ == "__main__":
    main()
