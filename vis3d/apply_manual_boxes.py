"""Merge hand-drawn boxes (annotate_boxes.py) into the detector's own predictions.

Manual boxes enter the pipeline at the same place GroundingDINO's do -- as mask
entries in mask_results_preds.json -- so nothing downstream needs to know a box
was drawn by a person: stage 3 lifts it, stage 3b smooths it, and the navsim
export carries it out like any other detection.

    python apply_manual_boxes.py --output_dir data/CARE_YTB/closetruck/1

Five things happen here:

  keyframes -> frames   A track carries a box only on the frames it was drawn
                        on; boxes for the frames between two keyframes are
                        interpolated linearly. Nothing is extrapolated past a
                        track's first or last keyframe, and a null keyframe
                        ends the track's run until the next real one.

  box -> mask           Everything downstream reads masks, not boxes, so each
                        box is prompted through SAM -- which is what the box
                        branch of Grounded-SAM already does with DINO's boxes.
                        --no-sam falls back to the filled rectangle, which needs
                        no GPU but hands the lifter a mask that includes
                        whatever background sits inside the box.

  manual wins           A manual box suppresses any detector box on that frame
                        it overlaps by more than --iou, so correcting a bad
                        detection means drawing the right box over it.

  reject regions        A track marked "reject" in manual_boxes.json suppresses
                        the same way but contributes no box of its own. It is
                        how a false positive is deleted rather than replaced --
                        a detection on the ego bonnet, a hedge read as a car --
                        which drawing over cannot do, since drawing leaves your
                        box behind in place of theirs.

  glimpses adopted      On the frames a track does *not* cover, a detection
                        overlapping where the track just was is given the
                        track's id instead. A detector that catches an object
                        in one- and two-frame glimpses would otherwise have
                        them all deleted by smooth_boxes --min_track_len, and
                        drawing boxes over the frames it missed would not save
                        them: drawn and detected boxes never merge downstream.
                        Adopting here, in image space, is what makes hand
                        annotation continue a detection rather than sit beside
                        it. See adopt_detections().

The detector's untouched output is copied aside to mask_results_preds.detector.json,
and a merge re-reads that copy whenever mask_results_preds.json already carries
manual entries. So this is idempotent -- edit the boxes, run it again, and the
result is the same as if it had merged once -- while a fresh stage-1 run, which
overwrites mask_results_preds.json with detections and no manual entries, is
picked up as the new detector output rather than being discarded in favour of a
stale copy.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils

ROOT = Path(__file__).resolve().parent

PRISTINE = "mask_results_preds.detector.json"
MERGED = "mask_results_preds.json"
MANUAL = "manual_boxes.json"


def load_sam():
    """SAM alone -- no GroundingDINO.

    grounding_sam.py builds both together because it needs both; this needs only
    the box-prompted segmenter, and loading a DINO checkpoint to not use it costs
    a GPU minute and several GB per run.
    """
    sys.path.append(str(ROOT / "Grounded-Segment-Anything"))
    ckpt = ROOT / "Grounded-Segment-Anything" / "sam_vit_h_4b8939.pth"
    if not ckpt.exists():
        raise SystemExit(f"error: no SAM checkpoint at {ckpt}\n"
                         f"       re-run with --no-sam to use box rectangles instead.")

    # Announced before the import, not after: torch alone is ~25 s here and the
    # checkpoint is 2.4 GB off network storage, so a silent minute is the normal
    # case and reads exactly like a hang.
    print("loading torch + SAM (~1 min; the checkpoint is 2.4 GB)...", flush=True)
    import torch
    from segment_anything import SamPredictor, build_sam

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print(f"warning: no GPU on {os.uname().nodename}; the ViT-H encoder runs once per\n"
              f"         annotated frame and takes minutes each on CPU. Either --no-sam,\n"
              f"         or let stage 1c do this inside the sbatch'd re-render, where it\n"
              f"         has a GPU and takes seconds.", flush=True)
    sam = build_sam(checkpoint=str(ckpt))
    sam.to(device=device)
    print(f"SAM ready on {device}", flush=True)
    return SamPredictor(sam), torch, device


def expand_tracks(manual, frames):
    """Keyframed tracks -> ({frame: [(track_id, category, box), ...]}, gone, reject).

    `gone` is {track id: {frame index, ...}}, the null keyframes -- the frames
    where the annotator said the object is not there. adopt_detections() reads
    it so a track cannot reach across a gap someone opened on purpose.

    `reject` is the ids of the tracks marked "reject": regions that delete
    detections rather than adding one. They are interpolated and keyframed like
    any other track -- a false positive that drifts across the frame is two
    keyframes, not forty -- which is why they live here rather than in a
    separate list of per-frame rectangles.

    Linear in both corners, which is what makes a straight approach or a steady
    pass across the frame need two keyframes rather than sixty. A turn or an
    occlusion needs a keyframe where the motion changes, same as any keyframed
    animation.
    """
    index = {f: i for i, f in enumerate(frames)}
    per_frame, gone, reject = {}, {}, set()
    for track in manual["tracks"]:
        keys = sorted(((index[f], box) for f, box in track["keyframes"].items()
                       if f in index), key=lambda kv: kv[0])
        gone[track["id"]] = {i for i, box in keys if box is None}
        if track.get("reject"):
            reject.add(track["id"])
        if not keys:
            continue
        for (i0, b0), (i1, b1) in zip(keys, keys[1:]):
            if b0 is None:
                continue                       # gone from i0 until the next real key
            per_frame.setdefault(frames[i0], []).append(
                (track["id"], track.get("category"), b0))
            if b1 is None:
                continue                       # b0's run ends at i0, no fill
            for i in range(i0 + 1, i1):
                t = (i - i0) / (i1 - i0)
                box = [a + t * (b - a) for a, b in zip(b0, b1)]
                per_frame.setdefault(frames[i], []).append(
                    (track["id"], track.get("category"), box))
        last_i, last_box = keys[-1]
        if last_box is not None:
            per_frame.setdefault(frames[last_i], []).append(
                (track["id"], track.get("category"), last_box))
    return per_frame, gone, reject


def rect_rle(box, height, width):
    mask = np.zeros((height, width), dtype=np.uint8)
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(width, x1), min(height, y1)
    mask[y0:y1, x0:x1] = 1
    return mask


def encode(mask):
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def iou_xyxy(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (area_a + area_b - inter)


def det_box(det):
    x, y, w, h = mask_utils.toBbox(det["mask"]).tolist()
    return [x, y, x + w, y + h]


def image_size(detections):
    """(height, width) off any detection's RLE, or None if there are none.

    Read from a mask rather than a frame because adoption compares boxes and
    nothing else: opening 60 JPEGs to learn a number every RLE already carries
    would be the slowest line in the file.
    """
    for det in detections:
        size = det.get("mask", {}).get("size")
        if size:
            return int(size[0]), int(size[1])
    return None


def clip_box(box, hw):
    if hw is None:
        return list(box)
    height, width = hw
    return [max(0.0, min(width, box[0])), max(0.0, min(height, box[1])),
            max(0.0, min(width, box[2])), max(0.0, min(height, box[3]))]


def suppress_detections(detections, per_frame, reject_ids, frames, hw, iou_min):
    """Drop the detections a drawn box or a reject region covers.

    Both do the same thing to a detection and differ only in what they leave
    behind: a drawn box replaces it, a reject region simply deletes it. Which
    one killed a box is worth reporting separately -- a reject that fires on
    nothing means the region is in the wrong place, and a count that lumps it in
    with the drawn boxes cannot show that -- so they are counted apart.

    Lifted out of the masking loop and run ahead of it because everything after
    it should see only the detections that survived. Adoption especially: giving
    a track's identity, and with it the --min_track_len exemption, to a box that
    is about to be thrown away is the one ordering mistake here that would be
    invisible in the output.
    """
    by_frame = {}
    for det in detections:
        by_frame.setdefault(det["frame"], []).append(det)

    killed, by_drawn, by_reject = set(), 0, 0
    for frame, items in per_frame.items():
        for det in by_frame.get(frame, []):
            box = det_box(det)
            worst_reject = worst_drawn = 0.0
            for tid, _category, manual_box in items:
                overlap = iou_xyxy(box, clip_box(manual_box, hw))
                if tid in reject_ids:
                    worst_reject = max(worst_reject, overlap)
                else:
                    worst_drawn = max(worst_drawn, overlap)
            # Reject tested first, so a detection covered by both is reported as
            # deleted rather than replaced. That is the annotator's more
            # specific instruction: a drawn box says "it is really here", a
            # reject says "nothing here is real", and the second is only ever
            # drawn deliberately.
            if worst_reject >= iou_min:
                killed.add(id(det)); by_reject += 1
            elif worst_drawn >= iou_min:
                killed.add(id(det)); by_drawn += 1
    return ([d for d in detections if id(d) not in killed], by_drawn, by_reject)


def adopt_detections(detections, per_frame, gone, reject_ids, frames, hw,
                     adopt_iou, max_gap):
    """Give a drawn track's identity to the detections that continue it.

    The problem this solves: a detector that catches an object in glimpses
    produces one- and two-frame tracks, and smooth_boxes --min_track_len
    deletes those as flicker. Drawing boxes over the frames it missed does not
    save them -- a drawn box and a detected box never merge in the 3D
    associator, deliberately, because splicing them there means guessing an
    identity from a lifted position that is itself a guess. So the glimpse
    stays a singleton and still dies, and annotating the clip leaves it no
    better off than before.

    Identity is much easier to establish here than downstream. This runs in
    image space, on the detector's own 2D masks and the boxes a person drew,
    before any of it is lifted through a depth map. If a detection overlaps
    where the track was one frame ago, it is the same object; there is nothing
    to infer about depth, scale or heading to decide that.

    So each track is swept forward from its last drawn box and backward from
    its first, and a detection overlapping the track's most recent box by
    `adopt_iou` is claimed: it keeps its own mask and score -- it is still a
    measurement, and a better one than an interpolated box -- and gains the
    track id, which is what buys it the --min_track_len exemption downstream.
    An adopted box then becomes the track's most recent box, so a run of
    glimpses is walked one frame at a time rather than all being matched
    against a keyframe that is receding into the past.

    Three things bound it:

      max_gap       how many frames the track may go unseen before its last
                    box is too stale to match against. 0 turns adoption off.
      reject_ids    a reject region blocks adoption over what it covers, and
                    never adopts anything itself. Without this a rejected false
                    positive could be pulled back in through the side door by a
                    neighbouring track, wearing that track's identity.
      gone          a null keyframe drops the track's tip, so an annotator who
                    marked an object absent is not overruled by a detection.

    Claimed detections are stamped in place. Returns one record per adoption
    for reporting.
    """
    index = {f: i for i, f in enumerate(frames)}
    by_frame = {}
    for det in detections:
        by_frame.setdefault(det["frame"], []).append(det)

    drawn, rejected_at, category = {}, {}, {}
    for frame, items in per_frame.items():
        fi = index[frame]
        drawn[fi] = {tid: clip_box(box, hw) for tid, _, box in items
                     if tid not in reject_ids}
        rejected_at[fi] = [clip_box(box, hw) for tid, _, box in items
                           if tid in reject_ids]
        category.update({tid: cat for tid, cat, _ in items if tid not in reject_ids})

    claimed = {}          # id(det) -> record, so the two sweeps do not fight

    def sweep(order):
        tips = {}         # track id -> (frame index, box) of its most recent box
        for fi in order:
            here = drawn.get(fi, {})
            for tid, box in here.items():
                tips[tid] = (fi, box)
            for tid, empty in gone.items():
                if fi in empty:
                    tips.pop(tid, None)
            if not tips:
                continue
            free = []
            veto = rejected_at.get(fi) or []
            for det in by_frame.get(frames[fi], []):
                if id(det) in claimed:
                    continue
                box = det_box(det)
                # Everything a drawn box overlaps hard enough to suppress is
                # already gone -- suppress_detections ran first -- so the only
                # gate left here is the reject regions, which bite at the
                # adoption threshold rather than the suppression one: a region
                # drawn to delete something should win the ties.
                if veto and max(iou_xyxy(box, b) for b in veto) >= adopt_iou:
                    continue
                free.append((det, box))
            if not free:
                continue
            pairs = []
            for tid, (tip_fi, tip_box) in tips.items():
                if tid in here or abs(fi - tip_fi) > max_gap:
                    continue
                for det, box in free:
                    overlap = iou_xyxy(box, tip_box)
                    if overlap >= adopt_iou:
                        pairs.append((overlap, tid, tip_fi, det, box))
            # Greedy on overlap, one detection per track per frame and one
            # track per detection -- the same assignment smooth_boxes.associate
            # makes, for the same reason: two candidates that both pass the gate
            # are resolved by which fits better, not by dict order.
            pairs.sort(key=lambda p: -p[0])
            used_t, used_d = set(), set()
            for overlap, tid, tip_fi, det, box in pairs:
                if tid in used_t or id(det) in used_d:
                    continue
                used_t.add(tid); used_d.add(id(det))
                claimed[id(det)] = {
                    "frame": frames[fi], "track": tid, "iou": overlap,
                    "from": frames[tip_fi], "score": det["score"],
                    "was": det["category"], "now": category[tid],
                }
                det["track"] = tid
                det["category"] = category[tid]
                tips[tid] = (fi, box)

    if max_gap > 0:
        sweep(range(len(frames)))                   # after the last drawn box
        sweep(range(len(frames) - 1, -1, -1))       # and before the first
    return sorted(claimed.values(), key=lambda r: r["frame"])


def report_adoptions(records, adopt_iou, max_gap, limit=12):
    if max_gap <= 0:
        print("adoption off (--adopt_gap 0)")
        return
    print(f"adopted {len(records)} detector box(es) into "
          f"{len({r['track'] for r in records})} drawn track(s) "
          f"(IoU >= {adopt_iou} with the track's last box, gap <= {max_gap} frames)")
    for r in records[:limit]:
        relabel = "" if r["was"] == r["now"] else f"  [{r['was']} -> {r['now']}]"
        print(f"    {r['frame']}  score {r['score']:.2f}  -> track #{r['track']}  "
              f"(IoU {r['iou']:.2f} with {r['from']}){relabel}")
    if len(records) > limit:
        print(f"    ... and {len(records) - limit} more")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output_dir", required=True,
                    help="run directory holding mask_results_preds.json and manual_boxes.json")
    ap.add_argument("--frames_dir", default=None,
                    help="frame images (default: <output_dir>/../frames)")
    ap.add_argument("--iou", type=float, default=0.5,
                    help="drop a detector box overlapping a manual box by at least this")
    ap.add_argument("--adopt_iou", type=float, default=0.3,
                    help="give a drawn track's identity to a detection overlapping its "
                         "most recent box by at least this, on the frames the track "
                         "does not cover")
    ap.add_argument("--adopt_gap", type=int, default=5,
                    help="how many frames a track may go unseen before its last box is "
                         "too stale to adopt against. 0 disables adoption")
    ap.add_argument("--no-sam", dest="use_sam", action="store_false",
                    help="use the filled box rectangle as the mask instead of prompting SAM")
    ap.add_argument("--dry-run", action="store_true", help="report, write nothing")
    args = ap.parse_args()

    output_dir = Path(args.output_dir).resolve()
    frames_dir = Path(args.frames_dir).resolve() if args.frames_dir \
        else output_dir.parent / "frames"
    manual_path = output_dir / MANUAL
    if not manual_path.exists():
        print(f"no {MANUAL} in {output_dir}; nothing to merge")
        return
    if not frames_dir.is_dir():
        raise SystemExit(f"error: no frames at {frames_dir}")

    frames = sorted(p.name for p in frames_dir.iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    manual = json.loads(manual_path.read_text())

    # Always merge from the detector's own output, never from a previous merge:
    # suppression is lossy, so merging a merge would compound it.
    pristine, merged = output_dir / PRISTINE, output_dir / MERGED
    if not merged.exists():
        raise SystemExit(f"error: no {MERGED} in {output_dir} -- run stage 1 first")
    current = json.loads(merged.read_text())

    # Whether mask_results_preds.json is a previous merge or a fresh stage-1
    # run is decided by whether it carries manual entries, not by whether a
    # pristine copy happens to exist. Re-running stage 1 overwrites the merged
    # file with new detections and knows nothing about the copy beside it, so
    # trusting a stale copy there would quietly discard every detection stage 1
    # had just found. A file with no manual entries *is* the detector's output,
    # so it becomes the new pristine copy.
    if any(d.get("manual") for d in current):
        if not pristine.exists():
            raise SystemExit(
                f"error: {MERGED} already holds manual boxes but {PRISTINE} is missing, "
                f"so the detector's own output cannot be recovered.\n"
                f"       re-run stage 1 (RUN_STAGE1_MASKS=1) to regenerate it.")
        detections = json.loads(pristine.read_text())
        source = f"{PRISTINE} ({MERGED} is a previous merge)"
    else:
        detections = current
        source = MERGED
        if not args.dry_run:
            # copy, not move: mask_results_preds.json may be a symlink into
            # another run (link_reused), and that run's copy is not ours to move.
            shutil.copyfile(merged, pristine)
    detections = [d for d in detections if not d.get("manual")]
    print(f"detector boxes read from {source}: {len(detections)}")

    per_frame, gone, reject_ids = expand_tracks(manual, frames)
    if not per_frame:
        print("manual_boxes.json has no usable keyframes; nothing to merge")
        return

    # Both passes run before any mask is made: they compare boxes and nothing
    # else, and a detection either survives with the mask the detector already
    # gave it or does not survive at all. Suppression first -- see its docstring
    # for why the order is load-bearing.
    hw = image_size(detections)
    detections, dropped, rejected = suppress_detections(
        detections, per_frame, reject_ids, frames, hw, args.iou)
    adopted = adopt_detections(detections, per_frame, gone, reject_ids, frames,
                               hw, args.adopt_iou, args.adopt_gap)

    predictor = torch = device = None
    if args.use_sam and not args.dry_run:
        predictor, torch, device = load_sam()

    import cv2
    from tqdm import tqdm

    new_entries, fills = [], []
    # Drawn boxes only: a reject region contributes nothing to mask, and a frame
    # holding nothing but reject regions has no work here at all. Only the
    # annotated frames are counted by the bar rather than the whole clip -- on a
    # 60-frame clip with 7 annotated frames this is 7 long steps, and a bar that
    # sat at 7/60 forever would misreport it as nearly stalled.
    drawn_at = {f: [b for b in items if b[0] not in reject_ids]
                for f, items in per_frame.items()}
    todo = [f for f in frames if drawn_at.get(f)]
    for frame in tqdm(todo, desc="masking", unit="frame"):
        boxes = drawn_at[frame]
        image = cv2.imread(str(frames_dir / frame))
        if image is None:
            print(f"warning: cannot read {frame}; skipping its {len(boxes)} box(es)")
            continue
        height, width = image.shape[:2]
        clipped = [[max(0.0, min(width, b[0])), max(0.0, min(height, b[1])),
                    max(0.0, min(width, b[2])), max(0.0, min(height, b[3]))]
                   for _, _, b in boxes]

        masks = None
        if predictor is not None:
            predictor.set_image(image[:, :, ::-1])       # SAM wants RGB
            t = predictor.transform.apply_boxes_torch(
                torch.tensor(clipped, dtype=torch.float, device=device), (height, width))
            with torch.no_grad():
                out, _, _ = predictor.predict_torch(point_coords=None, point_labels=None,
                                                    boxes=t, multimask_output=False)
            masks = out[:, 0].cpu().numpy()

        for i, (track_id, category, _) in enumerate(boxes):
            box = clipped[i]
            mask = masks[i] if masks is not None else rect_rle(box, height, width)
            area = float(mask.sum())
            box_area = max(1.0, (box[2] - box[0]) * (box[3] - box[1]))
            fills.append(area / box_area)
            if area < 1:
                # SAM occasionally returns nothing on a blown-out region, which
                # is exactly the footage this tool exists for. The rectangle is
                # a worse mask than a real one and a much better one than none.
                mask = rect_rle(box, height, width)
                print(f"note: SAM returned an empty mask on {frame} track #{track_id}; "
                      f"using the box rectangle")
            new_entries.append({
                "frame": frame,
                "category": category,
                "mask": encode(mask),
                "score": 1.0,
                "manual": True,
                "track": track_id,
            })

    result = detections + new_entries
    result.sort(key=lambda d: (d["frame"], -d["score"]))

    n_tracks = len({e["track"] for e in new_entries})
    print(f"{len(new_entries)} manual box(es) over {len(todo)} frame(s), "
          f"{n_tracks} track(s)")
    print(f"suppressed {dropped} detector box(es) at IoU >= {args.iou}")
    if reject_ids:
        reject_frames = sum(1 for items in per_frame.values()
                            if any(tid in reject_ids for tid, _, _ in items))
        print(f"rejected {rejected} detector box(es) with "
              f"{len(reject_ids)} reject region(s) over {reject_frames} frame(s)")
        if not rejected:
            print("    -- no detection fell inside a reject region. Either the "
                  "detector no longer\n       fires there, or the region is not "
                  "over what it was meant to delete.")
    report_adoptions(adopted, args.adopt_iou, args.adopt_gap)
    if fills:
        fills = np.array(fills)
        print(f"mask fill (mask area / box area): median {np.median(fills):.2f}, "
              f"min {fills.min():.2f}  -- a very low value means SAM latched onto "
              f"part of the object, not all of it")
    print(f"{len(result)} detection(s) total ({len(detections)} from the detector)")

    if args.dry_run:
        print("dry run: nothing written")
        return
    # os.replace over a symlink replaces the link itself, so a run that borrowed
    # its masks with link_reused gets its own merged copy here and the run it
    # borrowed from keeps its pristine one.
    tmp = merged.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2))
    os.replace(tmp, merged)
    print(f"wrote {merged}")
    print(f"detector-only copy kept at {pristine}")


if __name__ == "__main__":
    main()
