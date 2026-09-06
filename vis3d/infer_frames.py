import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
from tqdm import tqdm

from grounding_sam import GroundingSAM, plot_masks_and_boxes

ROOT = Path(__file__).resolve().parents[0]

# Same category vocabulary used for nuScenes (infer_nuscenes.py) - fits driving
# footage generally - plus 'traffic light', which nuScenes' object vocabulary
# has no equivalent of (it annotates traffic lights on the map, not as objects)
# but the rasterized view draws (see vis3d/traffic_lights.py).
DEFAULT_CATEGORIES = [
    'pedestrian', 'animal', 'car', 'motorcycle', 'bicycle', 'bus', 'truck',
    'construction', 'emergency', 'trailer', 'barrier', 'trafficcone',
    'pushable_pullable', 'debris', 'bicycle_rack', 'traffic light',
]

COLOR_PALETTE = {
    'animal': [102, 220, 225], 'barrier': [95, 179, 61], 'bicycle': [234, 203, 92],
    'bicycle_rack': [3, 98, 243], 'bus': [14, 149, 245], 'car': [6, 106, 244],
    'construction': [99, 187, 71], 'debris': [212, 153, 199], 'emergency': [188, 174, 65],
    'motorcycle': [153, 20, 44], 'pedestrian': [203, 152, 102], 'pushable_pullable': [214, 240, 39],
    'trafficcone': [121, 24, 34], 'trailer': [114, 210, 65], 'truck': [239, 39, 214],
    'traffic light': [255, 255, 255],
}


def predict_frames(grounding_sam: GroundingSAM, frames_dir: Path, output_dir: Path,
                    categories, save_vis: bool = False, box_threshold: float = 0.3, text_threshold: float = 0.25):
    """
    Run GroundingSAM detection+segmentation over every frame in a directory.

    Args:
        grounding_sam: Instance of GroundingSAM model.
        frames_dir: Directory containing frame images (e.g. extracted from a video).
        output_dir: Directory to write mask_results_preds.json (and vis/ if save_vis).
        categories: List of open-vocabulary category names to detect.
        save_vis: Whether to save annotated visualizations alongside predictions.
        box_threshold: GroundingDINO detection confidence floor.
        text_threshold: GroundingDINO phrase-grounding floor.
    """
    frame_paths = sorted(
        p for p in frames_dir.iterdir()
        if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}
    )
    assert frame_paths, f"No frame images found in {frames_dir}"

    output_preds = []

    for frame_path in tqdm(frame_paths):
        bboxes, scores, labels, masks = grounding_sam(
            str(frame_path), categories, box_threshold, text_threshold)

        for (box, score, label, mask) in zip(bboxes, scores, labels, masks):
            mask_rle = mask_utils.encode(np.array(mask[0].cpu(), order='F', dtype=np.uint8))
            mask_rle['counts'] = mask_rle['counts'].decode('utf-8')
            pred = {
                "frame": frame_path.name,
                "category": label,
                "mask": mask_rle,
                "score": float(score),
            }
            output_preds.append(pred)

        if save_vis:
            # cv2.imread returns BGR, but everything this image reaches is
            # PIL, which reads an array as RGB: plot_masks_and_boxes does
            # Image.fromarray() on it and the save below writes in RGB order.
            # Handing it BGR swapped red and blue in every annotated frame --
            # the photograph came out with an orange sky, while the overlay
            # drawn on top, built inside PIL, stayed correct. Converted once
            # here so both branches below write the same colour space.
            image = cv2.cvtColor(cv2.imread(str(frame_path)), cv2.COLOR_BGR2RGB)
            vis_path = output_dir / "vis" / frame_path.name
            os.makedirs(vis_path.parent, exist_ok=True)
            if len(bboxes) == 0:
                # A frame the detector found nothing in still gets written, as
                # itself. vis/ is read as a sequence -- flicked through, or
                # stitched at a fixed frame rate -- and a missing file there is
                # not "no detections", it is a jump cut, with a quiet stretch
                # playing back fast. Writing the plain frame says the same thing
                # without lying about the timing.
                #
                # Written through PIL rather than cv2.imwrite so it lands in the
                # same colour space as the annotated frames beside it. cv2 would
                # write this RGB array back as if it were BGR, which is how one
                # vis/ ended up with its colours shifting frame to frame
                # depending on whether the detector had found anything.
                Image.fromarray(image).save(str(vis_path))
                continue
            output_img = plot_masks_and_boxes(
                image,
                preds={"boxes": bboxes, "masks": masks, "labels": labels},
                # defaultdict: GroundingDINO labels a box with the phrase it
                # matched, which for a multi-word category can come back
                # partial ('traffic' for 'traffic light'). Only the debug
                # visualization's colour is at stake, so fall back to white
                # rather than failing the whole (GPU-expensive) pass on it.
                color_palette=defaultdict(lambda: [255, 255, 255], COLOR_PALETTE),
            )
            output_img.convert("RGB").save(str(vis_path))

    json_save_path = output_dir / "mask_results_preds.json"
    os.makedirs(json_save_path.parent, exist_ok=True)
    with open(json_save_path, "w") as f:
        json.dump(output_preds, f, indent=2)
    print(f"Saved mask predictions to {json_save_path}")


def main():
    parser = argparse.ArgumentParser(description="Run GroundingSAM inference over a directory of video frames.")
    parser.add_argument("--frames_dir", type=str, required=True,
                        help="Directory containing frame images.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to write mask_results_preds.json (and vis/ frames if --save_vis).")
    parser.add_argument("--categories", type=str, nargs="+", default=DEFAULT_CATEGORIES,
                        help="Open-vocabulary category names to detect.")
    parser.add_argument("--save_vis", action="store_true",
                        help="Save annotated (boxes+masks) copies of each frame.")
    parser.add_argument("--box_threshold", type=float, default=0.3,
                        help="GroundingDINO detection confidence floor. Raising it trades "
                             "recall for precision, but on this footage it does NOT buy "
                             "better masks: median mask fill is flat at 75-77%% across every "
                             "score bin, so 0.30 -> 0.50 discards 55%% of detections and "
                             "leaves mask quality unchanged. Leaky masks do concentrate "
                             "below 0.50 (20-27%% under 65%% fill, against 1-3%% above), so "
                             "it helps a little -- but at a steep cost in recall.")
    parser.add_argument("--text_threshold", type=float, default=0.25,
                        help="GroundingDINO phrase-grounding floor: how confidently a box "
                             "must match its category word. Raise it when objects are "
                             "labelled as the wrong category, not when masks are ragged.")
    args = parser.parse_args()

    frames_dir = Path(args.frames_dir).resolve()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    grounding_sam = GroundingSAM()
    predict_frames(grounding_sam, frames_dir, output_dir, args.categories,
                   save_vis=args.save_vis, box_threshold=args.box_threshold,
                   text_threshold=args.text_threshold)


if __name__ == "__main__":
    main()
