"""
Detects painted lane dividers in a directory of video frames with YOLOPv2,
writing one binary mask per frame for lift_frames_to_3d.py to lift, plus the
drivable-area mask from the same forward pass (lanes.py uses it to reject
markings that are not on the road).

Why a separate detector at all: stage 1's GroundingDINO+SAM is an *object*
detector, and lane markings are not objects -- they are thin, unbounded,
frequently dashed, and a box around one is meaningless. YOLOPv2 has a
segmentation head trained for exactly this (BDD100K lane lines), so it gives
a pixel mask directly.

Runs in the 'vis3d' env like everything else now. The masks are written as
PNGs rather than in mask_results_preds.json's RLE format -- a historical
consequence of this stage once living in an env without pycocotools, and kept
because lane masks are far denser than object masks, so RLE buys little.

Weights: weights/yolopv2.pt, from
https://github.com/CAIC-AD/YOLOPv2/releases/download/V0.0.1/yolopv2.pt
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[0]
DEFAULT_WEIGHTS = ROOT.parent / "weights" / "yolopv2.pt"

# The network's input size. Both must stay multiples of 32 (its stride), and
# 640x384 is what YOLOPv2 was trained and released at.
INPUT_WIDTH, INPUT_HEIGHT = 640, 384

# Letterbox padding colour, matching the YOLOP/YOLOv5 convention the model was
# trained with; grey rather than black so the padding doesn't read as road.
PAD_COLOR = 114

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _letterbox(image: np.ndarray):
    """Resizes `image` into INPUT_WIDTH x INPUT_HEIGHT preserving aspect ratio,
    padding the remainder. Returns (letterboxed, (x, y, w, h)) where the tuple
    is the region the frame actually occupies, so the mask can be cropped back
    out exactly -- the padding is not necessarily symmetric (an odd leftover
    puts one more row or column on one side)."""
    height, width = image.shape[:2]
    scale = min(INPUT_WIDTH / width, INPUT_HEIGHT / height)
    resized_w, resized_h = int(round(width * scale)), int(round(height * scale))
    pad_x = (INPUT_WIDTH - resized_w) // 2
    pad_y = (INPUT_HEIGHT - resized_h) // 2

    canvas = np.full((INPUT_HEIGHT, INPUT_WIDTH, 3), PAD_COLOR, np.uint8)
    canvas[pad_y:pad_y + resized_h, pad_x:pad_x + resized_w] = cv2.resize(image, (resized_w, resized_h))
    return canvas, (pad_x, pad_y, resized_w, resized_h)


def detect_lane_mask(model, image_bgr: np.ndarray, threshold: float):
    """Returns (lane-line mask, drivable-area mask) for `image_bgr`, each a
    uint8 0/255 array at the frame's own resolution.

    The drivable-area head comes free with the forward pass and is what tells a
    lane divider from the kerb beside it: both are long, thin, pale and parallel
    to the road, so no amount of reasoning about a lane mask alone separates
    them, and on wrongway/4 the kerb is segmented consistently enough to survive
    a temporal track. What it is not is road. See lanes.DRIVABLE_MASK_MARGIN_PX.
    """
    height, width = image_bgr.shape[:2]
    letterboxed, (pad_x, pad_y, resized_w, resized_h) = _letterbox(image_bgr)

    tensor = torch.from_numpy(letterboxed[:, :, ::-1].copy()).permute(2, 0, 1)[None]
    tensor = tensor.to(DEVICE).float() / 255.0
    with torch.no_grad():
        # YOLOPv2 returns (detections, drivable-area seg, lane-line seg).
        _, drivable, lane_line = model(tensor)

    def _to_frame(logits, channels_first_argmax):
        # The drivable head is two-channel (background, road) and is read by
        # argmax; the lane head is one-channel and is thresholded.
        array = (logits[0].float().cpu().numpy().argmax(axis=0) > 0
                 if channels_first_argmax
                 else logits[0, 0].float().cpu().numpy() > threshold)
        array = array[pad_y:pad_y + resized_h, pad_x:pad_x + resized_w]
        return cv2.resize(array.astype(np.uint8), (width, height),
                          interpolation=cv2.INTER_NEAREST) * 255

    return _to_frame(lane_line, False), _to_frame(drivable, True)


def detect_frames(model, frames_dir: Path, output_dir: Path, threshold: float) -> None:
    frame_paths = sorted(
        p for p in frames_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    assert frame_paths, f"No frame images found in {frames_dir}"

    masks_dir = output_dir / "lane_masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    drivable_dir = output_dir / "drivable_masks"
    drivable_dir.mkdir(parents=True, exist_ok=True)

    empty = 0
    for frame_path in frame_paths:
        image = cv2.imread(str(frame_path))
        if image is None:
            print(f"Warning: {frame_path} could not be read, skipping")
            continue
        mask, drivable = detect_lane_mask(model, image, threshold)
        cv2.imwrite(str(drivable_dir / f"{frame_path.stem}.png"), drivable)
        empty += not mask.any()
        # PNG, not the source frame's extension: these are binary masks and
        # must not be re-encoded lossily.
        cv2.imwrite(str(masks_dir / f"{frame_path.stem}.png"), mask)

    print(f"Wrote {len(frame_paths) - empty} lane masks to {masks_dir} "
          f"({empty} frames had no lane pixels).")


def main():
    parser = argparse.ArgumentParser(
        description="Detect painted lane dividers in video frames with YOLOPv2.")
    parser.add_argument("--frames_dir", type=str, required=True,
                        help="Directory containing frame images.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to write lane_masks/<frame>.png into.")
    parser.add_argument("--weights", type=str, default=str(DEFAULT_WEIGHTS),
                        help="Path to the YOLOPv2 TorchScript checkpoint.")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Lane-line probability above which a pixel counts as lane.")
    args = parser.parse_args()

    frames_dir = Path(args.frames_dir).resolve()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    model = torch.jit.load(args.weights, map_location=DEVICE).eval()
    detect_frames(model, frames_dir, output_dir, args.threshold)


if __name__ == "__main__":
    main()
