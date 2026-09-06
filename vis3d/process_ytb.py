import argparse
import os
import sys

import cv2


# Extract video wrt framerate

def open_video(video_path):
    """cv2.VideoCapture never raises: handed a path that does not exist it
    returns a capture that simply is not opened, whose FPS reads back as -1.0
    and whose first read() fails. Every caller downstream then behaves as if the
    clip were empty rather than missing, so a failed download surfaces as the
    perfectly calm 'Saved 0 frames'. Fail here instead."""
    if not os.path.isfile(video_path):
        sys.exit(f"No such video: {video_path}\n"
                 f"(if process_ytb.sh just ran, check that the download actually "
                 f"succeeded -- curl does not create the target directory.)")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        sys.exit(f"Could not open {video_path} ({os.path.getsize(video_path)} bytes). "
                 f"A truncated or HTML-error-page download looks like this.")
    return cap


def fit_cover(frame, target_w, target_h, crop_bias=0.5):
    """Scale to cover target_w x target_h, then crop the overflow away.

    One scale factor for both axes, never two: a non-uniform scale changes the
    aspect ratio, which is the one mutation a pinhole camera cannot express. It
    would make fx/fy disagree with the real lens, and every angle recovered from
    the image afterwards -- headings, the depth model's own intrinsic estimate --
    would be wrong in a way no downstream stage can detect or undo. Cropping is
    lossy but honest: it throws away field of view without distorting what is
    left, which is exactly what a smaller sensor behind the same lens would see.

    crop_bias picks where the surviving strip comes from, 0.0 = top/left,
    0.5 = centre, 1.0 = bottom/right. Only the overflowing axis is cropped.
    """
    h, w = frame.shape[:2]
    if (w, h) == (target_w, target_h):
        return frame

    scale = max(target_w / w, target_h / h)
    # max() against the target as well as rounding: at some scales round() lands
    # a pixel short, and a crop window one pixel wider than the image silently
    # yields a smaller frame rather than an error.
    new_w = max(target_w, round(w * scale))
    new_h = max(target_h, round(h * scale))
    # INTER_AREA is the right filter for shrinking (it averages the pixels it
    # discards); for enlarging it degenerates to nearest-neighbour.
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    frame = cv2.resize(frame, (new_w, new_h), interpolation=interp)

    off_x = int(round((new_w - target_w) * crop_bias))
    off_y = int(round((new_h - target_h) * crop_bias))
    return frame[off_y:off_y + target_h, off_x:off_x + target_w]


def describe_fit(w, h, target_w, target_h, crop_bias):
    """What fit_cover will do to a w x h frame, as a line worth printing. A
    crop this size changes the clip's field of view, and that is not something
    to discover months later from a model that will not converge."""
    if (w, h) == (target_w, target_h):
        return f"frames already {w}x{h}; no resize"
    scale = max(target_w / w, target_h / h)
    new_w = max(target_w, round(w * scale))
    new_h = max(target_h, round(h * scale))
    cut_x, cut_y = new_w - target_w, new_h - target_h
    msg = (f"fitting {w}x{h} (aspect {w / h:.2f}) -> {target_w}x{target_h}: "
           f"scale x{scale:.3f}, crop {cut_x}px wide / {cut_y}px tall "
           f"at bias {crop_bias:g}")
    worst = max(cut_x / new_w, cut_y / new_h)
    if worst > 0.10:
        axis = "horizontal" if cut_x / new_w > cut_y / new_h else "vertical"
        msg += (f"\n  WARNING: {worst:.0%} of the {axis} field of view is cropped "
                f"away. Source aspect {w / h:.2f} vs target "
                f"{target_w / target_h:.2f}.")
    return msg


def split_video_by_hz_precise(video_path, save_dir, target_hz,
                              target_size=None, crop_bias=0.5):
    os.makedirs(save_dir, exist_ok=True)
    cap = open_video(video_path)

    period = 1.0 / target_hz
    saved = 0
    described = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if target_size and not described:
            h, w = frame.shape[:2]
            print(describe_fit(w, h, target_size[0], target_size[1], crop_bias))
            described = True

        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # current time in seconds
        
        # saved * period rather than an accumulator: adding `period` each time
        # compounds float error (0.1 is not representable), which slips the
        # sample point by a whole source frame partway through a clip and makes
        # the extracted rate not quite the one requested.
        if t >= saved * period:
            if target_size:
                frame = fit_cover(frame, target_size[0], target_size[1], crop_bias)
            out_path = os.path.join(save_dir, f"{saved:06d}.jpg")
            cv2.imwrite(out_path, frame)
            saved += 1

    cap.release()
    if saved == 0:
        sys.exit(f"Decoded 0 frames from {video_path}; the file opened but "
                 f"yielded no images, so it is very likely truncated.")
    print(f"Saved {saved} frames at exactly {target_hz} Hz.")


def parse_args():
    parser = argparse.ArgumentParser(description="Extract frames from a video at a fixed rate.")
    parser.add_argument("--video", dest="video_path", required=True,
                         help="Path to the source video file.")
    parser.add_argument("--save-dir", dest="save_dir", default=None,
                         help="Directory to save extracted frames (default: <video_dir>/frames).")
    parser.add_argument("--hz", dest="target_hz", type=float, default=2,
                         help="Target frame extraction rate in Hz. Defaults to navsim's "
                              "own 2 Hz, so a clip needs no subsampling at export.")
    parser.add_argument("--size", dest="size", default="1920x1080",
                         help="WxH to fit every frame to, by uniform scale plus a "
                              "crop of the overflow -- never a stretch. Defaults to "
                              "navsim's own CAM_F0 size. Pass 'native' to keep the "
                              "source resolution.")
    parser.add_argument("--crop-bias", dest="crop_bias", type=float, default=0.5,
                         help="Where the kept strip comes from when cropping: "
                              "0 = top/left, 0.5 = centre (default), 1 = bottom/right.")
    return parser.parse_args()


def parse_size(text):
    if text.strip().lower() in ("native", "none", ""):
        return None
    try:
        w, h = (int(v) for v in text.lower().split("x"))
        if w <= 0 or h <= 0:
            raise ValueError
    except ValueError:
        sys.exit(f"--size wants WxH (e.g. 1920x1080) or 'native', got {text!r}")
    return w, h


if __name__ == '__main__':
    args = parse_args()
    save_dir = args.save_dir or os.path.join(os.path.dirname(args.video_path), 'frames')

    # measure FPS
    cap = open_video(args.video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    print(f"source fps: {fps}")
    if not 0.0 <= args.crop_bias <= 1.0:
        sys.exit(f"--crop-bias wants 0..1, got {args.crop_bias}")
    split_video_by_hz_precise(args.video_path, save_dir, args.target_hz,
                              parse_size(args.size), args.crop_bias)