#!/usr/bin/env python3
"""Stitch a directory of frame images into an mp4 video (H.264, browser-playable)."""
import glob
import os
import subprocess

import cv2

frame_dir = "/fs/nexus-projects/sim2real/aliu/RAP/data/test/street_race/1/vis3d_overlay"
output_dir = "/fs/nexus-projects/sim2real/aliu/RAP/scripts/vis3d/videos"
output_name = frame_dir.split("/")[-3]
fps = 15.0
pattern = None  # e.g. "*.jpg"; None = auto-detect *.jpg/*.jpeg/*.png


def find_frames(frame_dir, pattern):
    if pattern:
        paths = sorted(glob.glob(os.path.join(frame_dir, pattern)))
    else:
        exts = ("*.jpg", "*.jpeg", "*.png")
        paths = sorted(
            p for ext in exts for p in glob.glob(os.path.join(frame_dir, ext))
        )
    return paths


def main():
    frame_paths = find_frames(frame_dir, pattern)
    if not frame_paths:
        raise SystemExit(f"No frames found in {frame_dir}")

    video_name = output_name + ".mp4"
    output = os.path.join(output_dir, video_name)

    first = cv2.imread(frame_paths[0])
    if first is None:
        raise SystemExit(f"Failed to read first frame: {frame_paths[0]}")
    h, w = first.shape[:2]

    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        output,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    n_written = 0
    for path in frame_paths:
        frame = cv2.imread(path)
        if frame is None:
            print(f"Warning: skipping unreadable frame {path}")
            continue
        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h))
        proc.stdin.write(frame.tobytes())
        n_written += 1

    proc.stdin.close()
    proc.wait()
    print(f"Wrote {n_written} frames -> {output}")


if __name__ == "__main__":
    main()
