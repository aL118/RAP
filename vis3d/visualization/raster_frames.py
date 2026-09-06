"""
Thin sibling of raster.py for nightcrash-style footage: raster.py's
build_scenario_dict/render_rasterized_view assume a navsim Scene/Frame
carrying real ego pose, camera calibration, and an HD map -- none of which
exist for raw video frames -- and it imports ScenarioRenderer from the
installed navsim package (so it runs in the navsim/DrivoR env). This instead
draws lift_frames_to_3d.py's boxes_3d.json with renderer.py's cuboid
rasterizer directly, so it has no navsim dependency and can run in the same
env as infer_frames.py/lift_frames_to_3d.py (vis3d).

It deliberately does *not* go through ScenarioRenderer. That class renders
through navsim's fixed CAM_F0 rig (fx=fy=1545 on a 1920x1120 canvas, plus a
hardcoded 2 m back / 0.8 m up shift of the camera in observe()), which has
nothing to do with the camera UniDepth's point map -- and therefore every
lifted box -- is expressed in. Rendering through it put boxes hundreds of
pixels off their own masks, worst for near objects where the translation is
a large fraction of the depth; the closest car in a frame typically ended up
dragged off the bottom edge entirely. So the camera comes from boxes_3d.json's
per-frame "camera" entry instead, and the canvas is the frame's own size.
Ego is still assumed stationary at the origin facing forward every frame (no
IMU/steering data to do better). The map features that do have a source here
are detected per frame rather than queried: traffic lights come from the object
detector (carrying a measured position and a state read off the frame's own
pixels, where the navsim path places them at a fixed height over their lane
connector), and lane markings from detect_lanes.py, lifted onto a fitted ground
plane by lanes.py. Crosswalks, walkways and road boundaries still have none.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# run_render_vis3d.sh runs this as `cd visualization && python raster_frames.py`,
# so the interpreter's path starts at visualization/ and vis3d/ is not on it.
# Added here rather than left to renderer.py's own insert: this module imports
# box_schema first, and an import order is a bad thing to depend on.
sys.path.insert(0, str(ROOT))
import box_schema  # noqa: E402  (needs the path above)

from renderer import (COLOR_TABLE, draw_cuboids_with_occlusion, draw_polyline_depth,
                      draw_traffic_light, save_as_video)

# --- debug overlay ------------------------------------------------------------
# Colours are BGR: the debug marks are drawn straight onto the blended overlay
# (which is BGR, unlike render_frame's RGB canvas) so they stay full-opacity
# instead of being washed out by OVERLAY_ALPHA along with the boxes.
DEBUG_SIGHT_BGR = (0, 190, 255)      # amber: ego -> object ground track
DEBUG_HEADING_BGR = (80, 255, 80)    # green: fitted facing direction
DEBUG_COLLAPSED_BGR = (60, 60, 255)  # red: facing has collapsed onto the sight line
DEBUG_TEXT_BGR = (255, 255, 255)

# Sight lines start here rather than at the ego origin, which sits directly
# below the camera and therefore has camera-frame z = 0: it projects to
# infinity, so a line anchored there has no drawable start point.
DEBUG_SIGHT_START_M = 2.0

# Segments are clipped to this depth before projecting. Not 0: a point at
# z -> 0 projects to ~1e6 px, and passing that to cv2.line is pointless work.
DEBUG_NEAR_M = 0.5

# A heading this close to the object's own bearing from the camera is drawn red.
# It means minAreaRect elected the viewing ray as the box's long axis instead of
# the vehicle -- the box is a rod pointed at the camera, which reprojects far too
# small and slides off its object as the fit wobbles. Matches
# lift_frames_to_3d.py's HEADING_COLLAPSE_DEG, which reports the same thing
# numerically at the end of a lifting run.
DEBUG_COLLAPSE_DEG = 10.0


def render_frame(boxes, camera, traffic_lights=(), lanes=()) -> np.ndarray:
    """Rasterizes `boxes`, `traffic_lights` and `lanes` (all ego frame) onto a
    black canvas through `camera`, boxes_3d.json's per-frame camera entry.

    Returns RGB: draw_cuboids_with_occlusion's base_face_colors are RGB triples
    (they are the #F72585..#3A0CA3 palette, whatever its "BGR 格式" comment
    says) handed straight to cv2.fillConvexPoly, so the canvas comes back with
    channels in RGB order and callers must swap before cv2.imwrite. The
    traffic-light colours are COLOR_TABLE's, which are RGB as well.
    """
    height, width = camera["image_hw"]
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    ego_to_camera = np.asarray(camera["ego_to_camera"], dtype=np.float64)
    intrinsics = np.asarray(camera["intrinsics"], dtype=np.float64)

    # Lights, then map, then boxes -- the same order ScenarioRenderer.observe
    # draws them in, so a vehicle occludes the lane markings and lights behind it.
    for light in traffic_lights:
        draw_traffic_light(canvas, light["position"], ego_to_camera, intrinsics,
                           light["state"])
    for lane in lanes:
        draw_polyline_depth(canvas, np.asarray(lane, dtype=np.float32), ego_to_camera,
                            intrinsics, COLOR_TABLE["lanelines"], radius=2)
    draw_cuboids_with_occlusion(
        canvas,
        box_schema.to_array(boxes).astype(np.float32),
        ego_to_camera,
        intrinsics,
    )
    return canvas


def _project_ego(points_ego, ego_to_camera: np.ndarray, intrinsics: np.ndarray):
    """(N, 3) ego-frame points -> (N, 2) pixel coords and their camera-frame depths."""
    points = np.asarray(points_ego, dtype=np.float64).reshape(-1, 3).T
    camera = ego_to_camera[:3, :3] @ points + ego_to_camera[:3, 3:4]
    depth = camera[2]
    uv = (intrinsics @ camera)[:2] / np.maximum(depth, 1e-9)
    return uv.T, depth


def _draw_segment(image_bgr, start_ego, end_ego, ego_to_camera, intrinsics,
                  color, thickness=2, arrow=False):
    """Draws one ego-frame 3D segment, clipped to DEBUG_NEAR_M.

    Written here rather than reusing renderer.draw_heading_arrow because that
    one bails whenever an endpoint falls outside the image (`valid.all()`),
    which drops exactly the close, large objects this overlay exists to
    diagnose. cv2 clips off-image endpoints itself, so clipping in depth is
    the only clipping actually needed.
    """
    start, end = np.asarray(start_ego, float), np.asarray(end_ego, float)
    (_, _), depths = _project_ego([start, end], ego_to_camera, intrinsics)
    z0, z1 = depths
    if z0 < DEBUG_NEAR_M and z1 < DEBUG_NEAR_M:
        return
    # Both endpoints in front after clipping: move whichever is behind the near
    # plane along the segment until it reaches it.
    if z0 < DEBUG_NEAR_M:
        start = start + (end - start) * (DEBUG_NEAR_M - z0) / (z1 - z0)
    elif z1 < DEBUG_NEAR_M:
        end = start + (end - start) * (DEBUG_NEAR_M - z0) / (z1 - z0)
    uv, _ = _project_ego([start, end], ego_to_camera, intrinsics)
    p0, p1 = (tuple(np.clip(p, -1e5, 1e5).astype(int)) for p in uv)
    if arrow:
        cv2.arrowedLine(image_bgr, p0, p1, color, thickness, tipLength=0.3)
    else:
        cv2.line(image_bgr, p0, p1, color, thickness, cv2.LINE_AA)


def _draw_label(image_bgr, text, position_ego, ego_to_camera, intrinsics):
    """Puts `text` at an ego point, with a dark outline so it stays readable
    over both the bright raster and the photo."""
    uv, depth = _project_ego([position_ego], ego_to_camera, intrinsics)
    if depth[0] < DEBUG_NEAR_M:
        return
    u, v = uv[0]
    if not (-200 <= u <= image_bgr.shape[1] + 200 and -200 <= v <= image_bgr.shape[0] + 200):
        return
    anchor = (int(u) + 6, int(v) - 6)
    for color, thickness in (((0, 0, 0), 3), (DEBUG_TEXT_BGR, 1)):
        cv2.putText(image_bgr, text, anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    color, thickness, cv2.LINE_AA)


def draw_debug_overlay(image_bgr: np.ndarray, boxes, names, camera) -> np.ndarray:
    """Annotates a blended overlay with what the pipeline believes about each box.

    Per box: a sight line along the ground from the ego origin out to the box's
    ground track (so every line radiates from where the program thinks the
    camera is), a vertical riser from that ground point up to the box centre
    (showing how far off the ground the box was placed), and an arrow along the
    fitted heading -- drawn red when that heading has collapsed onto the sight
    line, which is the signature of a box fitted to depth smear rather than to
    an object. The label carries the fitted extent and range, so a box that
    looks small on screen can be told apart from one that *is* small in metres.

    Drawn in place on `image_bgr`, which is returned for convenience.
    """
    ego_to_camera = np.asarray(camera["ego_to_camera"], dtype=np.float64)
    intrinsics = np.asarray(camera["intrinsics"], dtype=np.float64)

    for index, box in enumerate(box_schema.to_array(boxes)):
        x, y, z, length, width, height, yaw = box
        name = names[index] if index < len(names) else "?"
        bearing = np.arctan2(y, x)
        # Folded to [0, 90]: the heading is only defined up to +-pi (it comes off
        # the longer minAreaRect edge), so 180 deg apart is the same axis.
        offset_deg = abs((np.degrees(yaw - bearing) + 90.0) % 180.0 - 90.0)
        collapsed = offset_deg < DEBUG_COLLAPSE_DEG

        _draw_segment(image_bgr, [DEBUG_SIGHT_START_M, 0.0, 0.0], [x, y, 0.0],
                      ego_to_camera, intrinsics, DEBUG_SIGHT_BGR, thickness=1)
        _draw_segment(image_bgr, [x, y, 0.0], [x, y, z],
                      ego_to_camera, intrinsics, DEBUG_SIGHT_BGR, thickness=1)
        # Arrow length is the box's own fitted length, so a rod-shaped box draws
        # a conspicuously long arrow rather than a plausible fixed-size one.
        _draw_segment(image_bgr, [x, y, z],
                      [x + np.cos(yaw) * length, y + np.sin(yaw) * length, z],
                      ego_to_camera, intrinsics,
                      DEBUG_COLLAPSED_BGR if collapsed else DEBUG_HEADING_BGR,
                      thickness=2, arrow=True)
        _draw_label(image_bgr,
                    f"{name} {length:.1f}x{width:.1f}x{height:.1f}m "
                    f"@{np.hypot(x, y):.0f}m {offset_deg:.0f}deg",
                    [x, y, z], ego_to_camera, intrinsics)
    return image_bgr


def overlay_on_frame(canvas_bgr: np.ndarray, frame_bgr: np.ndarray, alpha: float) -> np.ndarray:
    """Alpha-blends canvas_bgr (boxes rasterized on a black background, see
    render_frame) onto frame_bgr, touching only the pixels the renderer
    actually drew -- everywhere else canvas is pure black (its initial
    np.zeros canvas), so blending unconditionally would darken the whole frame
    instead of just the boxes.

    The canvas is rendered at the point map's resolution, which is the frame's
    own resolution for this pipeline; the resize is a guard for frames that
    were downscaled before UniDepth ran, not an expected path (resampling a
    rasterized overlay costs alignment, so it should not be relied on).
    """
    if canvas_bgr.shape[:2] != frame_bgr.shape[:2]:
        canvas_bgr = cv2.resize(
            canvas_bgr, (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask = np.any(canvas_bgr != 0, axis=2)
    blended = cv2.addWeighted(canvas_bgr, alpha, frame_bgr, 1 - alpha, 0)
    out = frame_bgr.copy()
    out[mask] = blended[mask]
    return out


def _canvas_hw(boxes_by_frame, frames_dir, frames):
    """(height, width) to rasterize a frame that lifted to nothing onto.

    Taken from any frame that does carry a camera, since the point map is one
    resolution for a whole run; only if no frame has one at all does this fall
    back to reading an image off disk.
    """
    for entry in boxes_by_frame.values():
        if "camera" in entry:
            return tuple(entry["camera"]["image_hw"])
    for frame in frames:
        image = cv2.imread(str(frames_dir / frame)) if frames_dir else None
        if image is not None:
            return image.shape[:2]
    raise ValueError("No camera entry and no readable frame: nothing to size a canvas from.")


def main():
    parser = argparse.ArgumentParser(
        description="Rasterize lift_frames_to_3d.py's 3D boxes onto nightcrash-style frames.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory containing boxes_3d.json (from lift_frames_to_3d.py); vis3d/ (and vis3d_overlay/, video) written here.")
    parser.add_argument("--frames_dir", type=str, default=None,
                        help="Directory of original frame images (e.g. infer_frames.py's --frames_dir). "
                             "If given, also writes vis3d_overlay/: each frame with its rasterized boxes "
                             "alpha-blended on top.")
    parser.add_argument("--overlay_alpha", type=float, default=0.6,
                        help="Opacity of the rasterized boxes in vis3d_overlay/ (0=invisible, 1=opaque).")
    parser.add_argument("--debug", action="store_true",
                        help="Annotate vis3d_overlay/ with per-box diagnostics: sight lines from "
                             "the ego origin to each object, a riser to the box centre, the fitted "
                             "heading (red when it has collapsed onto the sight line), and the "
                             "fitted extent in metres. Needs --frames_dir.")
    parser.add_argument("--save_video", action="store_true")
    args = parser.parse_args()
    if args.debug and args.frames_dir is None:
        parser.error("--debug annotates vis3d_overlay/, which is only written with --frames_dir.")

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    frames_dir = Path(args.frames_dir).resolve() if args.frames_dir else None

    with open(output_dir / "boxes_3d.json") as f:
        boxes_by_frame = json.load(f)

    vis_dir = output_dir / "vis3d"
    vis_dir.mkdir(parents=True, exist_ok=True)
    if frames_dir is not None:
        overlay_dir = output_dir / "vis3d_overlay"
        overlay_dir.mkdir(parents=True, exist_ok=True)

    # Every frame of the clip gets an output, not just the ones that lifted to
    # something. boxes_3d.json only carries a frame the detector found an object
    # or a lane in -- on snowcrash that is 318 of 407 -- and 20 more of its
    # entries are empty, so rendering its keys left 109 holes in a 407-frame
    # sequence. Nothing downstream tolerates that: frames_to_video.py stitches
    # whatever files exist at a fixed frame rate, so every hole is a jump cut,
    # and a stretch with no detections silently plays back fast.
    #
    # A frame with nothing to draw is a black canvas, which is the honest answer
    # -- the raster of a frame in which the pipeline found nothing -- and blends
    # to the untouched photograph in the overlay.
    frames = (sorted(p.name for p in frames_dir.iterdir()
                     if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
              if frames_dir is not None else sorted(boxes_by_frame))
    canvas_hw = _canvas_hw(boxes_by_frame, frames_dir, frames)

    frame_images = []
    for frame in frames:
        entry = boxes_by_frame.get(frame, {})
        # .get(): boxes_3d.json files written before traffic lights and lanes
        # were lifted carry boxes only, and still render fine.
        boxes = entry.get("boxes", [])
        traffic_lights = entry.get("traffic_lights", [])
        lanes = entry.get("lanes", [])
        if "camera" not in entry:
            if boxes or traffic_lights or lanes:
                raise KeyError(
                    f"{output_dir / 'boxes_3d.json'} has no per-frame 'camera' entry for "
                    f"{frame}. Re-run lift_frames_to_3d.py: without it there is no way to "
                    "render these boxes through the camera they were fit in (see this "
                    "module's docstring).")
            canvas = np.zeros((*canvas_hw, 3), dtype=np.uint8)
        else:
            canvas = render_frame(boxes, entry["camera"], traffic_lights, lanes)
        canvas_bgr = canvas[:, :, ::-1]
        cv2.imwrite(str(vis_dir / frame), canvas_bgr)
        if args.save_video:
            frame_images.append({"CAM_F0": canvas})  # save_as_video swaps on write

        if frames_dir is not None:
            frame_bgr = cv2.imread(str(frames_dir / frame))
            if frame_bgr is None:
                print(f"Warning: {frames_dir / frame} not found, skipping overlay for {frame}")
            else:
                overlay = overlay_on_frame(canvas_bgr, frame_bgr, args.overlay_alpha)
                if args.debug and "camera" in entry:
                    # After blending, so the annotations stay full-opacity.
                    draw_debug_overlay(overlay, boxes, entry.get("names", []),
                                       entry["camera"])
                cv2.imwrite(str(overlay_dir / frame), overlay)

    print(f"Saved {len(list(vis_dir.iterdir()))} rendered frames to {vis_dir}")
    if frames_dir is not None:
        print(f"Saved {len(list(overlay_dir.iterdir()))} overlay frames to {overlay_dir}")

    if args.save_video and frame_images:
        save_as_video(frame_images, str(output_dir / "boxes_3d.mp4"))


if __name__ == "__main__":
    main()
