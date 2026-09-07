#!/usr/bin/env python
"""
Render the 3D-rasterized view (as in the RAP paper's Figures) for one scene of
an existing navsim-format dataset, alongside the real camera image for the
same viewpoint.

Unlike process_data/create_openscene_metadata.py -- which builds this from raw
nuPlan .db logs via `scenarionet`/`metadrive` (not installed in the `rap` env,
and not needed just to draw a picture) -- this reads an already-processed
scene through the same navsim SceneLoader/Scene API used everywhere else in
this repo (training, eval), and re-derives only the two things that aren't
already stored in the processed metadata: live map-feature polygons (queried
from Scene.map_api, the same nuplan-devkit map object used elsewhere) and
traffic-light positions (map_api lookup by lane_connector_id).

This can also render two kinds of "off-nominal" viewpoints used elsewhere in the
repo to train recovery behavior, by re-deriving the same scenario dict from a
different pose and re-rendering it with the same ScenarioRenderer:

  --perturb       A synthetic recovery viewpoint: the real ego pose is jittered
                  by a random xy offset + heading offset, exactly like
                  process_data/create_openscene_metadata_purturbed.py's
                  get_ego_params(perturb=True). There is no real sensor image at
                  this synthetic pose, so only the rasterization is produced.

  --cross_agent   A real cross-agent viewpoint: re-renders the scene from a
                  different real vehicle's logged pose in this same frame (its
                  detected box becomes the new "ego", and the original ego is
                  reinserted as a regular vehicle box), as in
                  process_data/create_openscene_metadata_aug.py. This uses a
                  real detected pose, not a synthetic offset -- but since only
                  the ego vehicle carries cameras in this dataset, there is
                  still no real sensor image for that vehicle's viewpoint.

Usage:
    python scripts/evaluation/visualize_3d_rasterization.py \\
        --navsim_log_path $OPENSCENE_DATA_ROOT/navsim_logs/mini \\
        --sensor_blobs_path $OPENSCENE_DATA_ROOT/mini_sensor_blobs/mini \\
        --output_dir ./raster_viz
        [--token <scene_token>]   # omit to just use the first available token
        [--camera cam_f0]         # any of cam_f0/l0/l1/l2/r0/r1/r2/b0
        [--perturb] [--perturb_xy 0.5] [--perturb_yaw_deg 15] [--seed 0]
        [--cross_agent] [--cross_agent_track_token <track_token>]
        [--separate]              # save each image on its own, unlabeled, instead of one combined labeled strip
"""
import argparse
import math
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# renderer.py itself has no scenarionet/metadrive dependency (just cv2/numpy/tqdm) --
# only create_openscene_metadata.py's *other* helpers need those, which we don't import.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "process_data"))
from helpers.renderer import ScenarioRenderer  # noqa: E402

from nuplan.common.maps.maps_datatypes import SemanticMapLayer  # noqa: E402
from shapely.geometry import Point as Point2D  # noqa: E402

from navsim.common.dataclasses import SceneFilter, SensorConfig  # noqa: E402
from navsim.common.dataloader import SceneLoader  # noqa: E402

# Placeholder box size (L, W, H) used to draw the *original* ego back in as a
# regular vehicle box when rendering from a --cross_agent viewpoint. Roughly a
# sedan's footprint (process_data/create_openscene_metadata_aug.py's own
# ORIGINAL_EGO_DIMS uses H=5, which looks like a typo -- a car isn't 5m tall --
# so this uses a more plausible height instead).
ORIGINAL_EGO_DIMS = np.array([4.6, 1.8, 1.7], dtype=np.float32)


def _camera_only_sensor_config() -> SensorConfig:
    """
    Cameras, no lidar. Scene.from_scene_dict_list hardcodes lidar_path=None when
    building Lidar, so requesting 'lidar_pc' (as SensorConfig.build_all_sensors()
    does) makes Lidar.from_paths do `sensor_blobs_path / None` and raise -- which
    a broad try/except upstream swallows, silently nulling out `cameras` too even
    though camera loading itself succeeded. We only need images, so just don't
    ask for lidar.
    """
    cfg = SensorConfig.build_all_sensors()
    cfg.lidar_pc = False
    return cfg


def _extract_map_features_lite(map_api, ego_pos, radius=100.0):
    """
    Minimal stand-in for create_openscene_metadata.py's extract_map_features():
    produces only what ScenarioRenderer.observe() actually reads (a 'type'
    string containing 'LANE' / 'CROSSWALK' / 'BOUNDARY' and a 'polygon' or
    'polyline' key, in ego-centered / globally-oriented 2D coords -- see
    _boxes_ego_to_world's docstring for why "globally-oriented"). Skips the
    original's MetaDrive-format bookkeeping (lane connectivity, left/right
    neighbor edges), which is irrelevant to just drawing a picture.
    """
    ret = {}
    ex, ey = ego_pos[0], ego_pos[1]
    center = Point2D(ex, ey)
    layers = [
        SemanticMapLayer.LANE,
        SemanticMapLayer.LANE_CONNECTOR,
        SemanticMapLayer.CROSSWALK,
        SemanticMapLayer.WALKWAYS,
    ]
    nearby = map_api.get_proximal_map_objects(center, radius, layers)

    for layer in (SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR):
        for lane in nearby[layer]:
            path = lane.baseline_path.discrete_path
            pts = np.array([[p.x - ex, p.y - ey] for p in path], dtype=np.float32)
            ret[f"lane_{lane.id}"] = {"type": "LANE_SURFACE_STREET", "polygon": pts}

    for area in nearby[SemanticMapLayer.CROSSWALK]:
        xs, ys = area.polygon.exterior.xy
        pts = np.array([[x - ex, y - ey] for x, y in zip(xs, ys)], dtype=np.float32)
        ret[f"crosswalk_{area.id}"] = {"type": "CROSSWALK", "polygon": pts}

    for area in nearby[SemanticMapLayer.WALKWAYS]:
        xs, ys = area.polygon.exterior.xy
        pts = np.array([[x - ex, y - ey] for x, y in zip(xs, ys)], dtype=np.float32)
        ret[f"walkway_{area.id}"] = {"type": "BOUNDARY_SIDEWALK", "polyline": pts}

    return ret


def _traffic_light_position(map_api, lane_connector_id, ego_pos, target_position=8.0):
    """Inline port of create_openscene_metadata.py's set_light_position (nuplan-devkit only)."""
    lane = map_api.get_map_object(str(lane_connector_id), SemanticMapLayer.LANE_CONNECTOR)
    if lane is None:
        return None
    path = lane.baseline_path.discrete_path
    acc_length = 0.0
    point = path[0]
    for k in range(1, len(path)):
        prev = path[k - 1]
        acc_length += float(np.linalg.norm([path[k].x - prev.x, path[k].y - prev.y]))
        point = path[k]
        if acc_length > target_position:
            break
    return [point.x - ego_pos[0], point.y - ego_pos[1]]


def _boxes_ego_to_world(boxes, ego_yaw):
    """
    navsim's Annotations.boxes are (x, y, z, l, w, h, heading), fully in the
    ego frame: translated to the ego origin *and* rotated by -ego_yaw (see
    create_openscene_metadata.py's `locs = inv_ego_r @ (translation - ego_t)`,
    `rots = rots - ego_yaw`). ScenarioRenderer instead expects the "world"
    convention that same script's `gt_boxes_world` uses: translated to ego
    origin but NOT rotated -- world_to_camera_T applies the ego-heading
    rotation itself, via its `lidar_yaw` argument. So invert navsim's rotation
    (yaw-only; ignores any pitch/roll in the original 3D pose inverse, an
    acceptable approximation for a visualization).
    """
    boxes = boxes.copy()
    c, s = np.cos(ego_yaw), np.sin(ego_yaw)
    x, y = boxes[:, 0].copy(), boxes[:, 1].copy()
    boxes[:, 0] = c * x - s * y
    boxes[:, 1] = s * x + c * y
    boxes[:, 6] = boxes[:, 6] + ego_yaw
    return boxes


def _scenario_pose_features(scene, frame, ego_pos, ego_yaw):
    """
    The map/traffic-light part of the scenario dict, re-derived for an arbitrary
    (ego_pos, ego_yaw) rather than the real logged one -- shared by the base,
    --perturb, and --cross_agent renders below.
    """
    return {
        "ego_pos": ego_pos,
        "ego_heading": float(ego_yaw),
        "map_features": _extract_map_features_lite(scene.map_api, ego_pos),
        "traffic_lights": [
            (lane_id, is_red, pos)
            for lane_id, is_red in frame.traffic_lights
            for pos in [_traffic_light_position(scene.map_api, lane_id, ego_pos)]
            if pos is not None
        ],
    }


def _perturb_pose(ego_pos, ego_yaw, xy_range=0.5, yaw_range_deg=15.0, rng=random):
    """
    Synthetic "recovery" perturbation: a random lateral/longitudinal offset plus
    a heading jitter applied to the real ego pose, so the rasterized view shows
    an off-nominal viewpoint. Mirrors process_data/create_openscene_metadata_purturbed.py's
    get_ego_params(perturb=True) (same +/-0.5m xy, +/-15deg yaw ranges by default).
    """
    dx = rng.uniform(-xy_range, xy_range)
    dy = rng.uniform(-xy_range, xy_range)
    dyaw = math.radians(rng.uniform(-yaw_range_deg, yaw_range_deg))
    new_pos = [ego_pos[0] + dx, ego_pos[1] + dy]
    new_yaw = (ego_yaw + dyaw + math.pi) % (2 * math.pi) - math.pi
    return new_pos, new_yaw, dx, dy


def _shift_boxes_world(gt_boxes_world, dx, dy):
    """Re-express gt_boxes_world (positions relative to the real ego) relative to a
    pose offset by (dx, dy) from the real ego. Headings are world-absolute already
    (see _boxes_ego_to_world) so they don't need adjusting."""
    shifted = gt_boxes_world.copy()
    shifted[:, 0] -= dx
    shifted[:, 1] -= dy
    return shifted


def _pick_cross_agent_index(gt_names, track_tokens, track_token=None):
    if track_token is not None:
        matches = [i for i, t in enumerate(track_tokens) if t == track_token]
        if not matches:
            raise ValueError(f"track_token {track_token!r} not found among this frame's boxes")
        return matches[0]
    matches = [i for i, n in enumerate(gt_names) if n == "vehicle"]
    if not matches:
        raise ValueError("No 'vehicle' boxes in this frame to use as a --cross_agent viewpoint")
    return matches[0]


def _cross_agent_scenario(scene, frame, ego_pos, ego_yaw, gt_boxes_world, gt_names, track_tokens, track_token=None):
    """
    Real cross-agent viewpoint: swap in a different real vehicle (its logged box
    in this frame) as the new "ego", and reinsert the original ego as a regular
    vehicle box. Mirrors process_data/create_openscene_metadata_aug.py's approach,
    simplified to a single frame (the training script's multi-frame motion filter
    is irrelevant for just rendering one picture).
    """
    idx = _pick_cross_agent_index(gt_names, track_tokens, track_token)
    used_track_token = track_tokens[idx]

    veh_offset = gt_boxes_world[idx, :2].copy()  # chosen vehicle's world pos, relative to the real ego
    veh_yaw_world = float(gt_boxes_world[idx, 6])
    new_ego_pos = [ego_pos[0] + veh_offset[0], ego_pos[1] + veh_offset[1]]

    other_boxes = np.delete(gt_boxes_world, idx, axis=0)
    other_boxes[:, :2] -= veh_offset  # reproject onto the new ego's origin
    other_names = np.delete(np.array(gt_names), idx, axis=0)

    orig_ego_box = np.array(
        [[-veh_offset[0], -veh_offset[1], 0.0, *ORIGINAL_EGO_DIMS, ego_yaw]], dtype=np.float32
    )
    boxes = np.concatenate([other_boxes, orig_ego_box], axis=0)
    names = np.append(other_names, "vehicle")

    scenario = _scenario_pose_features(scene, frame, new_ego_pos, veh_yaw_world)
    scenario["anns"] = {"gt_boxes_world": boxes, "gt_names": names}
    return scenario, used_track_token


def _label_panel(image_array: np.ndarray, label: str) -> np.ndarray:
    """Stamp a caption bar across the top of an image so a multi-panel strip is
    self-explanatory without an external legend."""
    img = Image.fromarray(image_array).convert("RGB")
    draw = ImageDraw.Draw(img)
    bar_h = max(28, img.height // 22)
    font_size = max(16, int(bar_h * 0.6))
    try:
        font = ImageFont.load_default(size=font_size)
    except TypeError:  # older Pillow: load_default() takes no size argument
        font = ImageFont.load_default()
    draw.rectangle([0, 0, img.width, bar_h], fill=(0, 0, 0))
    text_bbox = draw.textbbox((0, 0), label, font=font)
    text_h = text_bbox[3] - text_bbox[1]
    draw.text((8, max(2, (bar_h - text_h) // 2)), label, fill=(255, 255, 255), font=font)
    return np.array(img)


def _labeled_row(panels) -> np.ndarray:
    """panels: list of (label, image_array). Resizes all to a common height,
    stamps each with its label, and concatenates them left-to-right."""
    h = min(img.shape[0] for _, img in panels)
    labeled = []
    for label, img in panels:
        resized = np.array(Image.fromarray(img).resize((img.shape[1] * h // img.shape[0], h)))
        labeled.append(_label_panel(resized, label))
    return np.concatenate(labeled, axis=1)


def render_token(
    token: str,
    navsim_log_path: str,
    sensor_blobs_path: str,
    camera: str = "cam_f0",
    perturb: bool = False,
    perturb_xy: float = 0.5,
    perturb_yaw_deg: float = 15.0,
    rng=random,
    cross_agent: bool = False,
    cross_agent_track_token: str = None,
):
    scene_filter = SceneFilter(num_history_frames=1, num_future_frames=0, has_route=False, tokens=[token])
    scene_loader = SceneLoader(
        data_path=Path(navsim_log_path),
        sensor_blobs_path=Path(sensor_blobs_path),
        scene_filter=scene_filter,
        sensor_config=_camera_only_sensor_config(),
        enable_filter=True,  # scan the whole log dir instead of requiring an explicit log_names list
    )
    if token not in scene_loader.tokens:
        raise ValueError(f"Token {token!r} not found under {navsim_log_path}")

    scene = scene_loader.get_scene_from_token(token)
    frame = scene.frames[-1]

    ego_x, ego_y, ego_yaw = frame.ego_status.ego_pose
    ego_pos = [float(ego_x), float(ego_y)]
    ego_yaw = float(ego_yaw)

    gt_boxes_world = _boxes_ego_to_world(frame.annotations.boxes, ego_yaw)
    gt_names = frame.annotations.names

    scenario = _scenario_pose_features(scene, frame, ego_pos, ego_yaw)
    scenario["anns"] = {"gt_boxes_world": gt_boxes_world, "gt_names": gt_names}

    renderer = ScenarioRenderer(camera_channel_list=[camera.upper()])
    channel = camera.upper()
    rendered = {"base": renderer.observe(scenario)[channel]}
    meta = {}

    if perturb:
        new_pos, new_yaw, dx, dy = _perturb_pose(ego_pos, ego_yaw, perturb_xy, perturb_yaw_deg, rng)
        perturbed_scenario = _scenario_pose_features(scene, frame, new_pos, new_yaw)
        perturbed_scenario["anns"] = {
            "gt_boxes_world": _shift_boxes_world(gt_boxes_world, dx, dy),
            "gt_names": gt_names,
        }
        rendered["perturbed"] = renderer.observe(perturbed_scenario)[channel]
        meta["perturb_dx"] = dx
        meta["perturb_dy"] = dy
        meta["perturb_dyaw_deg"] = math.degrees(new_yaw - ego_yaw)

    if cross_agent or cross_agent_track_token is not None:
        cross_scenario, used_track_token = _cross_agent_scenario(
            scene, frame, ego_pos, ego_yaw, gt_boxes_world, gt_names,
            frame.annotations.track_tokens, cross_agent_track_token,
        )
        rendered["cross_agent"] = renderer.observe(cross_scenario)[channel]
        meta["cross_agent_track_token"] = used_track_token

    real_camera = getattr(frame.cameras, camera)
    real_image = real_camera.image if real_camera.image is not None else None

    return real_image, rendered, meta


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--navsim_log_path", required=True, help="e.g. $OPENSCENE_DATA_ROOT/navsim_logs/mini")
    parser.add_argument("--sensor_blobs_path", required=True, help="e.g. $OPENSCENE_DATA_ROOT/mini_sensor_blobs/mini")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--token", default=None, help="scene token; omit to use the first available one")
    parser.add_argument("--camera", default="cam_f0", choices=["cam_f0", "cam_l0", "cam_l1", "cam_l2", "cam_r0", "cam_r1", "cam_r2", "cam_b0"])
    parser.add_argument("--perturb", action="store_true",
                         help="Also render a synthetic recovery viewpoint (random xy + yaw jitter of the real ego pose)")
    parser.add_argument("--perturb_xy", type=float, default=0.5, help="+/- meters of xy jitter for --perturb")
    parser.add_argument("--perturb_yaw_deg", type=float, default=15.0, help="+/- degrees of yaw jitter for --perturb")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for --perturb; omit for a fresh random jitter each run")
    parser.add_argument("--cross_agent", action="store_true",
                         help="Also render a real cross-agent viewpoint (a different real vehicle in this frame becomes the ego)")
    parser.add_argument("--cross_agent_track_token", default=None,
                         help="Specific vehicle track_token to use for --cross_agent; omit to use the first vehicle box in the frame")
    parser.add_argument("--separate", action="store_true",
                         help="Save each image (original, rasterized, rasterized perturbed, cross-view) as its own "
                              "unlabeled file, instead of one combined labeled strip")
    args = parser.parse_args()

    token = args.token
    if token is None:
        probe_filter = SceneFilter(num_history_frames=1, num_future_frames=0, has_route=False)
        probe_loader = SceneLoader(
            data_path=Path(args.navsim_log_path),
            sensor_blobs_path=Path(args.sensor_blobs_path),
            scene_filter=probe_filter,
            sensor_config=SensorConfig.build_no_sensors(),
            enable_filter=True,
        )
        assert len(probe_loader.tokens) > 0, f"No scenes found under {args.navsim_log_path}"
        token = probe_loader.tokens[0]
        print(f"No --token given, using first available scene: {token}")

    rng = random.Random(args.seed) if args.seed is not None else random

    real_image, rendered, meta = render_token(
        token, args.navsim_log_path, args.sensor_blobs_path, args.camera,
        perturb=args.perturb, perturb_xy=args.perturb_xy, perturb_yaw_deg=args.perturb_yaw_deg, rng=rng,
        cross_agent=args.cross_agent, cross_agent_track_token=args.cross_agent_track_token,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.separate:
        # Each panel as its own unlabeled file -- e.g. for dropping straight into
        # a figure/slide, where a baked-in caption bar isn't wanted.
        saved = []
        if real_image is not None:
            path = output_dir / f"{token}_{args.camera}_original.png"
            Image.fromarray(real_image).save(path)
            saved.append(path)

        path = output_dir / f"{token}_{args.camera}_rasterized.png"
        Image.fromarray(rendered["base"]).save(path)
        saved.append(path)

        if "perturbed" in rendered:
            path = output_dir / f"{token}_{args.camera}_rasterized_perturbed.png"
            Image.fromarray(rendered["perturbed"]).save(path)
            saved.append(path)

        if "cross_agent" in rendered:
            path = output_dir / f"{token}_{args.camera}_cross_view.png"
            Image.fromarray(rendered["cross_agent"]).save(path)
            saved.append(path)

        print(f"Saved {len(saved)} separate images to {output_dir}")
        return

    # One combined, labeled strip: real image (if any) + base rasterization + any
    # requested variants -- rather than a separate comparison file per variant.
    panels = []
    filename_suffix = ""
    if real_image is not None:
        # This is always the *original* ego's real camera image: only the ego
        # carries cameras in this dataset, so there's no real image at the
        # perturbed or cross-agent viewpoints. Labeled explicitly so it's not
        # mistaken for a real image at those viewpoints.
        panels.append(("Real (original ego view)", real_image))
    panels.append(("Rasterized (original ego view)", rendered["base"]))

    if "perturbed" in rendered:
        dx, dy, dyaw = meta["perturb_dx"], meta["perturb_dy"], meta["perturb_dyaw_deg"]
        panels.append((f"Rasterized (perturbed dx={dx:+.2f}m dy={dy:+.2f}m dyaw={dyaw:+.1f}deg)", rendered["perturbed"]))
        filename_suffix += f"_perturbed_dx{dx:+.2f}_dy{dy:+.2f}_dyaw{dyaw:+.1f}"

    if "cross_agent" in rendered:
        track_token = meta["cross_agent_track_token"]
        panels.append((f"Rasterized (cross-agent: vehicle {track_token})", rendered["cross_agent"]))
        filename_suffix += f"_crossagent_{track_token}"

    combined = _labeled_row(panels)
    path = output_dir / f"{token}_{args.camera}_comparison{filename_suffix}.png"
    Image.fromarray(combined).save(path)
    print(f"Saved {len(panels)}-panel comparison to {path}")


if __name__ == "__main__":
    main()
