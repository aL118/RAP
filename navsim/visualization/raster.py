"""
Builds the ScenarioRenderer scenario dict (map features, traffic lights, boxes)
for a navsim Scene/Frame and renders its 3D-rasterized view.

Ported from RAP's scripts/evaluation/visualize_3d_rasterization.py: this reads
an already-processed scene through the same navsim SceneLoader/Scene API used
elsewhere in this repo, and re-derives only the two things that aren't already
stored in the processed metadata -- live map-feature polygons (queried from
Scene.map_api, the same nuplan-devkit map object used elsewhere) and
traffic-light positions (map_api lookup by lane_connector_id). See that
script's module docstring for the full rationale, in particular why this is a
"lite" re-derivation rather than a port of create_openscene_metadata.py's
extract_map_features (which bakes in MetaDrive-format bookkeeping this repo
doesn't need just to draw a picture).
"""
import numpy as np
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from shapely.geometry import Point as Point2D

from navsim.common.dataclasses import AgentInput, Frame, Scene
from navsim.visualization.renderer import ScenarioRenderer

CAMERA_CHANNELS = ("cam_f0", "cam_l0", "cam_l1", "cam_l2", "cam_r0", "cam_r1", "cam_r2", "cam_b0")


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


def build_scenario_dict(scene: Scene, frame: Frame) -> dict:
    """Builds the scenario dict ScenarioRenderer.observe() expects, from `frame`'s real (global) ego pose."""
    ego_x, ego_y, ego_yaw = frame.ego_status.ego_pose
    ego_pos = [float(ego_x), float(ego_y)]
    ego_yaw = float(ego_yaw)

    gt_boxes_world = _boxes_ego_to_world(frame.annotations.boxes, ego_yaw)

    return {
        "ego_pos": ego_pos,
        "ego_heading": ego_yaw,
        "map_features": _extract_map_features_lite(scene.map_api, ego_pos),
        "traffic_lights": [
            (lane_id, is_red, pos)
            for lane_id, is_red in frame.traffic_lights
            for pos in [_traffic_light_position(scene.map_api, lane_id, ego_pos)]
            if pos is not None
        ],
        "anns": {"gt_boxes_world": gt_boxes_world, "gt_names": frame.annotations.names},
    }


def render_rasterized_views(scene: Scene, frame: Frame, camera_channels) -> dict:
    """
    Renders the 3D-rasterized view of `frame`'s real ego pose for several camera
    channels at once. Builds the scenario dict (map query + traffic lights) only
    once and shares it across channels -- ScenarioRenderer.observe() already
    renders every configured channel in one pass, so calling this instead of
    render_rasterized_view per-channel avoids redundant map_api queries when a
    scene has multiple cameras configured.
    :param camera_channels: iterable of channel names, case-insensitive (e.g. "cam_f0")
    :return: dict of channel name (as given, not upper-cased) -> rendered image
    """
    scenario = build_scenario_dict(scene, frame)
    channels = list(camera_channels)
    renderer = ScenarioRenderer(camera_channel_list=[c.upper() for c in channels])
    rendered = renderer.observe(scenario)
    return {channel: rendered[channel.upper()] for channel in channels}


def render_rasterized_view(scene: Scene, frame: Frame, camera_channel: str = "cam_f0") -> np.ndarray:
    """Renders the 3D-rasterized view of `frame`'s real ego pose, for one camera channel."""
    return render_rasterized_views(scene, frame, [camera_channel])[camera_channel]


def rasterize_agent_input_cameras(scene: Scene, agent_input: AgentInput) -> None:
    """
    Replaces every loaded (non-None) camera image in `agent_input.cameras` with its
    3D-rasterized rendering derived from `scene`, in place.

    `agent_input.cameras[t]` is assumed to align 1:1 with `scene.frames[t]` -- true
    whenever both were built from the same SceneLoader/scene_filter for the same
    token (as in navsim.planning.training.dataset.Dataset, the only caller), since
    AgentInput.from_scene_dict_list and Scene.from_scene_dict_list both iterate the
    same underlying scene_dict_list frame-by-frame from index 0.
    """
    for t, cameras in enumerate(agent_input.cameras):
        channels = [ch for ch in CAMERA_CHANNELS if getattr(cameras, ch).image is not None]
        if not channels:
            continue
        rendered = render_rasterized_views(scene, scene.frames[t], channels)
        for channel, image in rendered.items():
            getattr(cameras, channel).image = image
