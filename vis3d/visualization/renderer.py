from __future__ import annotations

import cv2
import numpy as np
from tqdm import tqdm
from numpy import array
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import box_schema

def build_se3(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """4×4 SE(3) 齐次矩阵"""
    T = np.eye(4, dtype=np.float32)
    T[:3, :3], T[:3, 3] = R, t
    return T


COLOR_TABLE = {
    'lanelines': np.array([98, 183, 249], np.uint8),  # 浅蓝
    'lanes': np.array([56, 103, 221], np.uint8),  # 深蓝
    'road_boundaries': np.array([200, 36, 35], np.uint8),  # 深红
    'crosswalks': np.array([206, 131, 63], np.uint8),  # 土黄
    'traffic_light_red': np.array([255, 0, 0], np.uint8),  # 红
    'traffic_light_yellow': np.array([255, 255, 0], np.uint8),  # 黄
    'traffic_light_green': np.array([0, 255, 0], np.uint8),  # 绿
    'traffic_light_unknown': np.array([255, 255, 255], np.uint8),  # 白
    'pedestrian': np.array( [255, 0, 255], np.uint8),  # 青
    'vehicle': np.array([0, 128, 255], np.uint8),  # 蓝
    'bicycle': np.array([255, 255, 0], np.uint8),  # 黑
}

def rpy_to_rot(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """R = Rz(yaw) @ Ry(pitch) @ Rx(roll); equals yaw_to_rot when roll=pitch=0."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ], dtype=np.float32)


def yaw_to_rot(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]], dtype=np.float32)

def world_to_camera_T(lidar_pos, lidar_yaw,
                      cam2lidar_t, cam2lidar_R) -> np.ndarray:
    """
    构造世界到相机的齐次变换
    world ─► lidar ─► camera
    """
    T_w_lidar = build_se3(yaw_to_rot(lidar_yaw), lidar_pos)  # world→lidar
    T_cam_lidar = build_se3(cam2lidar_R, cam2lidar_t)  # cam→lidar (给定)
    T_w_cam = T_w_lidar @ T_cam_lidar  # world→cam
    return np.linalg.inv(T_w_cam)  # 取逆得 cam←world


def project_points_cam(points_cam: np.ndarray,
                       K: np.ndarray, img_hw) -> tuple[np.ndarray, np.ndarray]:
    """
    相机坐标系点集 → 像素坐标 & 可见 mask
    points_cam: (N,3)
    """
    x, y, z = points_cam.T
    eps_mask = z > 1e-3
    u = K[0, 0] * x / z + K[0, 2]
    v = K[1, 1] * y / z + K[1, 2]
    H, W = img_hw
    uv = np.stack([u, v], axis=1)
    in_img = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    valid = eps_mask & in_img
    return uv.astype(np.int32), valid


camera_params = {'CAM_F0': {'distortion': array([-0.356123, 0.172545, -0.00213, 0.000464, -0.05231]),
                            'intrinsics': array([[1.545e+03, 0.000e+00, 9.600e+02],
                                                 [0.000e+00, 1.545e+03, 5.600e+02],
                                                 [0.000e+00, 0.000e+00, 1.000e+00]]),
                            'sensor2lidar_rotation': array([[-0.00785972, -0.02271912, 0.99971099],
                                                            [-0.99994262, 0.00745516, -0.00769211],
                                                            [-0.00727825, -0.99971409, -0.02277642]]),
                            'sensor2lidar_translation': array([1.65506747, -0.01168732, 1.49112208])},
                 'CAM_L0': {'distortion': array([-0.356123, 0.172545, -0.00213, 0.000464, -0.05231]),
                            'intrinsics': array([[1.545e+03, 0.000e+00, 9.600e+02],
                                                 [0.000e+00, 1.545e+03, 5.600e+02],
                                                 [0.000e+00, 0.000e+00, 1.000e+00]]),
                            'sensor2lidar_rotation': array([[0.81776776, -0.0057693, 0.57551942],
                                                            [-0.57553938, -0.01377628, 0.81765802],
                                                            [0.0032112, -0.99988846, -0.01458626]]),
                            'sensor2lidar_translation': array([1.63069485, 0.11956747, 1.48117884])},
                 'CAM_L1': {'distortion': array([-0.356123, 0.172545, -0.00213, 0.000464, -0.05231]),
                            'intrinsics': array([[1.545e+03, 0.000e+00, 9.600e+02],
                                                 [0.000e+00, 1.545e+03, 5.600e+02],
                                                 [0.000e+00, 0.000e+00, 1.000e+00]]),
                            'sensor2lidar_rotation': array([[0.93120104, 0.00261563, -0.36449662],
                                                            [0.36447127, -0.02048653, 0.93098926],
                                                            [-0.00503215, -0.99978671, -0.0200304]]),
                            'sensor2lidar_translation': array([1.29939471, 0.63819702, 1.36736822])},
                 'CAM_L2': {'distortion': array([-0.356123, 0.172545, -0.00213, 0.000464, -0.05231]),
                            'intrinsics': array([[1.545e+03, 0.000e+00, 9.600e+02],
                                                 [0.000e+00, 1.545e+03, 5.600e+02],
                                                 [0.000e+00, 0.000e+00, 1.000e+00]]),
                            'sensor2lidar_rotation': array([[0.63520782, 0.01497516, -0.77219607],
                                                            [0.77232489, -0.00580669, 0.63520119],
                                                            [0.00502834, -0.99987101, -0.01525415]]),
                            'sensor2lidar_translation': array([-0.49561003, 0.54750373, 1.3472672])},
                 'CAM_R0': {'distortion': array([-0.356123, 0.172545, -0.00213, 0.000464, -0.05231]),
                            'intrinsics': array([[1.545e+03, 0.000e+00, 9.600e+02],
                                                 [0.000e+00, 1.545e+03, 5.600e+02],
                                                 [0.000e+00, 0.000e+00, 1.000e+00]]),
                            'sensor2lidar_rotation': array([[-0.82454901, 0.01165722, 0.56567043],
                                                            [-0.56528395, 0.02532491, -0.82450755],
                                                            [-0.02393702, -0.9996113, -0.01429199]]),
                            'sensor2lidar_translation': array([1.61828343, -0.15532203, 1.49007665])},
                 'CAM_R1': {'distortion': array([-0.356123, 0.172545, -0.00213, 0.000464, -0.05231]),
                            'intrinsics': array([[1.545e+03, 0.000e+00, 9.600e+02],
                                                 [0.000e+00, 1.545e+03, 5.600e+02],
                                                 [0.000e+00, 0.000e+00, 1.000e+00]]),
                            'sensor2lidar_rotation': array([[-0.92684778, 0.02177016, -0.37480562],
                                                            [0.37497631, 0.00421964, -0.92702479],
                                                            [-0.01859993, -0.9997541, -0.01207426]]),
                            'sensor2lidar_translation': array([1.27299407, -0.60973112, 1.37217911])},
                 'CAM_R2': {'distortion': array([-0.356123, 0.172545, -0.00213, 0.000464, -0.05231]),
                            'intrinsics': array([[1.545e+03, 0.000e+00, 9.600e+02],
                                                 [0.000e+00, 1.545e+03, 5.600e+02],
                                                 [0.000e+00, 0.000e+00, 1.000e+00]]),
                            'sensor2lidar_rotation': array([[-0.62253245, 0.03706878, -0.78171558],
                                                            [0.78163434, -0.02000083, -0.62341618],
                                                            [-0.03874424, -0.99911254, -0.01652307]]),
                            'sensor2lidar_translation': array([-0.48771615, -0.493167, 1.35027683])},
                 'CAM_B0': {'distortion': array([-0.356123, 0.172545, -0.00213, 0.000464, -0.05231]),
                            'intrinsics': array([[1.545e+03, 0.000e+00, 9.600e+02],
                                                 [0.000e+00, 1.545e+03, 5.600e+02],
                                                 [0.000e+00, 0.000e+00, 1.000e+00]]),
                            'sensor2lidar_rotation': array([[0.00802542, 0.01047463, -0.99991293],
                                                            [0.99989075, -0.01249671, 0.00789433],
                                                            [-0.01241293, -0.99986705, -0.01057378]]),
                            'sensor2lidar_translation': array([-0.47463312, 0.02368552, 1.4341838])}}

COLOR_TABLE = {
    'lanelines': np.array([98, 183, 249], np.uint8),  # 浅蓝
    'lanes': np.array([56, 103, 221], np.uint8),  # 深蓝
    'road_boundaries': np.array([200, 36, 35], np.uint8),  # 深红
    'crosswalks': np.array([206, 131, 63], np.uint8),  # 土黄
    'traffic_light_red': np.array([255, 0, 0], np.uint8),  # 红
    'traffic_light_yellow': np.array([255, 255, 0], np.uint8),  # 黄
    'traffic_light_green': np.array([0, 255, 0], np.uint8),  # 绿
    'traffic_light_unknown': np.array([255, 255, 255], np.uint8),  # 白
    'pedestrian': np.array( [255, 0, 255], np.uint8),  # 青
    'vehicle': np.array([0, 128, 255], np.uint8),  # 蓝
    'bicycle': np.array([255, 255, 0], np.uint8),  # 黑
}

# Traffic lights are drawn as upright cuboids of fixed size (they are too
# small and too self-similar to fit an extent to), so the only thing that
# distinguishes one from another on the canvas is the colour of its state.
TRAFFIC_LIGHT_DIMS = (0.5, 0.5, 1.0)   # (L, W, H) in metres
TRAFFIC_LIGHT_BASE_HEIGHT = 5.0        # bottom face above the ground, in metres


def traffic_light_color(state):
    """COLOR_TABLE entry for a traffic-light state, as an RGB list.

    Accepts either a state name ('red'/'yellow'/'green'/'unknown'), as the
    video pipeline reads off the frame pixels, or nuPlan's is_red bool, which
    only distinguishes red from not-red. Anything unrecognised draws white
    rather than guessing a colour the state doesn't support.
    """
    if isinstance(state, (bool, np.bool_)):
        state = 'red' if state else 'green'
    return COLOR_TABLE.get(f'traffic_light_{state}',
                           COLOR_TABLE['traffic_light_unknown']).tolist()


def draw_traffic_light(canvas, position, T_w2c, K, state,
                       dims=TRAFFIC_LIGHT_DIMS):
    """Fills a fixed-size upright cuboid at `position` (its bottom-face centre,
    in the same frame T_w2c maps from) in the colour of its state."""
    draw_cuboid_at(canvas, position, dims, T_w2c, K,
                   color_rgb=traffic_light_color(state), thickness=-1)


def save_as_video(img_list, save_path):
    # 确定视频的保存路径和帧率
    fps = 10  # 可以根据需要调整帧率

    # 获取图像尺寸
    first_frame = img_list[0]
    h, w, c = first_frame['CAM_F0'].shape
    # 拼接后宽度
    total_width = w

    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(save_path, fourcc, fps, (total_width, h))

    for frame_dict in img_list:
        # 获取三张图像
        #img_L = frame_dict['CAM_L0']
        img_F = frame_dict['CAM_F0']
        #img_R = frame_dict['CAM_R0']

        # 确保图像格式是uint8
        #img_L = img_L.astype(np.uint8)
        img_F = img_F.astype(np.uint8)
        #img_R = img_R.astype(np.uint8)

        # 横向拼接
        # concatenated_img = np.hstack((img_L, img_F, img_R))
        # concatenated_img = concatenated_img[:, :, ::-1]  # BGR to RGB
        # 写入视频
       # video_writer.write(concatenated_img)
        video_writer.write(img_F[:, :, ::-1])

    # 释放资源
    video_writer.release()
    print(f'视频已保存到 {save_path}')

def draw_polyline_depth(canvas, polyline3d, T_w2c, K, color,
                        radius=8, seg_interval=0.5,
                        near=1e-3, depth_max=80.):
    H, W = canvas.shape[:2]

    # ---------- 1. 一次性变换 & 投影 ---------- #
    pts_cam = (T_w2c[:3, :3] @ polyline3d.T + T_w2c[:3, 3:4]).T
    z = pts_cam[:, 2]
    cam_mask = z >= near  # 在近平面前方的点
    proj_uv = (K @ pts_cam.T)[:2].T  # shape (N, 2)
    proj_uv /= z[:, None]  # (x/z, y/z)

    u, v = proj_uv[:, 0], proj_uv[:, 1]

    # ---------- 2. per-segment 处理 ---------- #
    for i in range(len(pts_cam) - 1):
        p1c, p2c = pts_cam[i].copy(), pts_cam[i + 1].copy()
        z1, z2 = z[i], z[i + 1]

        # 2-a) z-裁剪到 NEAR
        if z1 < near and z2 < near:
            continue
        if z1 < near or z2 < near:
            t = (near - z1) / (z2 - z1) if z1 < near else (near - z2) / (z1 - z2)
            inter = p1c + t * (p2c - p1c) if z1 < near else p2c + t * (p1c - p2c)
            if z1 < near:
                p1c, z1 = inter, near
            else:
                p2c, z2 = inter, near

            # 只需要为**新增的交点**再算一次投影
            p = (K @ p1c) if z1 == near and (p1c is inter) else (K @ p2c)
            if z1 == near and (p1c is inter):
                proj_uv[i] = p[:2] / p[2]
            else:
                proj_uv[i + 1] = p[:2] / p[2]
            u, v = proj_uv[:, 0], proj_uv[:, 1]  # 更新引用

        # 2-b) 端点像素
        p1 = (int(round(u[i])), int(round(v[i])))
        p2 = (int(round(u[i + 1])), int(round(v[i + 1])))

        # 快速判定“整段在画面内” → 省一次 clipLine
        inside = (
                0 <= p1[0] < W and 0 <= p1[1] < H and
                0 <= p2[0] < W and 0 <= p2[1] < H
        )
        if inside:
            p1_img, p2_img = p1, p2
        else:
            ok, p1_img, p2_img = cv2.clipLine((0, 0, W - 1, H - 1), p1, p2)
            if not ok:
                continue

        # 2-c) 着色
        depth_mean = max(min((z1 + z2) * 0.5, depth_max), 0.)
        alpha = (depth_max - depth_mean) / depth_max
        col = (alpha * color).astype(np.uint8).tolist()

        cv2.line(canvas, p1_img, p2_img, col, radius, cv2.LINE_AA)


def _sutherland_hodgman(poly: np.ndarray, w: int, h: int) -> np.ndarray:
    """Clip a 2‑D polygon against an axis‑aligned screen rectangle using the
    Sutherland–Hodgman algorithm.

    Parameters
    ----------
    poly : (N, 2) array_like
        Polygon vertices (x, y) in image coordinates *in order*.
    w, h : int
        Image width and height.

    Returns
    -------
    np.ndarray, shape (M, 2)
        The clipped polygon (may be empty).
    """

    def clip_edge(pts: list[np.ndarray], inside_fn, intersect_fn):
        if not pts:
            return []
        output = []
        prev = pts[-1]
        prev_inside = inside_fn(prev)
        for curr in pts:
            curr_inside = inside_fn(curr)
            if curr_inside:
                if not prev_inside:  # entering – add intersection first
                    output.append(intersect_fn(prev, curr))
                output.append(curr)
            elif prev_inside:  # leaving – add intersection only
                output.append(intersect_fn(prev, curr))
            prev, prev_inside = curr, curr_inside
        return output

    # Work in float to avoid precision loss
    pts = [np.asarray(p, float) for p in poly.tolist()]

    # Left   (x >= 0)
    pts = clip_edge(
        pts,
        inside_fn=lambda p: p[0] >= 0,
        intersect_fn=lambda p, q: p + (q - p) * ((0 - p[0]) / (q[0] - p[0]))
    )
    if not pts:
        return np.empty((0, 2))

    # Right  (x <= w-1)
    pts = clip_edge(
        pts,
        inside_fn=lambda p: p[0] <= w - 1,
        intersect_fn=lambda p, q: p + (q - p) * ((w - 1 - p[0]) / (q[0] - p[0]))
    )
    if not pts:
        return np.empty((0, 2))

    # Top    (y >= 0)
    pts = clip_edge(
        pts,
        inside_fn=lambda p: p[1] >= 0,
        intersect_fn=lambda p, q: p + (q - p) * ((0 - p[1]) / (q[1] - p[1]))
    )
    if not pts:
        return np.empty((0, 2))

    # Bottom (y <= h-1)
    pts = clip_edge(
        pts,
        inside_fn=lambda p: p[1] <= h - 1,
        intersect_fn=lambda p, q: p + (q - p) * ((h - 1 - p[1]) / (q[1] - p[1]))
    )

    return np.asarray(pts, dtype=np.float32)


def draw_polygon_depth(canvas: np.ndarray,
                       hull3d: np.ndarray,
                       T_w2c: np.ndarray,
                       K: np.ndarray,
                       color: np.ndarray,
                       depth_max) -> None:
    """Project a convex 3‑D polygon and draw its visible part with depth shading.

    Compared with the original implementation, this version **clips** the
    projected polygon against the image boundary so that even if some of the
    polygon’s vertices are outside the frame (or behind the camera), the visible
    portion is still rendered.
    """

    # --- World → camera space ------------------------------------------------
    pts_cam = (T_w2c[:3, :3] @ hull3d.T + T_w2c[:3, 3:4]).T  # (N, 3)

    # Cull vertices that are *behind* the camera (negative z). We ignore them
    # for projection but keep their depth for α if any remain in front.
    in_front = pts_cam[:, 2] > 1e-6
    if not np.any(in_front):
        return  # whole polygon is behind camera

    pts_cam_front = pts_cam[in_front]

    # --- Perspective projection (no validity filtering yet) -----------------
    uv_h = (K @ pts_cam_front.T).T  # (M, 3) – homogeneous
    uv = uv_h[:, :2] / uv_h[:, 2:3]

    # --- Clip against the image rectangle -----------------------------------
    h, w = canvas.shape[:2]
    poly_clipped = _sutherland_hodgman(uv, w, h)
    if poly_clipped.shape[0] < 3:
        return  # Vanishes after clipping

    hull_uv = poly_clipped.astype(np.int32)

    # --- Depth‑based alpha ---------------------------------------------------
    depth_mean = float(np.clip(pts_cam_front[:, 2].mean(), 0.0, depth_max))
    alpha = (depth_max - depth_mean) / depth_max
    col = (alpha * np.asarray(color, dtype=float)).astype(np.uint8).tolist()

    # --- Rasterisation -------------------------------------------------------
    cv2.fillConvexPoly(canvas, hull_uv, col)


# Pixel coordinates are clamped to this before rasterising. Off-screen corners
# are legitimate -- a truncated object genuinely continues past the frame edge --
# but a corner near the camera plane projects to a coordinate large enough to
# overflow int32, which yields a garbage polygon rather than a clipped one.
PIXEL_CLAMP = 100000


def project_points_unclipped(points_cam, K):
    """(N,3) camera-frame points -> (N,2) float pixel coords, off-image included.

    Unlike project_points_cam this keeps points outside the frame instead of
    flagging them invalid, because a cuboid may be mostly off-screen and still
    have visible faces; only the near plane and PIXEL_CLAMP are enforced.
    """
    z = np.maximum(points_cam[:, 2], 1e-3)
    u = K[0, 0] * points_cam[:, 0] / z + K[0, 2]
    v = K[1, 1] * points_cam[:, 1] / z + K[1, 2]
    return np.clip(np.stack([u, v], axis=1), -PIXEL_CLAMP, PIXEL_CLAMP)


def poly_intersects_image(poly, width, height) -> bool:
    """Whether a polygon's bounding box overlaps the canvas at all."""
    return bool(poly[:, 0].max() >= 0 and poly[:, 0].min() < width and
                poly[:, 1].max() >= 0 and poly[:, 1].min() < height)


# A face is clipped against this plane before it is projected. A corner behind
# the camera has no projection at all, and clamping its pixel coordinate
# (project_points_unclipped) turns the face into a polygon whose edges sweep
# across the frame instead of a clipped one; clipping in camera space keeps
# exactly the part the camera can see. Not 0: a vertex left sitting on the
# camera plane still divides by ~0 when projected.
NEAR_PLANE_M = 0.05


def clip_polygon_near(points_cam, near=NEAR_PLANE_M):
    """Sutherland-Hodgman clip of a camera-frame polygon against z >= near.

    Returns the (M, 3) visible part, empty when the polygon lies entirely
    behind the near plane. Convexity is preserved, so the result is still safe
    to hand to cv2.fillConvexPoly -- it just may have five corners instead of
    four, where the plane cuts across a face.
    """
    points = np.asarray(points_cam, dtype=np.float64)
    kept = []
    for i in range(len(points)):
        current, following = points[i], points[(i + 1) % len(points)]
        current_in, following_in = current[2] >= near, following[2] >= near
        if current_in:
            kept.append(current)
        if current_in != following_in:
            t = (near - current[2]) / (following[2] - current[2])
            kept.append(current + t * (following - current))
    return np.array(kept, dtype=np.float64) if kept else np.empty((0, 3))


def draw_cuboids_with_occlusion(canvas, bboxes, T_w2c, K, depth_max=120.0):
    """
    在一张 canvas（H×W×3）上，将所有车辆的 3D 立方体面进行深度排序后填充：
    - 使用低饱和度的“粉彩”式颜色作为每个面的基础色，
    - 并根据面到相机的平均深度做线性颜色衰减（越远越暗）。
    - bboxes: 形状为 (N, >=7) 的数组。每一行至少包含 [x, y, z, L, W, H, yaw, ...]
    - T_w2c: 4×4 世界到相机的变换矩阵
    - K:      3×3 相机内参
    - depth_max: 用于裁剪深度时的最大深度（如果 Z 超过该值，就当作 depth_max 处理）

    Objects are ordered as wholes, and only then are one object's own faces
    ordered among themselves. Sorting every face of every object into one list
    -- which is what this used to do -- lets two objects interleave, and that is
    not a thing solid vehicles can do: a distant car whose box fell inside the
    long axis of a nearer van's box was drawn over the van's far face, so the
    small far box sat on top of the big near one. Face-level sorting is only
    needed for geometry that can interpenetrate; within one convex box,
    back-to-front by mean depth is exact.

    Between objects the order comes from where each box meets the road -- the
    lowest row its silhouette reaches -- and not from its depth. On a common
    ground plane the two say the same thing, and the contact row says it far
    more reliably: it is a projected measurement of a box already anchored to
    its mask, while the depth is the monocular range, which lift_frames_to_3d's
    GROUND_CONTACT_RANGE measures as three times noisier and which is simply
    absent whenever the ground-contact fit was rejected and the point map had to
    stand in.

    That rejection is what makes this more than a tidy-up, because it fires
    precisely on the objects this has to order: a partly occluded one. On the
    ambulance clip's frame 69 the car behind the ambulance shows only a 71x66
    px sliver, whose implied height is 0.42x the car prior and so fails
    GROUND_SIZE_BAND; its range falls back to the point map at 11 m against the
    ambulance's 14 m, and the car in front. Its box still meets the road 59 rows
    higher up the image than the ambulance's does, which is the truth the
    occlusion destroyed and the anchoring kept.

    Ordering by the nearest corner, which this did before, does not help there
    -- it was chosen over centre depth because a 9 m van and a 2 m car can share
    a centre depth while the van's nose is 4 m nearer, and the contact row keeps
    that property for free: the van's nose reaches further down the image than
    the car does.
    """
    H, W = canvas.shape[:2]

    # ---- 1) 低饱和度粉彩底色（BGR 格式） ----
    #    颜色值都在 80~150 之间，保证偏灰，但又带一点色彩
    base_face_colors = [
        (247, 37, 133),  # front 面（微暖粉色）
        (76, 201, 240),  # back  面（微暖绿色）
        (114, 9, 183),  # left  面（微暖蓝色）
        (67, 97, 238),  # right 面（微暖黄色）
        (58, 12, 163),  # top   面（微暖青色）
        (58, 12, 163),  # bottom面（微暖紫色）
    ]

    # ---- 2) 面索引定义，与 vehicle_corners_local 返回的 8 个点顺序保持一致 ----
    face_indices = [
        [0, 1, 5, 4],  # front 面
        [2, 3, 7, 6],  # back  面
        [3, 0, 4, 7],  # left  面
        [1, 2, 6, 5],  # right 面
        [3, 2, 1, 0],  # top   面
        [4, 5, 6, 7],  # bottom面
    ]

    # ---- 3) 收集所有要绘制的“物体”，每个带自己的面 ----
    objects_to_draw = []  # 每项：{'contact_row': float, 'faces': [{'poly', 'depth', 'base_color'}]}

    num_vehicles = bboxes.shape[0]
    for vi in range(num_vehicles):
        info = bboxes[vi]
        pos   = info[box_schema.CENTER]              # (x, y, z)
        L, Wd, H_box = info[box_schema.DIMS]         # 长、宽、高
        roll, pitch, yaw = info[box_schema.RPY]      # 滚转、俯仰、偏航

        # 3.1) 局部角点，(8,3)
        corners_loc = vehicle_corners_local(L, Wd, H_box)

        # 3.2) 世界坐标系下旋转 + 平移
        R_box = rpy_to_rot(roll, pitch, yaw)
        corners_world = (R_box @ corners_loc.T).T + pos

        # 3.3) 转到相机坐标系
        pts_cam = (T_w2c[:3, :3] @ corners_world.T + T_w2c[:3, 3:4]).T  # (8,3)

        # 3.4) 投影到像素平面（保留画面外的角点）
        uv = project_points_unclipped(pts_cam, K)  # (8,2) float

        # Skip only what cannot contribute a pixel: a box entirely behind the
        # camera, or one whose whole silhouette misses the canvas. The old test
        # here required 4 of the 8 corners to land *inside* the image, which
        # dropped every correctly-placed truncated object -- a truck at the
        # frame edge continues off-screen, so most of its corners are outside
        # and it vanished entirely instead of being drawn clipped.
        in_front = pts_cam[:, 2] > 1e-3
        if not in_front.any() or not poly_intersects_image(uv[in_front], W, H):
            continue

        # 3.5) 遍历 6 个面，收集可绘制的面
        faces = []
        for fi, idxs in enumerate(face_indices):
            # Near-plane clip first: a face with a corner behind the camera has
            # no meaningful projection until the invisible part is cut away.
            poly_cam = clip_polygon_near(pts_cam[idxs])
            if len(poly_cam) < 3:
                continue

            # 计算这个面顶点的平均深度，并 clamp 到 [0, depth_max]
            z_vals = poly_cam[:, 2].clip(0, depth_max)
            z_mean = float(np.mean(z_vals))

            # 顶点在图像平面上的整数像素坐标
            poly_2d = project_points_unclipped(poly_cam, K).astype(np.int32)

            # Keep a face whose polygon overlaps the canvas even when none of its
            # own corners is inside it -- a near, wide face can span the frame
            # with all four corners beyond the edges. cv2.fillConvexPoly clips.
            if not poly_intersects_image(poly_2d, W, H):
                continue

            faces.append({
                'poly': poly_2d,
                'depth': z_mean,
                'base_color': base_face_colors[fi]
            })

        if not faces:
            continue
        # Where this box meets the road, as an image row: the lowest point of
        # the silhouette, taken over the corners that are in front of the
        # camera. A box straddling the camera plane has corners that project
        # nowhere meaningful, so it falls back to the bottom of the canvas --
        # it is the nearest thing there is, and sorts that way.
        if in_front.all():
            contact_row = float(uv[:, 1].max())
        else:
            contact_row = float(H)
        objects_to_draw.append({
            'contact_row': contact_row,
            'faces': faces,
        })

    # ---- 4) 物体从远到近排序，物体内部的面同样从远到近 ----
    # Far to near is low row to high row: the road recedes up the image.
    objects_to_draw.sort(key=lambda o: o['contact_row'])
    faces_to_draw = [face
                     for obj in objects_to_draw
                     for face in sorted(obj['faces'], key=lambda f: f['depth'], reverse=True)]

    # ---- 5) 按顺序绘制所有面，并做深度衰减（越远越暗） ----
    for face in faces_to_draw:
        poly       = face['poly']         # (M,2) 的 int32
        depth_mean = face['depth']        # 平均深度
        base_B, base_G, base_R = face['base_color']

        # 线性深度衰减系数 alpha ∈ [0,1]：1 表示最近，0 表示 depth_max
        alpha = np.clip((depth_max - depth_mean) / depth_max, 0.0, 1.0)

        # 应用衰减：直接在 BGR 三个通道上乘以 alpha
        B = int(base_B * alpha)
        G = int(base_G * alpha)
        R = int(base_R * alpha)

        cv2.fillConvexPoly(canvas, poly, (B, G, R), cv2.LINE_AA)


def vehicle_corners_local(L, W, H):
    """返回 (8,3) 车辆局部坐标顶点，Z 轴向上"""
    return np.array([
        [L / 2, W / 2, H / 2],  # 0 前左上
        [L / 2, -W / 2, H / 2],  # 1 前右上
        [-L / 2, -W / 2, H / 2],  # 2 后右上
        [-L / 2, W / 2, H / 2],  # 3 后左上
        [L / 2, W / 2, -H / 2],  # 4 前左下
        [L / 2, -W / 2, -H / 2],  # 5 前右下
        [-L / 2, -W / 2, -H / 2],  # 6 后右下
        [-L / 2, W / 2, -H / 2],  # 7 后左下
    ], dtype=np.float32)


def draw_cuboids_depth(canvas,
                       cuboids_world,       # list[(8,3)]
                       T_w2c,               # 4×4
                       K,                   # 3×3
                       colors_rgb=None,     # list[(r,g,b)]，可 None
                       depth_max=120.0,
                       edge_thickness=2):
    """
    在同一张 canvas 上绘制 N 个车辆 cuboid。
    可见性由面级深度排序确保（近处自动遮挡远处）。
    """
    H, W = canvas.shape[:2]
    if colors_rgb is None:
        colors_rgb = [(200, 0, 0)] * len(cuboids_world)

    # ——— 立方体 6 个面顶点索引 ———
    faces = [
        (0, 1, 2, 3),  # top
        (4, 5, 6, 7),  # bottom
        (0, 1, 5, 4),  # front
        (2, 3, 7, 6),  # back
        (1, 2, 6, 5),  # right
        (0, 3, 7, 4)   # left
    ]

    # ------------------------------------------------
    # 1⃣️ 先把所有 cuboid 的所有面丢进列表并算平均深度
    # ------------------------------------------------
    face_buffer = []   # (z_mean, poly_int32, fill_color_bgr)

    for corners_world, base_col_rgb in zip(cuboids_world, colors_rgb):
        # 世界 → 相机
        pts_cam = (T_w2c[:3, :3] @ corners_world.T + T_w2c[:3, 3:4]).T
        uv, valid = project_points_cam(pts_cam, K, (H, W))

        # 任一顶点可见即可尝试该面；如果所有顶点都看不见就跳过
        for idx in faces:
            if not valid[list(idx)].any():
                continue

            pts_cam_face = pts_cam[list(idx)]
            z_mean = float(np.clip(pts_cam_face[:, 2].mean(), 0, depth_max))

            # ——— 颜色：基础色 * 深度衰减 * 朝向阴影 ———
            depth_alpha = (depth_max - z_mean) / depth_max          # 近→1 远→0
            # 用简单 Lambert 估计：normal z 分量越负越朝向相机
            n = np.cross(pts_cam_face[1] - pts_cam_face[0],
                         pts_cam_face[2] - pts_cam_face[0])
            n = n / (np.linalg.norm(n) + 1e-6)
            facing = max(-n[2], 0.0)                                # 0~1
            shade = 0.3 + 0.7 * facing                              # 侧面更暗
            shade *= 0.6 + 0.4 * depth_alpha                        # 远处整体更暗
            col_face = tuple(int(shade * c) for c in base_col_rgb)  # RGB

            poly = uv[list(idx)].astype(np.int32)                   # (4,2)
            face_buffer.append((z_mean, poly, col_face[::-1]))      # BGR

    # ------------------------------------------------
    # 2⃣️ 远 → 近 排序并填充
    # ------------------------------------------------
    face_buffer.sort(key=lambda x: x[0], reverse=True)
    for _, poly, col_bgr in face_buffer:
        cv2.fillConvexPoly(canvas, poly, col_bgr, cv2.LINE_AA)

    # ------------------------------------------------
    # 3⃣️ 最后勾勒所有棱线（可选）
    # ------------------------------------------------
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]
    for corners_world, base_col_rgb in zip(cuboids_world, colors_rgb):
        pts_cam = (T_w2c[:3, :3] @ corners_world.T + T_w2c[:3, 3:4]).T
        uv, valid = project_points_cam(pts_cam, K, (H, W))

        for i, j in edges:
            if not (valid[i] or valid[j]):
                continue
            z_mean = np.clip((pts_cam[i, 2] + pts_cam[j, 2]) * 0.5, 0, depth_max)
            depth_alpha = (depth_max - z_mean) / depth_max
            col_edge = tuple(int(depth_alpha * c) for c in base_col_rgb[::-1])  # BGR
            p1, p2 = tuple(uv[i]), tuple(uv[j])
            ok, p1c, p2c = cv2.clipLine((0, 0, W, H), p1, p2)
            if ok:
                cv2.line(canvas, p1c, p2c, col_edge, edge_thickness, cv2.LINE_AA)

import numpy as np
import cv2

def draw_cuboid_at(canvas,
                   center_pos,   # [x, y, z], 世界坐标系中长方体底面中心位置
                   dims,         # (L, W, H)
                   T_w2c,        # 4×4 世界→相机
                   K,            # 3×3 相机内参
                   color_rgb=(0, 255, 0),
                   thickness=-1  # -1 表示填充，>0 表示画线框
                   ):
    """
    在 canvas 上，以 center_pos 作为长方体底面中心，把一个 (L, W, H) 的长方体投影并绘制到图像上。
    - canvas: uint8 图像，H×W×3
    - center_pos: 长方体底面中心在世界坐标系中的 [x, y, z]
    - dims: (L, W, H)
    - T_w2c: 4×4 世界到相机坐标变换矩阵
    - K: 3×3 相机内参
    - color_rgb: (R, G, B)
    - thickness: -1 填充，>0 画边线
    """

    H_img, W_img = canvas.shape[:2]

    # 1) 先在“Local”坐标系里得到 8 个顶点。让底面中心位于 (0,0,0)：
    L, W, H_box = dims
    # 本地坐标系下的 8 个顶点（x, y, z）
    # 底面 z=0，顶面 z=H_box
    # 这里：x 轴指向正前方，y 轴指向右方，z 轴指向上方。
    # 下面顺序便于组合各面：
    local_corners = np.array([
        [ L/2,  W/2, 0.0],  # 0: front-right-bottom
        [ L/2, -W/2, 0.0],  # 1: front-left -bottom
        [-L/2, -W/2, 0.0],  # 2: back -left -bottom
        [-L/2,  W/2, 0.0],  # 3: back -right-bottom
        [ L/2,  W/2, H_box],# 4: front-right-top
        [ L/2, -W/2, H_box],# 5: front-left -top
        [-L/2, -W/2, H_box],# 6: back -left -top
        [-L/2,  W/2, H_box] # 7: back -right-top
    ], dtype=np.float32)  # (8,3)

    # 2) 从 Local → World：直接平移到 center_pos。因为交通信号灯一般竖直不旋转，这里不考虑 yaw。
    #    如果你要让它围绕 z 轴有朝向（比如竖直柱子有朝北朝南的方向），再插入旋转矩阵就行。
    center_pos = np.array(center_pos, dtype=np.float32).reshape(3,)
    world_corners = local_corners + center_pos  # (8,3)

    # 3) World → Camera 坐标系：
    #    pts_cam = R * world + t  （R = T_w2c[:3,:3], t = T_w2c[:3,3])
    pts_cam = (T_w2c[:3, :3] @ world_corners.T + T_w2c[:3, 3:4]).T  # (8,3)

    # 4) 投影到像素面，得到 uv=(8,2) 和 valid=(8,) boolean 掩码
    uv, valid = project_points_cam(pts_cam, K, (H_img, W_img))

    # 5) 定义 6 个面用到的顶点索引（4 个点一组），按照 local_corners 的顺序
    face_idxs = [
        [0, 1, 2, 3],  # 底面 （z=0）
        [4, 5, 6, 7],  # 顶面 （z=H_box）
        [0, 1, 5, 4],  # 前面（front）
        [1, 2, 6, 5],  # 左面（left）
        [2, 3, 7, 6],  # 后面（back）
        [3, 0, 4, 7],  # 右面（right）
    ]

    # 6) 遍历每个面，如果该面的4个顶点中至少有1个 “valid”（在图像内、Z>0），就画出该面
    for idxs in face_idxs:
        # 先检查是否有任意一个顶点在相机前方且投影落在图像范围内
        if not np.any([valid[i] for i in idxs]):
            continue

        # 再检查该面所有点是否都在相机后方: 如果都在 Z<=0，就跳过
        pts_face_cam = pts_cam[idxs]  # (4,3)
        if np.all(pts_face_cam[:, 2] <= 0):
            continue

        # 取出 2D 投影坐标（整数）
        poly2d = np.array([uv[i] for i in idxs], dtype=np.int32)  # (4,2)

        # 用 OpenCV 填充或画线
        if thickness < 0:
            cv2.fillConvexPoly(canvas, poly2d, color_rgb, cv2.LINE_AA)
        else:
            cv2.polylines(canvas, [poly2d], isClosed=True, color=color_rgb, thickness=thickness, lineType=cv2.LINE_AA)

def draw_heading_arrow(canvas,
                       pos_world,  # (3,)  物体在世界坐标中的质心
                       yaw,  # 标量 (弧度)
                       T_w2c,  # (4,4) 世界→相机
                       K,  # (3,3) 内参
                       color_rgb=(255, 255, 0),
                       arrow_len=3.0,  # 以米为单位，在图上可调
                       thickness=6):
    H, W = canvas.shape[:2]
    color_bgr = tuple(int(c) for c in color_rgb)

    # ---- 1. 计算箭头两端的世界坐标 ------------------------------------------
    # “车头”方向向量（世界系）
    dir_world = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    p_tail_w = pos_world
    p_head_w = pos_world + dir_world * arrow_len

    # ---- 2. 世界 → 相机 -------------------------------------------------------
    pw_tail_c = T_w2c[:3, :3] @ p_tail_w + T_w2c[:3, 3]
    pw_head_c = T_w2c[:3, :3] @ p_head_w + T_w2c[:3, 3]

    # 过滤：若尾点就在相机后面（z<=0），直接跳过
    if pw_tail_c[2] <= 0 or pw_head_c[2] <= 0:
        return

    # ---- 3. 投影到像素坐标 ----------------------------------------------------
    pts_cam = np.vstack([pw_tail_c, pw_head_c])  # shape (2,3)
    uv, valid = project_points_cam(pts_cam, K, (H, W))  # uv: (2,2)

    if not valid.all():
        return

    p_tail_px, p_head_px = map(tuple, uv.astype(int))

    # ---- 4. 画箭头 ------------------------------------------------------------
    cv2.arrowedLine(canvas,
                    p_tail_px,
                    p_head_px,
                    color_bgr,
                    thickness,
                    tipLength=0.25)  # tipLength 相对箭头长度的比例


class ScenarioRenderer:
    def __init__(self, camera_channel_list=['CAM_F0', 'CAM_L0', 'CAM_R0'], width=1920, height=1120, depth_max=120.0):
        self.width = width
        self.height = height
        self.depth_max = depth_max
        self.camera_models = {}
        for k, v in camera_params.items():
            if not k in camera_channel_list: continue
            self.camera_models[k] = v

    def observe(self, scenario):
        lidar_pos = np.zeros(3)
        lidar_yaw = scenario['ego_heading']
        ret_dict = {}
        for cam_id, cam_model in self.camera_models.items():
            canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            cam_t = cam_model["sensor2lidar_translation"].copy()  # (3,)
            cam_t[2] += 0.8
            cam_t[0] -= 2
            cam_R = cam_model["sensor2lidar_rotation"]  # (3,3)
            K = cam_model["intrinsics"]  # (3,3)
            T_w2c = world_to_camera_T(lidar_pos, lidar_yaw, cam_t, cam_R)  # 4×4

            for feat in scenario['traffic_lights']:
                is_red = feat[1]
                xy     = feat[2]     # [x, y]，注意还缺 z
                # 地图只给出信号灯所属车道的位置，没有高度，所以底座高度取固定值
                pos_world = [xy[0], xy[1], TRAFFIC_LIGHT_BASE_HEIGHT]

                draw_traffic_light(canvas, pos_world, T_w2c, K, is_red)

            for feat in scenario['map_features'].values():
                ftype = feat['type']
                if 'LANE' in ftype:
                    poly2d = feat['polygon'].astype(np.float32)
                    pts3d = np.hstack([poly2d, np.zeros((poly2d.shape[0], 1), np.float32)])
                    pts_dist = np.linalg.norm(poly2d - lidar_pos[np.newaxis, :2], axis=1)
                    if np.min(pts_dist) > self.depth_max:
                        continue
                    draw_polyline_depth(canvas, pts3d, T_w2c, K, COLOR_TABLE['lanelines'], radius=2,depth_max=self.depth_max)

                elif 'CROSSWALK' in ftype or 'SPEED_BUMP' in ftype:
                    poly2d = feat['polygon'].astype(np.float32)
                    pts3d = np.hstack([poly2d, np.zeros((poly2d.shape[0], 1), np.float32)])
                    draw_polygon_depth(canvas, pts3d, T_w2c, K, COLOR_TABLE['crosswalks'],self.depth_max)
                    draw_polyline_depth(canvas, pts3d, T_w2c, K, COLOR_TABLE['lanelines'],depth_max=self.depth_max)

                elif 'BOUNDARY' in ftype or 'SOLID' in ftype:
                    poly2d = feat['polyline'].astype(np.float32)
                    pts3d = np.hstack([poly2d, np.zeros((poly2d.shape[0], 1), np.float32)])
                    draw_polyline_depth(canvas, pts3d, T_w2c, K,
                                        COLOR_TABLE['road_boundaries'], radius=10,depth_max=self.depth_max)
            anns = scenario["anns"]
            bboxes = anns["gt_boxes_world"]
            names = anns["gt_names"]
            draw_cuboids_with_occlusion(canvas, bboxes, T_w2c, K)
            
            ret_dict[cam_id] = canvas            
            # import matplotlib.pyplot as plt
            # plt.imshow(canvas)
            # plt.show()
            #print("canvas shape", canvas.shape)
            # ret_dict[cam_id] = canvas
            # cuboids = []
            # for i in range(bboxes.shape[0]):
            #     info = bboxes[i]
            #     name = names[i]
            #     pos = info[:3]
            #     yaw = info[6]
            #     L = info[3]
            #     Wd = info[4]
            #     H_box = info[5]

            #     color = COLOR_TABLE['vehicle']

            #     corners_loc = vehicle_corners_local(L, Wd, H_box)
            #     R_yaw = yaw_to_rot(yaw)
            #     corners_world = (R_yaw @ corners_loc.T).T + pos
            #     cuboids.append(corners_world)

            # draw_scene_cuboids(canvas,
            #                     cuboids,
            #                     T_w2c, K, color)

            #     # if track["type"] == "BICYCLE":
            #     #     color = COLOR_TABLE['bicycle']
            #     # elif track["type"] == "PEDESTRIAN":
            #     #     color = COLOR_TABLE['pedestrian']
            #     # elif track["type"] == "VEHICLE":
            #     #     color = COLOR_TABLE['vehicle']
            #     #     Wd, H_box = H_box, Wd
            #     # else:

                



        return ret_dict



def make_sky_ground_canvas(
    H, W,
    horizon=0.60,
    sky_top=(180,120,60),
    sky_horizon=(230,205,185),
    ground_far=(105,105,105),
    ground_near=(35,35,35),
    sun=None,                      # e.g. {'azim':0.65,'elev':0.22,'radius':70,'glow_sigma':150,'intensity':0.9}
    vignette_strength=0.22,
    noise_std=2,
    seed=None
):
    if seed is not None:
        np.random.seed(seed)

    # ---------- 1) 按行生成天空/地面（明确通道维） ----------
    y = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]   # (H,1,1)  0顶/1底
    sky_mask = (y < horizon).astype(np.float32)                  # (H,1,1)
    t_sky = np.clip(y / max(horizon, 1e-6), 0, 1)                # (H,1,1)
    t_gnd = np.clip((y - horizon) / max(1 - horizon, 1e-6), 0, 1)

    sky_top  = np.array(sky_top,  np.float32)[None, None, :]     # (1,1,3)
    sky_hori = np.array(sky_horizon, np.float32)[None, None, :]
    gnd_far  = np.array(ground_far,  np.float32)[None, None, :]
    gnd_near = np.array(ground_near, np.float32)[None, None, :]

    sky = (1 - t_sky) * sky_top + t_sky * sky_hori               # (H,1,3)
    gnd = (1 - t_gnd) * gnd_far + t_gnd * gnd_near               # (H,1,3)
    row_bg = sky_mask * sky + (1 - sky_mask) * gnd               # (H,1,3)

    # 扩展到整幅宽度，显式 (H,W,3)
    bg = np.broadcast_to(row_bg, (H, W, 3)).astype(np.float32).copy()

    # ---------- 2) 太阳（带柔光），用显式通道维 ----------
    if isinstance(sun, dict):
        az  = float(sun.get('azim', 0.5))       # 0左 1右
        el  = float(sun.get('elev', 0.18))      # 0顶 1底
        r   = int(sun.get('radius', 80))
        sgm = float(sun.get('glow_sigma', 140))
        inten = float(sun.get('intensity', 0.9))

        x0, y0 = int(W * az), int(H * el)
        yy, xx = np.ogrid[:H, :W]              # 省内存
        d2 = (xx - x0)**2 + (yy - y0)**2

        disk = (d2 <= r*r).astype(np.float32)[..., None]         # (H,W,1)
        glow = np.exp(-0.5 * d2 / (sgm*sgm)).astype(np.float32)[..., None]

        sun_col = np.array([255, 255, 255], np.float32)[None, None, :]  # (1,1,3)

        bg += disk * (sun_col - bg) * (0.8 * inten)
        bg += glow * (sun_col - bg) * (0.25 * inten)

    # ---------- 3) 暗角 ----------
    if vignette_strength > 0:
        nx = np.linspace(-1, 1, W, dtype=np.float32)
        ny = np.linspace(-1, 1, H, dtype=np.float32)
        xx, yy = np.meshgrid(nx, ny)
        rr = np.sqrt(xx*xx + yy*yy)
        vign = np.clip(1 - vignette_strength * rr*rr, 0.75, 1.0).astype(np.float32)
        bg *= vign[..., None]

    # ---------- 4) 轻噪声 ----------
    if noise_std > 0:
        noise = np.random.normal(0, noise_std, (H, W, 3)).astype(np.float32)
        bg += noise

    return np.clip(bg, 0, 255).astype(np.uint8)