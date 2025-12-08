#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Waymo Semantic Geometric Builder (for CASA-Calib Dataset Construction)
Author: Yuan-Ting Fu (NTUT)

----------------------------------------------------------------------------
Purpose (Corresponds to Contribution 3 of the CASA-Calib Paper)
----------------------------------------------------------------------------
This script constructs a curated semantic–geometric test set from the Waymo
Open Dataset. Its goal is to provide reliable, instance-level LiDAR–camera
correspondences for accurate evaluation and benchmarking of semantic-assisted
extrinsic calibration methods such as CASA-Calib.

Waymo’s original dataset contains high-quality but independently generated
LiDAR and camera segmentation labels. This extractor performs:

    • Consistent extraction of aligned semantic information from BOTH modalities
    • Conversion of LiDAR range-image segmentation into point-level labels
    • Saving camera RGB, panoptic, semantic, and instance masks
    • Saving calibration matrices, LiDAR XYZ point clouds, and segmentation
    • Filtering only valid frames containing complete segmentation annotations

The resulting dataset enables:
    • Instance-level geometric consistency checking
    • Manual / automatic semantic verification
    • Cross-sensor semantic reasoning
    • Robust evaluation of semantic-assisted calibration algorithms

This extractor is a key component in the CASA-Calib pipeline for constructing
clean, reliable benchmark data for LiDAR–camera extrinsic calibration research.

----------------------------------------------------------------------------
"""

import os
os.environ["MPLBACKEND"] = "Qt5Agg"  # Force Qt5 backend for Matplotlib

import matplotlib
matplotlib.use("Qt5Agg")  # Interactive Qt5 backend for GUI

import numpy as np
import tensorflow.compat.v1 as tf
from matplotlib.font_manager import FontProperties
from matplotlib.colors import ListedColormap
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import cv2
import csv
from pathlib import Path
import shutil
from collections import Counter

from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import (
    camera_segmentation_utils,
    transform_utils,
    range_image_utils,
)
from waymo_open_dataset.protos import (
    segmentation_metrics_pb2,
    segmentation_submission_pb2,
)
from matplotlib.widgets import Button

tf.enable_eager_execution()

# -----------------------------------------------------------------------------
# Global state for CSV initialization
# -----------------------------------------------------------------------------
CSV_INITED = set()  # Tracks which sequence directories have initialized CSV


# -----------------------------------------------------------------------------
# User configuration
# -----------------------------------------------------------------------------
DATASET_FOLDER = "/mnt/e/tar_and_tfrecord_file/06"  # Folder containing .tfrecord files

# Build a 60-color colormap from 3 tab20 variants
colors = (
    list(plt.get_cmap("tab20").colors)
    + list(plt.get_cmap("tab20b").colors)
    + list(plt.get_cmap("tab20c").colors)
)
tab60 = ListedColormap(colors[:60])

# Set font (for environments that need specific font, e.g. Windows MingLiU)
#font_path = "/mnt/c/Windows/Fonts/msjh.ttc"
#font_prop = FontProperties(fname=font_path)
#plt.rcParams["font.family"] = font_prop.get_name()
#plt.rcParams["axes.unicode_minus"] = False  # Avoid minus sign rendering issues


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def make_dirs(path: str) -> None:
    """Create directory if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path)


def ensure_summary_csv_initialized(seq_dir: Path) -> Path:
    """
    Ensure that 'pair_summary.csv' exists in the sequence directory, is cleared,
    and has a header row. This initialization is done at most once per sequence.

    Parameters
    ----------
    seq_dir : Path
        The path to the sequence directory.

    Returns
    -------
    Path
        The path to 'pair_summary.csv' in the sequence directory.
    """
    csv_path = seq_dir / "pair_summary.csv"
    if seq_dir not in CSV_INITED:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "seq",
                    "tfrecord_name",
                    "frame_id",
                    "pixel_file_name",
                    "lidar_file_name",
                ]
            )
        CSV_INITED.add(seq_dir)
    return csv_path


def cart_to_homo(mat: np.ndarray) -> np.ndarray:
    """
    Convert a 3x3 or 3x4 matrix to a 4x4 homogeneous transform.

    Parameters
    ----------
    mat : np.ndarray
        Input matrix of shape (3, 3) or (3, 4).

    Returns
    -------
    np.ndarray
        4x4 homogeneous matrix.
    """
    ret = np.eye(4)
    if mat.shape == (3, 3):
        ret[:3, :3] = mat
    elif mat.shape == (3, 4):
        ret[:3, :] = mat
    else:
        raise ValueError(f"Unsupported shape in cart_to_homo: {mat.shape}")
    return ret


def find_tfrecord_files_in_dir(folder_path: str) -> list:
    """Return a sorted list of all .tfrecord files in a directory."""
    return sorted(
        [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.endswith(".tfrecord")
        ]
    )


def extract_target_frame_indices(filename: str) -> list:
    """
    Scan a TFRecord file and collect indices of frames that have both
    image panoptic labels and LiDAR segmentation labels.

    Parameters
    ----------
    filename : str
        Path to the .tfrecord file.

    Returns
    -------
    list
        List of frame indices that can be used for semantic–geometric extraction.
    """
    try:
        dataset = tf.data.TFRecordDataset(filename, compression_type="")
        indices = []
        for ind, data in enumerate(dataset):
            try:
                frame = open_dataset.Frame()
                frame.ParseFromString(bytearray(data.numpy()))
                has_front_panoptic = (
                    frame.images[0].camera_segmentation_label.panoptic_label
                )
                has_lidar_seg = frame.lasers[0].ri_return1.segmentation_label_compressed
                if has_front_panoptic and has_lidar_seg:
                    indices.append(ind)
            except tf.errors.DataLossError as e:
                print(f"⚠️ Frame {ind} is corrupted; skipping: {e}")
                continue
        return indices
    except tf.errors.DataLossError as e:
        print(f"❌ Failed to read {filename}; file appears corrupted, skipping.")
        print(f"   Error: {e}")
        return []


def parse_range_image_and_camera_projection(frame: open_dataset.Frame):
    """
    Parse range images, camera projections, and segmentation labels from a frame.

    Returns
    -------
    range_images : dict
    camera_projections : dict
    segmentation_labels : dict
    range_image_top_pose : open_dataset.MatrixFloat or None
    """
    range_images, camera_projections, segmentation_labels = {}, {}, {}
    range_image_top_pose = None

    for laser in frame.lasers:
        # First return
        if len(laser.ri_return1.range_image_compressed) > 0:
            ri1 = open_dataset.MatrixFloat()
            ri1.ParseFromString(
                tf.decode_compressed(
                    laser.ri_return1.range_image_compressed, "ZLIB"
                ).numpy()
            )
            range_images[laser.name] = [ri1]

            if laser.name == open_dataset.LaserName.TOP:
                pose = open_dataset.MatrixFloat()
                pose.ParseFromString(
                    tf.decode_compressed(
                        laser.ri_return1.range_image_pose_compressed, "ZLIB"
                    ).numpy()
                )
                range_image_top_pose = pose

            cp1 = open_dataset.MatrixInt32()
            cp1.ParseFromString(
                tf.decode_compressed(
                    laser.ri_return1.camera_projection_compressed, "ZLIB"
                ).numpy()
            )
            camera_projections[laser.name] = [cp1]

            if laser.ri_return1.segmentation_label_compressed:
                sl1 = open_dataset.MatrixInt32()
                sl1.ParseFromString(
                    tf.decode_compressed(
                        laser.ri_return1.segmentation_label_compressed, "ZLIB"
                    ).numpy()
                )
                segmentation_labels[laser.name] = [sl1]

        # Second return
        if len(laser.ri_return2.range_image_compressed) > 0:
            ri2 = open_dataset.MatrixFloat()
            ri2.ParseFromString(
                tf.decode_compressed(
                    laser.ri_return2.range_image_compressed, "ZLIB"
                ).numpy()
            )
            range_images[laser.name].append(ri2)

            cp2 = open_dataset.MatrixInt32()
            cp2.ParseFromString(
                tf.decode_compressed(
                    laser.ri_return2.camera_projection_compressed, "ZLIB"
                ).numpy()
            )
            camera_projections[laser.name].append(cp2)

            if laser.ri_return2.segmentation_label_compressed:
                sl2 = open_dataset.MatrixInt32()
                sl2.ParseFromString(
                    tf.decode_compressed(
                        laser.ri_return2.segmentation_label_compressed, "ZLIB"
                    ).numpy()
                )
                segmentation_labels[laser.name].append(sl2)

    return range_images, camera_projections, segmentation_labels, range_image_top_pose


def save_calib(frame: open_dataset.Frame, save_dir: str) -> None:
    """
    Save calibration in KITTI-style format.

    - P0..P3: 3x4 projection matrices (P2 is the real camera intrinsic)
    - R0_rect: identity (no rectification)
    - Tr_velo_to_cam: LiDAR-to-camera extrinsic

    Parameters
    ----------
    frame : open_dataset.Frame
    save_dir : str
    """
    # Use FRONT camera as reference
    for camera in frame.context.camera_calibrations:
        if camera.name == open_dataset.CameraName.FRONT:
            T_front_cam_to_vehicle = np.array(camera.extrinsic.transform).reshape(
                4, 4
            )
            T_vehicle_to_front_cam = np.linalg.inv(T_front_cam_to_vehicle)

            intrinsic = np.zeros((3, 4))
            intrinsic[0, 0], intrinsic[1, 1] = camera.intrinsic[0], camera.intrinsic[1]
            intrinsic[0, 2] = camera.intrinsic[2]
            intrinsic[1, 2] = camera.intrinsic[3]
            intrinsic[2, 2] = 1.0
            break
    else:
        raise RuntimeError("FRONT camera calibration not found.")

    # LiDAR → camera transform (Waymo to KITTI-like convention)
    T_ref = np.array(
        [[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]]
    )  # Waymo axes to KITTI-style
    Tr_velo_to_cam = cart_to_homo(T_ref) @ np.linalg.inv(T_front_cam_to_vehicle)

    calib_text = ""
    for i in range(4):
        if i == 2:
            P = intrinsic.reshape(12)
        else:
            P = np.eye(4)[:3, :].reshape(12)
        calib_text += f"P{i}: {' '.join(map(str, P))}\n"

    calib_text += f"R0_rect: {' '.join(map(str, np.eye(3).flatten()))}\n"
    calib_text += (
        f"Tr_velo_to_cam: "
        f"{' '.join(map(str, Tr_velo_to_cam[:3, :].reshape(12)))}\n"
    )

    with open(os.path.join(save_dir, "calib.txt"), "w+") as f:
        f.write(calib_text)


def convert_range_image_to_point_cloud_labels(
    frame: open_dataset.Frame,
    range_images: dict,
    segmentation_labels: dict,
    ri_index: int = 0,
) -> list:
    """
    Convert LiDAR range image segmentation labels into per-point labels.

    Returns
    -------
    list of np.ndarray
        Each element is an (N, 2) array of [instance_id, semantic_id].
    """
    calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
    point_labels = []

    for c in calibrations:
        ri_tensor = tf.reshape(
            tf.convert_to_tensor(range_images[c.name][ri_index].data),
            range_images[c.name][ri_index].shape.dims,
        )
        mask = ri_tensor[..., 0] > 0  # Valid range

        if c.name in segmentation_labels:
            sl_tensor = tf.reshape(
                tf.convert_to_tensor(segmentation_labels[c.name][ri_index].data),
                segmentation_labels[c.name][ri_index].shape.dims,
            )
            sl_points = tf.gather_nd(sl_tensor, tf.where(mask))
        else:
            # No segmentation label for this laser → zeros
            sl_points = tf.zeros(
                [tf.reduce_sum(tf.cast(mask, tf.int32)), 2], dtype=tf.int32
            )

        point_labels.append(sl_points.numpy())

    return point_labels


def convert_range_image_to_point_cloud(
    frame: open_dataset.Frame,
    range_images: dict,
    camera_projections: dict,
    range_image_top_pose,
    ri_index: int = 0,
):
    """
    Convert range images to 3D LiDAR point clouds and related attributes.

    Returns
    -------
    points : list of np.ndarray
        List of (N_i, 3) arrays of XYZ points in vehicle frame.
    cp_points : list of np.ndarray
        List of camera projection index tensors (for each point).
    intensity : list of np.ndarray
        List of intensity values for each point.
    """
    filter_no_label_zone_points = True
    calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)

    points, cp_points, intensity = [], [], []

    frame_pose = tf.convert_to_tensor(
        np.reshape(np.array(frame.pose.transform), [4, 4])
    )

    # Top LiDAR pose (pixel-wise)
    top_pose_tensor = tf.reshape(
        tf.convert_to_tensor(range_image_top_pose.data),
        range_image_top_pose.shape.dims,
    )
    top_pose_rot = transform_utils.get_rotation_matrix(
        top_pose_tensor[..., 0],
        top_pose_tensor[..., 1],
        top_pose_tensor[..., 2],
    )
    top_pose_trans = top_pose_tensor[..., 3:]
    top_pose = transform_utils.get_transform(top_pose_rot, top_pose_trans)

    for c in calibrations:
        range_image = range_images[c.name][ri_index]

        # Beam inclinations
        if not c.beam_inclinations:
            inclinations = range_image_utils.compute_inclination(
                tf.constant([c.beam_inclination_min, c.beam_inclination_max]),
                height=range_image.shape.dims[0],
            )
        else:
            inclinations = tf.constant(c.beam_inclinations)

        inclinations = tf.reverse(inclinations, [-1])

        extrinsic = np.reshape(np.array(c.extrinsic.transform), [4, 4])
        ri_tensor = tf.reshape(
            tf.convert_to_tensor(range_image.data), range_image.shape.dims
        )

        # Valid range mask
        mask = ri_tensor[..., 0] > 0
        if filter_no_label_zone_points:
            # Filter out "no-label" zone (label=1.0 in Waymo range image semantics)
            mask &= ri_tensor[..., 3] != 1.0

        # Pixel-wise pose only for TOP
        pixel_pose = (
            tf.expand_dims(top_pose, axis=0)
            if c.name == open_dataset.LaserName.TOP
            else None
        )
        frame_pose_local = (
            tf.expand_dims(frame_pose, axis=0)
            if c.name == open_dataset.LaserName.TOP
            else None
        )

        ri_cart = range_image_utils.extract_point_cloud_from_range_image(
            tf.expand_dims(ri_tensor[..., 0], 0),
            tf.expand_dims(extrinsic, 0),
            tf.expand_dims(inclinations, 0),
            pixel_pose=pixel_pose,
            frame_pose=frame_pose_local,
        )

        points_tensor = tf.gather_nd(tf.squeeze(ri_cart, axis=0), tf.where(mask))

        cp_tensor = tf.reshape(
            tf.convert_to_tensor(camera_projections[c.name][ri_index].data),
            camera_projections[c.name][ri_index].shape.dims,
        )
        cp_points_tensor = tf.gather_nd(cp_tensor, tf.where(mask))

        intensity_tensor = tf.gather_nd(ri_tensor, tf.where(mask))

        points.append(points_tensor.numpy())
        cp_points.append(cp_points_tensor.numpy())
        intensity.append(intensity_tensor.numpy()[:, 1])

    return points, cp_points, intensity


def save_lidar_and_labels(frame: open_dataset.Frame, save_dir: str) -> None:
    """
    Save LiDAR 3D points and point-level labels for a given frame.

    Outputs in save_dir:
        - points_all.txt         : all concatenated XYZ points (text)
        - point_labels_all.txt   : all concatenated [instance, semantic] labels
        - lidar.bin              : XYZ points (float32, binary)
    """
    ri, cp, labels, pose = parse_range_image_and_camera_projection(frame)
    pts, _, _ = convert_range_image_to_point_cloud(frame, ri, cp, pose)
    labels = convert_range_image_to_point_cloud_labels(frame, ri, labels)

    all_pts = np.concatenate(pts, axis=0)
    all_labels = np.concatenate(labels, axis=0)

    np.savetxt(os.path.join(save_dir, "points_all.txt"), all_pts)
    np.savetxt(os.path.join(save_dir, "point_labels_all.txt"), all_labels)

    all_pts.astype(np.float32).tofile(os.path.join(save_dir, "lidar.bin"))


def save_images_and_segmentation(frame: open_dataset.Frame, save_dir: str) -> None:
    """
    Save camera RGB and segmentation-related images for the FRONT camera.

    Outputs in save_dir:
        - img_raw.png                 : raw FRONT camera RGB image
        - panoptic_label_front.png    : panoptic label (uint16)
        - instance_label_front.png    : instance label (uint16)
        - instance_waymo.png          : instance label (uint8 visualization)
    """
    from waymo_open_dataset.dataset_pb2 import CameraName

    front_rgb = None
    front_panoptic = None
    front_label_divisor = None

    # Find FRONT camera image and panoptic label
    for img in frame.images:
        if img.name == CameraName.FRONT:
            front_rgb = cv2.imdecode(
                np.frombuffer(img.image, np.uint8), cv2.IMREAD_COLOR
            )
            if img.camera_segmentation_label:
                front_panoptic = img.camera_segmentation_label
                front_label_divisor = img.camera_segmentation_label.panoptic_label_divisor
            break

    if front_rgb is None:
        print("❌ No FRONT camera image found; skipping this frame.")
        return

    if front_panoptic is None or front_panoptic.panoptic_label == b"":
        print("⚠️ No FRONT panoptic label found; skipping this frame.")
        return

    # Save RGB image
    cv2.imwrite(os.path.join(save_dir, "img_raw.png"), front_rgb)

    # Decode panoptic label into semantic + instance
    panoptic = camera_segmentation_utils.decode_single_panoptic_label_from_proto(
        front_panoptic
    )
    sem, inst = (
        camera_segmentation_utils.decode_semantic_and_instance_labels_from_panoptic_label(
            panoptic, front_label_divisor
        )
    )

    cv2.imwrite(
        os.path.join(save_dir, "panoptic_label_front.png"),
        panoptic.astype(np.uint16),
    )
    cv2.imwrite(
        os.path.join(save_dir, "instance_label_front.png"), inst.astype(np.uint16)
    )
    cv2.imwrite(os.path.join(save_dir, "instance_waymo.png"), inst)


def project_lidar_to_image(
    points: np.ndarray, calib_path: str, image_shape=None
) -> np.ndarray:
    """
    Project LiDAR points to image plane using KITTI-like calibration file.

    Parameters
    ----------
    points : np.ndarray
        (N, 3+) XYZ points.
    calib_path : str
        Path to 'calib.txt'.
    image_shape : tuple or None
        Optional (H, W, C). If provided, points outside image bounds are filtered.

    Returns
    -------
    np.ndarray
        (M, 2) array of [u, v] projected points inside image bounds.
    """
    with open(calib_path, "r") as f:
        calib = f.read().splitlines()

    P2 = np.array([float(x) for x in calib[2].split()[1:]]).reshape(3, 4)
    R0_rect = np.array([float(x) for x in calib[4].split()[1:]]).reshape(3, 3)
    Tr_velo_to_cam = np.array(
        [float(x) for x in calib[5].split()[1:]]
    ).reshape(3, 4)

    R0_rect_homo = np.eye(4)
    R0_rect_homo[:3, :3] = R0_rect
    Tr_velo_to_cam_homo = np.eye(4)
    Tr_velo_to_cam_homo[:3, :] = Tr_velo_to_cam

    proj_mat = P2 @ R0_rect_homo @ Tr_velo_to_cam_homo

    pts_hom = np.concatenate([points[:, :3], np.ones((points.shape[0], 1))], axis=1).T
    uvw = proj_mat @ pts_hom
    uv = (uvw[:2] / uvw[2:]).T
    valid = uvw[2, :] > 0

    if image_shape is not None:
        H, W = image_shape[:2]
        in_img = (
            (uv[:, 0] >= 0)
            & (uv[:, 0] < W)
            & (uv[:, 1] >= 0)
            & (uv[:, 1] < H)
        )
        valid = valid & in_img

    return uv[valid]


# -----------------------------------------------------------------------------
# Interactive GUI for manual LiDAR–image instance matching
# -----------------------------------------------------------------------------
def manual_match_gui(save_dir: str, frame_id: int) -> bool:
    """
    Launch an interactive GUI to manually match instance IDs
    between FRONT image and LiDAR instances.

    Parameters
    ----------
    save_dir : str
        Directory containing pre-saved data for this frame.
    frame_id : int
        Frame index within the TFRecord.

    Returns
    -------
    bool
        True if at least one valid pair is saved; False otherwise.
    """
    instance_img_path = os.path.join(save_dir, "instance_label_front.png")
    point_label_path = os.path.join(save_dir, "point_labels_all.txt")
    point_coord_path = os.path.join(save_dir, "points_all.txt")
    calib_path = os.path.join(save_dir, "calib.txt")
    rgb_img_path = os.path.join(save_dir, "img_raw.png")

    required_paths = [
        instance_img_path,
        point_label_path,
        point_coord_path,
        calib_path,
        rgb_img_path,
    ]
    if not all(os.path.exists(p) for p in required_paths):
        print(
            f"❌ Missing required files for GUI matching; "
            f"skipping frame directory: {save_dir}"
        )
        return False

    img = cv2.imread(instance_img_path, cv2.IMREAD_UNCHANGED)
    rgb_img = cv2.cvtColor(cv2.imread(rgb_img_path), cv2.COLOR_BGR2RGB)

    point_labels = np.loadtxt(point_label_path, dtype=np.int32)
    points = np.loadtxt(point_coord_path)

    # Ensure shape is (N, 2) or (N, >=2)
    if point_labels.ndim == 1:
        point_labels = point_labels.reshape(-1, 1)

    # Auto-trim to minimum length if length mismatch
    if point_labels.shape[0] != points.shape[0]:
        min_len = min(point_labels.shape[0], points.shape[0])
        print(
            f"⚠️ point_labels and points have different lengths; "
            f"auto-trimmed to {min_len} entries."
        )
        point_labels = point_labels[:min_len]
        points = points[:min_len]

    if point_labels.shape[0] != points.shape[0]:
        print(
            f"❌ Still mismatched lengths after trimming: "
            f"point_labels={point_labels.shape}, points={points.shape}"
        )
        return False

    # Handle colorbar reference inside closure
    colorbar_handle = [None]

    # Filter: keep only instance_id > 0, semantic_id == vehicle_semantic_id, X > 0
    vehicle_semantic_id = 1  # Adjust if your semantic id for vehicles differs
    mask = (
        (point_labels[:, 0] > 0)
        & (point_labels[:, 1] == vehicle_semantic_id)
        & (points[:, 0] > 0)
    )
    filtered_labels = point_labels[mask]
    filtered_points = points[mask]

    instance_counter = Counter(filtered_labels[:, 0])
    TOP_K = 7
    top_ids = [i for i, _ in instance_counter.most_common(TOP_K)]

    if len(top_ids) == 0:
        print(f"⚠️ Frame {frame_id} has no valid instance IDs; skipping.")
        return False

    fig = plt.figure(figsize=(49, 35))

    # Maximize window (delayed 100 ms) for Qt5Agg
    from PyQt5.QtCore import QTimer

    if matplotlib.get_backend() == "Qt5Agg":
        manager = plt.get_current_fig_manager()
        QTimer.singleShot(100, manager.window.showMaximized)

    # Grid layout: 2 rows x 2 columns
    gs = gridspec.GridSpec(
        2,
        2,
        height_ratios=[1, 1],
        width_ratios=[1, 1],
        wspace=0.5,
        hspace=0.2,
    )

    fig.subplots_adjust(left=0.08, right=0.92, top=0.95, bottom=0.1)

    # Subplots:
    # [0] Image instance labels
    # [1] LiDAR BEV projection (top K instances)
    # [2] RGB + all LiDAR points
    # [3] RGB + LiDAR points for top K instances
    axs = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
    ]

    for ax in axs:
        ax.set_aspect("equal", adjustable="box")

    fig.patch.set_edgecolor("black")
    fig.patch.set_linewidth(5)

    scale_factors = [1.0] * 4
    default_limits = [None] * 4
    for i, ax in enumerate(axs):
        default_limits[i] = (ax.get_xlim(), ax.get_ylim())

    selected_pixel = []  # (u, v) points on the image clicked by the user
    selected_lidar = []  # BEV points clicked by the user
    highlight_lidar_ids = []  # instance IDs selected on LiDAR side

    result = {"saved": False}

    # -----------------------------
    # Zoom & reset interactions
    # -----------------------------
    def on_scroll(event):
        for i, ax in enumerate(axs):
            if event.inaxes == ax:
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
                scale = 0.9 if event.button == "up" else 1.1
                scale_factors[i] *= scale
                x_center = (xlim[0] + xlim[1]) * 0.5
                y_center = (ylim[0] + ylim[1]) * 0.5
                x_range = (xlim[1] - xlim[0]) * scale
                y_range = (ylim[1] - ylim[0]) * scale
                ax.set_xlim(
                    [x_center - x_range / 2.0, x_center + x_range / 2.0]
                )
                ax.set_ylim(
                    [y_center - y_range / 2.0, y_center + y_range / 2.0]
                )
        fig.canvas.draw_idle()

    def on_reset(event):
        for i, ax in enumerate(axs):
            ax.set_xlim(default_limits[i][0])
            ax.set_ylim(default_limits[i][1])
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("scroll_event", on_scroll)

    ax_reset = plt.axes([0.69, 0.01, 0.1, 0.05])
    btn_reset = Button(ax_reset, "Reset View")
    btn_reset.on_clicked(on_reset)

    # -----------------------------
    # Initial plotting
    # -----------------------------
    cmap = tab60

    # (0) Image instance label map
    axs[0].imshow(img, cmap)
    axs[0].set_title("Image Instance Label")

    # Place instance IDs roughly at centers
    for inst_id in np.unique(img):
        if inst_id > 0:
            yx = np.argwhere(img == inst_id)
            y, x = yx[len(yx) // 2]
            axs[0].text(x, y, str(inst_id), color="yellow", fontsize=8)

    # (1) LiDAR BEV projection of top K instances
    axs[1].set_title("LiDAR BEV Projection (Top Instances)")
    axs[1].set_xlabel("-Y (m)")
    axs[1].set_ylabel("X (m)")
    axs[1].set_xlim(-27, 27)

    for idx, inst_id in enumerate(top_ids):
        pts = filtered_points[filtered_labels[:, 0] == inst_id]
        # Rotate (x, y) to (-y, x) for BEV rotation
        rot_pts = np.stack([-pts[:, 1], pts[:, 0]], axis=1)
        axs[1].scatter(
            rot_pts[:, 0],
            rot_pts[:, 1],
            s=1,
            color=cmap(idx),
            label=f"ID {inst_id}",
        )
        if len(rot_pts) > 0:
            cx, cy = np.mean(rot_pts[:, 0]), np.mean(rot_pts[:, 1])
            axs[1].text(cx, cy, f"{inst_id}", color=cmap(idx), fontsize=10)

    axs[1].legend()

    # (2) RGB + all LiDAR points (colored by depth)
    uv_all = project_lidar_to_image(points, calib_path)
    axs[2].imshow(rgb_img)
    depths = points[:, 2]
    sc = axs[2].scatter(
        uv_all[:, 0],
        uv_all[:, 1],
        s=0.2,
        c=depths[: uv_all.shape[0]],
        cmap="plasma",
        alpha=0.7,
    )
    axs[2].set_title("All LiDAR Points Projected to Image")
    axs[2].set_xlim(0, rgb_img.shape[1])
    axs[2].set_ylim(rgb_img.shape[0], 0)
    colorbar_handle[0] = plt.colorbar(
        sc, ax=axs[2], fraction=0.046, pad=0.04, label="Depth (m)"
    )

    # (3) RGB + semantic LiDAR points for top K instances
    axs[3].imshow(rgb_img)
    for idx, inst_id in enumerate(top_ids):
        pts = points[point_labels[:, 0] == inst_id]
        if len(pts) > 0:
            uv_pts = project_lidar_to_image(pts, calib_path)
            axs[3].scatter(
                uv_pts[:, 0],
                uv_pts[:, 1],
                s=1,
                color=cmap(idx),
                label=f"ID {inst_id}",
            )
    axs[3].set_title("Semantic Point Cloud Projection (Colored by Instance)")
    axs[3].set_xlim(0, rgb_img.shape[1])
    axs[3].set_ylim(rgb_img.shape[0], 0)
    axs[3].legend(markerscale=5)

    # -----------------------------
    # Mouse click interactions
    # -----------------------------
    def onclick(event):
        # Left-top: image instance label
        if event.inaxes == axs[0]:
            u, v = int(event.xdata), int(event.ydata)
            selected_pixel.append((u, v))
            axs[0].plot(u, v, "ro")

            inst_id = img[v, u]
            if inst_id > 0:
                yx = np.argwhere(img == inst_id)
                axs[0].scatter(
                    yx[:, 1], yx[:, 0], s=1, color="cyan", alpha=0.4
                )

        # Right-top: LiDAR BEV view
        elif event.inaxes == axs[1]:
            x_rot, y_rot = event.xdata, event.ydata  # This is (-Y, X) space
            selected_lidar.append((x_rot, y_rot))
            axs[1].plot(x_rot, y_rot, "bo")

            # Convert back to original LiDAR XY coordinates:
            # rot_pts = [-Y, X] → inverse = [Y = -x_rot, X = y_rot]
            clicked_ori = np.array([y_rot, -x_rot])  # [X, Y]

            best_id, min_dist = None, float("inf")
            for inst_id in top_ids:
                pts = filtered_points[filtered_labels[:, 0] == inst_id]
                dists = np.linalg.norm(pts[:, :2] - clicked_ori, axis=1)
                if dists.min() < min_dist:
                    best_id = inst_id
                    min_dist = dists.min()

            if best_id is not None:
                highlight_lidar_ids.append(best_id)
                pts = filtered_points[filtered_labels[:, 0] == best_id]
                rot_pts = np.stack([-pts[:, 1], pts[:, 0]], axis=1)
                axs[1].scatter(
                    rot_pts[:, 0],
                    rot_pts[:, 1],
                    s=20,
                    edgecolors="yellow",
                    facecolors="none",
                )

        fig.canvas.draw()

    fig.canvas.mpl_connect("button_press_event", onclick)

    # -----------------------------
    # Save / Clear / Skip buttons
    # -----------------------------
    def on_save(event):
        print("🔘 Save button clicked.")

        inst_id_pixel_list = []
        inst_id_lidar_list = list(highlight_lidar_ids)

        # From clicked pixels on image, collect their instance IDs
        for u, v in selected_pixel:
            inst_id = img[v, u]
            if inst_id > 0:
                inst_id_pixel_list.append(inst_id)

        # Number of pairs is the minimum length among pixel and LiDAR selections
        num_pairs = min(len(inst_id_pixel_list), len(inst_id_lidar_list))

        if num_pairs == 0:
            print(
                "⚠️ No valid matches found. "
                "This frame directory will be discarded and no CSV entry will be written."
            )
            result["saved"] = False
            plt.close()
            return

        # Save matched pairs
        for i in range(num_pairs):
            pixel_path = os.path.join(
                save_dir, f"match_pixel_coords_{i + 1}.txt"
            )
            lidar_path = os.path.join(
                save_dir, f"match_lidar_coords_{i + 1}.txt"
            )

            inst_pixel_id = inst_id_pixel_list[i]
            inst_lidar_id = inst_id_lidar_list[i]

            # Save image pixel coordinates
            with open(pixel_path, "w") as f1:
                yx_coords = np.argwhere(img == inst_pixel_id)
                f1.write(
                    f"frame: {frame_id}  instance_id: {inst_pixel_id}  "
                    f"total_pixels: {len(yx_coords)}\n"
                )
                for v_, u_ in yx_coords:
                    f1.write(f"{u_}, {v_}\n")

            # Save LiDAR 3D points
            with open(lidar_path, "w") as f2:
                mask_inst = point_labels[:, 0] == inst_lidar_id
                inst_pts = points[mask_inst]
                f2.write(
                    f"frame: {frame_id}  instance_id: {inst_lidar_id}  "
                    f"total_points: {inst_pts.shape[0]}\n"
                )
                for pt in inst_pts:
                    f2.write(f"{pt[0]:.4f}, {pt[1]:.4f}, {pt[2]:.4f}\n")

            print(
                f"💾 Saved the {i + 1}-th match for frame {frame_id} "
                f"({inst_pixel_id} ↔ {inst_lidar_id})."
            )

        # Update global pair_summary.csv for this sequence
        current_path = Path(save_dir).resolve()
        frame_id_str = current_path.name  # e.g. '29'
        tfrecord_dir = current_path.parent  # directory of this TFRecord
        seq_dir = tfrecord_dir.parent  # sequence directory

        seq_name = seq_dir.name
        tfrecord_name = tfrecord_dir.name
        frame_id_int = int(frame_id_str)

        csv_path = ensure_summary_csv_initialized(seq_dir)

        with open(csv_path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            for i in range(num_pairs):
                writer.writerow(
                    [
                        seq_name,
                        tfrecord_name,
                        frame_id_int,
                        f"match_pixel_coords_{i + 1}.txt",
                        f"match_lidar_coords_{i + 1}.txt",
                    ]
                )

        print(
            f"📄 Updated {csv_path.name} with {num_pairs} matched pair(s) "
            f"for frame {frame_id}."
        )
        result["saved"] = True
        plt.close()

    def on_clear(event):
        # Clear all selections and redraw original plots
        selected_pixel.clear()
        selected_lidar.clear()
        highlight_lidar_ids.clear()

        for ax in axs:
            ax.cla()

        cmap = tab60

        # (0) Image instance labels
        axs[0].imshow(img, cmap)
        axs[0].set_title("Image Instance Label")
        for inst_id in np.unique(img):
            if inst_id > 0:
                yx = np.argwhere(img == inst_id)
                y, x = yx[len(yx) // 2]
                axs[0].text(x, y, str(inst_id), color="yellow", fontsize=8)

        # (1) LiDAR BEV projection of top K instances
        axs[1].set_title("LiDAR BEV Projection (Top Instances)")
        axs[1].set_xlabel("-Y (m)")
        axs[1].set_ylabel("X (m)")
        axs[1].set_xlim(-27, 27)

        for idx, inst_id in enumerate(top_ids):
            pts = filtered_points[filtered_labels[:, 0] == inst_id]
            rot_pts = np.stack([-pts[:, 1], pts[:, 0]], axis=1)
            axs[1].scatter(
                rot_pts[:, 0],
                rot_pts[:, 1],
                s=1,
                color=cmap(idx),
                label=f"ID {inst_id}",
            )
            if len(rot_pts) > 0:
                cx, cy = np.mean(rot_pts[:, 0]), np.mean(rot_pts[:, 1])
                axs[1].text(cx, cy, f"{inst_id}", color=cmap(idx), fontsize=10)

        axs[1].legend()

        # (2) All LiDAR points projected to RGB
        axs[2].imshow(rgb_img)
        depths = points[:, 2]
        uv_all = project_lidar_to_image(points, calib_path)
        sc = axs[2].scatter(
            uv_all[:, 0],
            uv_all[:, 1],
            s=0.2,
            c=depths[: uv_all.shape[0]],
            cmap="plasma",
            alpha=0.7,
        )
        axs[2].set_title("All LiDAR Points Projected to Image")
        axs[2].set_xlim(0, rgb_img.shape[1])
        axs[2].set_ylim(rgb_img.shape[0], 0)

        # Reset colorbar
        if colorbar_handle[0] is not None:
            colorbar_handle[0].remove()
        colorbar_handle[0] = plt.colorbar(
            sc, ax=axs[2], fraction=0.046, pad=0.04, label="Depth (m)"
        )

        # (3) Semantic LiDAR points for top K instances
        axs[3].imshow(rgb_img)
        for idx, inst_id in enumerate(top_ids):
            pts = points[point_labels[:, 0] == inst_id]
            if len(pts) > 0:
                uv_pts = project_lidar_to_image(pts, calib_path)
                axs[3].scatter(
                    uv_pts[:, 0],
                    uv_pts[:, 1],
                    s=1,
                    color=cmap(idx),
                    label=f"ID {inst_id}",
                )
        axs[3].set_title(
            "Semantic Point Cloud Projection (Colored by Instance)"
        )
        axs[3].set_xlim(0, rgb_img.shape[1])
        axs[3].set_ylim(rgb_img.shape[0], 0)
        axs[3].legend(markerscale=5)

        fig.canvas.draw()

    def on_skip(event):
        print(f"⚠️ User chose to skip frame {frame_id}.")
        result["saved"] = False
        plt.close()

    # Button layout
    ax_clear = plt.axes([0.33, 0.01, 0.1, 0.05])
    ax_save = plt.axes([0.45, 0.01, 0.1, 0.05])
    ax_skip = plt.axes([0.57, 0.01, 0.1, 0.05])

    btn_clear = Button(ax_clear, "Clear Selection")
    btn_clear.on_clicked(on_clear)

    btn_save = Button(ax_save, "Save Match")
    btn_save.on_clicked(on_save)

    btn_skip = Button(ax_skip, "Skip Frame")
    btn_skip.on_clicked(on_skip)

    plt.suptitle(
        f"Frame {frame_id} Matching GUI (Left = Image, Right = LiDAR)",
        fontsize=18,
    )
    plt.show(block=True)

    return result["saved"]


# -----------------------------------------------------------------------------
# Frame processing & directory cleanup
# -----------------------------------------------------------------------------
def process_frames(filename: str, indices: list, save_root: str) -> None:
    """
    Process the given TFRecord file:

    - For each frame index in 'indices':
        - Extract calibration, LiDAR, and segmentation.
        - Save them to disk.
        - Open GUI for manual matching.
        - If user saves at least one match, keep the frame directory.
          Otherwise, delete it.
    """
    dataset = tf.data.TFRecordDataset(filename, compression_type="")
    for i, data in enumerate(dataset):
        if i not in indices:
            continue

        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))

        save_dir = os.path.join(save_root, str(i))
        make_dirs(save_dir)

        print(f"▶️ Frame {i}: saving calibration, LiDAR, and segmentation...")
        save_calib(frame, save_dir)
        save_lidar_and_labels(frame, save_dir)
        save_images_and_segmentation(frame, save_dir)

        print(f"▶️ Frame {i}: launching manual matching GUI...")
        matched = manual_match_gui(save_dir, i)

        if matched:
            print(f"💾 Frame {i}: at least one match saved. Keeping directory.")
        else:
            print(f"🚫 Frame {i}: no matches saved. Removing temporary directory.")
            shutil.rmtree(save_dir, ignore_errors=True)


def delete_empty_segment_dirs(root_dir: str) -> None:
    """
    Recursively delete all empty directories under 'root_dir'.

    Procedure:
        1) Remove any zero-byte CSV or hidden files first.
        2) If a directory becomes completely empty (no files, no subdirs),
           delete the directory itself.

    Parameters
    ----------
    root_dir : str
        Root directory to start cleanup from.
    """
    print("\n🧹 Starting cleanup of empty directories...")
    deleted_count = 0

    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
        # Remove zero-byte files (e.g., accidentally created empty CSVs)
        for name in list(filenames):
            fullp = os.path.join(dirpath, name)
            try:
                if os.path.isfile(fullp) and os.path.getsize(fullp) == 0:
                    os.remove(fullp)
                    filenames.remove(name)
                    print(f"   - Removed empty file: {fullp}")
            except FileNotFoundError:
                pass

        # Check if directory is now empty
        if not filenames and not dirnames:
            try:
                shutil.rmtree(dirpath)
                deleted_count += 1
                print(f"🗑️ Removed empty directory: {dirpath}")
            except Exception as e:
                print(f"   ! Failed to remove directory: {dirpath} → {e}")

    print(f"✅ Cleanup complete. Removed {deleted_count} empty directories.")


# -----------------------------------------------------------------------------
# Main entry
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    tfrecord_list = find_tfrecord_files_in_dir(DATASET_FOLDER)
    print(f"🔍 Detected {len(tfrecord_list)} TFRecord file(s) in {DATASET_FOLDER}")

    for tfrecord_path in tfrecord_list:
        tfrecord_name = os.path.basename(tfrecord_path)
        save_root = os.path.join(
            "waymo_segment_data", os.path.basename(DATASET_FOLDER), tfrecord_name
        )

        indices = extract_target_frame_indices(tfrecord_path)
        if not indices:
            print(
                f"⛔ Skipping {tfrecord_name}: "
                f"no valid segmentation frame or file is corrupted."
            )
            continue

        print(
            f"\n📂 Processing {tfrecord_name}: "
            f"{len(indices)} frame(s) have segmentation labels."
        )
        process_frames(tfrecord_path, indices, save_root)

    print("\n✅ All TFRecord files have been processed.")
    delete_empty_segment_dirs(
        os.path.join("waymo_segment_data", os.path.basename(DATASET_FOLDER))
    )

