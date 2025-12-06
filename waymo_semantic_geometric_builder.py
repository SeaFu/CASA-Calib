#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Waymo Semantic–Geometric Extractor (for CASA-Calib Dataset Construction)
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
    • instance-level geometric consistency checking
    • manual / automatic semantic verification
    • cross-sensor semantic reasoning
    • robust evaluation of semantic-assisted calibration algorithms

This extractor is a key component in the CASA-Calib pipeline for constructing
clean, reliable benchmark data for LiDAR–camera extrinsic calibration research.

----------------------------------------------------------------------------
"""

import os
import numpy as np
import tensorflow.compat.v1 as tf
import cv2

from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import (
    camera_segmentation_utils,
    transform_utils,
    range_image_utils,
)

tf.enable_eager_execution()

# ======================================================================
# User Configuration
# ======================================================================

DATASET_FOLDER = "/mnt/d/waymo_1.4.3/04"   # Path containing .tfrecord files

# ======================================================================
# Utility Functions
# ======================================================================

def make_dirs(path):
    """Create directory if not exists."""
    if not os.path.exists(path):
        os.makedirs(path)


def cart_to_homo(mat):
    """Convert 3×3 or 3×4 matrix to homogeneous 4×4."""
    ret = np.eye(4)
    if mat.shape == (3, 3):
        ret[:3, :3] = mat
    elif mat.shape == (3, 4):
        ret[:3, :] = mat
    else:
        raise ValueError(f"Unsupported matrix shape: {mat.shape}")
    return ret


def find_tfrecord_files(folder_path):
    """Return all .tfrecord files in sorted order."""
    return sorted([
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.endswith(".tfrecord")
    ])


def extract_valid_frame_indices(filename):
    """
    Scan the .tfrecord and identify frames containing BOTH:
        • camera panoptic segmentation label
        • LiDAR range-image segmentation label

    These frames are the only ones suitable for semantic–geometric testing.
    """
    try:
        dataset = tf.data.TFRecordDataset(filename, compression_type="")
        indices = []

        for idx, data in enumerate(dataset):
            try:
                frame = open_dataset.Frame()
                frame.ParseFromString(bytearray(data.numpy()))

                cam_ok = frame.images[0].camera_segmentation_label.panoptic_label
                lidar_ok = frame.lasers[0].ri_return1.segmentation_label_compressed

                if cam_ok and lidar_ok:
                    indices.append(idx)

            except tf.errors.DataLossError as e:
                print(f"[Warning] Skipping corrupted frame {idx}: {e}")
                continue

        return indices

    except tf.errors.DataLossError as e:
        print(f"[Error] Unable to read file {filename}: {e}")
        return []


def parse_range_image_and_camera_projection(frame):
    """
    Parse raw Waymo range images, segmentation labels, and camera projections.
    """
    range_images, camera_projections, segmentation_labels = {}, {}, {}
    range_image_top_pose = None

    for laser in frame.lasers:

        # Return 1
        if len(laser.ri_return1.range_image_compressed) > 0:
            ri1 = open_dataset.MatrixFloat()
            ri1.ParseFromString(
                tf.decode_compressed(laser.ri_return1.range_image_compressed, "ZLIB").numpy()
            )
            range_images[laser.name] = [ri1]

            # Pose (for TOP LiDAR)
            if laser.name == open_dataset.LaserName.TOP:
                pose = open_dataset.MatrixFloat()
                pose.ParseFromString(
                    tf.decode_compressed(laser.ri_return1.range_image_pose_compressed, "ZLIB").numpy()
                )
                range_image_top_pose = pose

            # Camera projection
            cp1 = open_dataset.MatrixInt32()
            cp1.ParseFromString(
                tf.decode_compressed(laser.ri_return1.camera_projection_compressed, "ZLIB").numpy()
            )
            camera_projections[laser.name] = [cp1]

            # Segmentation labels
            if laser.ri_return1.segmentation_label_compressed:
                sl1 = open_dataset.MatrixInt32()
                sl1.ParseFromString(
                    tf.decode_compressed(laser.ri_return1.segmentation_label_compressed, "ZLIB").numpy()
                )
                segmentation_labels[laser.name] = [sl1]

        # Return 2
        if len(laser.ri_return2.range_image_compressed) > 0:

            ri2 = open_dataset.MatrixFloat()
            ri2.ParseFromString(
                tf.decode_compressed(laser.ri_return2.range_image_compressed, "ZLIB").numpy()
            )
            range_images[laser.name].append(ri2)

            cp2 = open_dataset.MatrixInt32()
            cp2.ParseFromString(
                tf.decode_compressed(laser.ri_return2.camera_projection_compressed, "ZLIB").numpy()
            )
            camera_projections[laser.name].append(cp2)

            if laser.ri_return2.segmentation_label_compressed:
                sl2 = open_dataset.MatrixInt32()
                sl2.ParseFromString(
                    tf.decode_compressed(laser.ri_return2.segmentation_label_compressed, "ZLIB").numpy()
                )
                segmentation_labels[laser.name].append(sl2)

    return range_images, camera_projections, segmentation_labels, range_image_top_pose


def save_calibration(frame, save_dir):
    """
    Extract and save camera intrinsic/extrinsic + LiDAR→camera transform.
    """
    for cam in frame.context.camera_calibrations:
        if cam.name == 1:  # FRONT camera
            T_front_cam_to_vehicle = np.array(cam.extrinsic.transform).reshape(4, 4)
            intrinsic = np.zeros((3, 4))
            intrinsic[0, 0], intrinsic[1, 1] = cam.intrinsic[0], cam.intrinsic[1]
            intrinsic[0, 2], intrinsic[1, 2], intrinsic[2, 2] = cam.intrinsic[2], cam.intrinsic[3], 1
            break

    # Convert to KITTI-style reference
    T_ref = np.array([[0.0, -1.0, 0.0],
                      [0.0,  0.0, -1.0],
                      [1.0,  0.0,  0.0]])

    Tr_velo_to_cam = cart_to_homo(T_ref) @ np.linalg.inv(T_front_cam_to_vehicle)

    # Save calibration in KITTI format
    calib_text = ""
    for i in range(4):
        if i == 2:
            P = intrinsic.reshape(12)
        else:
            P = np.eye(4)[:3, :].reshape(12)
        calib_text += f"P{i}: {' '.join(map(str, P))}\n"

    calib_text += f"R0_rect: {' '.join(map(str, np.eye(3).flatten()))}\n"
    calib_text += f"Tr_velo_to_cam: {' '.join(map(str, Tr_velo_to_cam[:3, :].reshape(12)))}\n"

    with open(os.path.join(save_dir, "calib.txt"), "w+") as f:
        f.write(calib_text)


def convert_range_image_to_point_cloud_labels(frame, range_images, segmentation_labels, ri_index=0):
    """
    Convert LiDAR range-image segmentation label into per-point labels.
    """
    calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
    point_labels = []

    for c in calibrations:
        ri_tensor = tf.reshape(
            tf.convert_to_tensor(range_images[c.name][ri_index].data),
            range_images[c.name][ri_index].shape.dims,
        )

        mask = ri_tensor[..., 0] > 0

        if c.name in segmentation_labels:
            sl_tensor = tf.reshape(
                tf.convert_to_tensor(segmentation_labels[c.name][ri_index].data),
                segmentation_labels[c.name][ri_index].shape.dims,
            )
            sl_points = tf.gather_nd(sl_tensor, tf.where(mask))
        else:
            sl_points = tf.zeros([tf.reduce_sum(tf.cast(mask, tf.int32)), 2], dtype=tf.int32)

        point_labels.append(sl_points.numpy())

    return point_labels


def convert_range_image_to_point_cloud(frame, range_images, camera_projections, range_image_top_pose, ri_index=0):
    """
    Convert LiDAR range images into XYZ point clouds.
    """
    filter_no_label_zone_points = True
    calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)

    points, cp_points, intensity = [], [], []

    frame_pose = tf.convert_to_tensor(np.reshape(np.array(frame.pose.transform), [4, 4]))
    pose_tensor = tf.reshape(tf.convert_to_tensor(range_image_top_pose.data), range_image_top_pose.shape.dims)

    pose_rot = transform_utils.get_rotation_matrix(pose_tensor[..., 0],
                                                   pose_tensor[..., 1],
                                                   pose_tensor[..., 2])
    pose_trans = pose_tensor[..., 3:]
    top_pose = transform_utils.get_transform(pose_rot, pose_trans)

    for c in calibrations:

        range_image = range_images[c.name][ri_index]

        # Determine inclinations
        if not c.beam_inclinations:
            inclinations = range_image_utils.compute_inclination(
                tf.constant([c.beam_inclination_min, c.beam_inclination_max]),
                height=range_image.shape.dims[0],
            )
        else:
            inclinations = tf.constant(c.beam_inclinations)

        inclinations = tf.reverse(inclinations, [-1])

        extrinsic = np.reshape(np.array(c.extrinsic.transform), [4, 4])

        ri_tensor = tf.reshape(tf.convert_to_tensor(range_image.data),
                               range_image.shape.dims)

        mask = ri_tensor[..., 0] > 0
        if filter_no_label_zone_points:
            mask &= (ri_tensor[..., 3] != 1.0)

        pixel_pose = tf.expand_dims(top_pose, axis=0) if c.name == open_dataset.LaserName.TOP else None
        frame_pose_local = tf.expand_dims(frame_pose, axis=0) if c.name == open_dataset.LaserName.TOP else None

        ri_cart = range_image_utils.extract_point_cloud_from_range_image(
            tf.expand_dims(ri_tensor[..., 0], 0),
            tf.expand_dims(extrinsic, 0),
            tf.expand_dims(inclinations, 0),
            pixel_pose=pixel_pose,
            frame_pose=frame_pose_local,
        )

        points_tensor = tf.gather_nd(tf.squeeze(ri_cart, axis=0), tf.where(mask))
        points.append(points_tensor.numpy())

    return points, None, None


def save_lidar_and_labels(frame, save_dir):
    """Save LiDAR XYZ point cloud and semantic label file."""
    ri, cp, labels, pose = parse_range_image_and_camera_projection(frame)
    pts, _, _ = convert_range_image_to_point_cloud(frame, ri, cp, pose)
    labels = convert_range_image_to_point_cloud_labels(frame, ri, labels)

    np.savetxt(os.path.join(save_dir, "points_all.txt"), np.concatenate(pts, axis=0))
    np.savetxt(os.path.join(save_dir, "point_labels_all.txt"), np.concatenate(labels, axis=0))

    np.concatenate(pts, axis=0).astype(np.float32).tofile(os.path.join(save_dir, "lidar.bin"))


def save_images_and_segmentation(frame, save_dir):
    """Save raw RGB image, panoptic mask, and instance segmentation."""

    # Save camera RGB
    img = frame.images[0]
    rgb = cv2.cvtColor(
        cv2.imdecode(np.frombuffer(img.image, np.uint8), cv2.IMREAD_COLOR),
        cv2.COLOR_RGB2BGR,
    )
    cv2.imwrite(os.path.join(save_dir, "img_raw.png"), rgb)

    # Decode camera panoptic segmentation
    panoptic = camera_segmentation_utils.decode_single_panoptic_label_from_proto(
        frame.images[2].camera_segmentation_label
    )

    sem, inst = camera_segmentation_utils.decode_semantic_and_instance_labels_from_panoptic_label(
        panoptic,
        frame.images[2].camera_segmentation_label.panoptic_label_divisor,
    )

    # Save masks
    cv2.imwrite(os.path.join(save_dir, "panoptic_label_front.png"), panoptic.astype(np.uint16))
    cv2.imwrite(os.path.join(save_dir, "instance_label_front.png"), inst.astype(np.uint16))
    cv2.imwrite(os.path.join(save_dir, "instance_waymo.png"), inst)


def process_frames(tfrecord_path, indices, save_root):
    """Process selected frames within a .tfrecord file."""
    dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type="")

    for idx, data in enumerate(dataset):

        if idx not in indices:
            continue

        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))

        save_dir = os.path.join(save_root, str(idx))
        make_dirs(save_dir)

        print(f"[INFO] Processing frame {idx} ...")

        save_calibration(frame, save_dir)
        save_lidar_and_labels(frame, save_dir)
        save_images_and_segmentation(frame, save_dir)


# ======================================================================
# Main Execution
# ======================================================================

if __name__ == "__main__":

    tfrecord_list = find_tfrecord_files(DATASET_FOLDER)
    print(f"[INFO] Detected {len(tfrecord_list)} .tfrecord files.")

    for tfrecord_path in tfrecord_list:

        filename = os.path.basename(tfrecord_path)
        save_root = os.path.join("waymo_segment_data", os.path.basename(DATASET_FOLDER), filename)

        indices = extract_valid_frame_indices(tfrecord_path)

        if not indices:
            print(f"[Warning] Skipping {filename}: no valid segmentation frames.")
            continue

        print(f"\n[INFO] Processing {filename} — {len(indices)} valid segmentation frames found.")
        process_frames(tfrecord_path, indices, save_root)

    print("\n[DONE] All .tfrecord files processed successfully.")
