#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Offline aligner: Match depth timestamps to closest pose messages and generate extrinsics

import os
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R

def load_pose_log(path):
    """
    Loads and sorts a pose log by timestamp.

    Args:
        path (str): Path to the pose log file (NumPy .npy format).

    Returns:
        list: A list of pose dictionaries sorted by their "timestamp" field.
    """
    pose_log = np.load(path, allow_pickle=True)
    return sorted(pose_log, key=lambda p: p["timestamp"])

def find_closest_pose(pose_log, t_depth):
    """
    Finds the closest pose in the pose log to a given depth timestamp.

    Args:
        pose_log (list): A list of pose dictionaries, each containing a "timestamp" field.
        t_depth (float): The timestamp of the depth frame to match.

    Returns:
        dict: The pose dictionary from the pose log that has the closest timestamp to the given depth timestamp.
    """
    return min(pose_log, key=lambda p: abs(p["timestamp"] - t_depth))

def build_extrinsic_matrix(position, quaternion):
    """
    Builds a 4x4 extrinsic transformation matrix from a position vector and a quaternion.

    Args:
        position (list or numpy.ndarray): A 3-element list or array representing the translation vector [x, y, z].
        quaternion (list or numpy.ndarray): A 4-element list or array representing the rotation as a quaternion [x, y, z, w].

    Returns:
        numpy.ndarray: A 4x4 extrinsic transformation matrix combining the rotation and translation.
    """
    rot = R.from_quat(quaternion).as_matrix()
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = rot
    extrinsic[:3, 3] = position
    return extrinsic

def main():
    parser = argparse.ArgumentParser(description="Align depth timestamps to pose log and output extrinsics.")
    parser.add_argument("--depth-ts", type=str, required=True, help="Path to depth_timestamps.npy")
    parser.add_argument("--pose-log", type=str, required=True, help="Path to pose_log.npy")
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory")
    args = parser.parse_args()

    crater_name = os.path.basename(os.path.dirname(args.depth_ts))
    output_dir = os.path.join(args.out_dir, crater_name)
    os.makedirs(output_dir, exist_ok=True)

    print("Loading data...")
    depth_timestamps = np.load(args.depth_ts, allow_pickle=True)
    pose_log = load_pose_log(args.pose_log)

    print("Matching depth frames to closest poses...")
    matched = []
    extrinsic_matrices = []

    for t_depth in depth_timestamps:
        best_pose = find_closest_pose(pose_log, t_depth)
        matched.append({
            "depth_timestamp": t_depth,
            "pose_timestamp": best_pose["timestamp"],
            "translation": best_pose["position"],
            "quaternion": best_pose["quaternion"]
        })
        extrinsic_matrices.append(
            build_extrinsic_matrix(best_pose["position"], best_pose["quaternion"])
        )

    # Save output
    np.save(os.path.join(output_dir, "extrinsics_data.npy"), np.array(matched, dtype=object))
    np.save(os.path.join(output_dir, "extrinsics_matrices.npy"), np.array(extrinsic_matrices))

    print(f"Saved {len(matched)} aligned extrinsics to {args.out_dir}")

if __name__ == "__main__":
    main()