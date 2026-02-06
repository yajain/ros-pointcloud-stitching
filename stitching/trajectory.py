#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Open3D-compatible trajectory JSON from extrinsic matrices and intrinsics.
"""

import os
import json
import yaml
import argparse
import numpy as np

def load_intrinsics(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    fx = data["fx"]
    fy = data["fy"]
    cx = data["cx"]
    cy = data["cy"]
    width = data["width"]
    height = data["height"]

    intrinsic_matrix = [
        fx, 0.0, 0.0,
        0.0, fy, 0.0,
        cx, cy, 1.0
    ]

    return intrinsic_matrix, width, height

def build_trajectory_json(extrinsics, intrinsic_matrix, width, height):
    trajectory = {
        "class_name": "PinholeCameraTrajectory",
        "parameters": [],
        "version_major": 1,
        "version_minor": 0
    }

    for E in extrinsics:
        param = {
            "class_name": "PinholeCameraParameters",
            "intrinsic": {
                "width":  width,
                "height": height,
                "intrinsic_matrix": [float(x) for x in intrinsic_matrix]
            },
            "extrinsic": [float(x) for x in E.flatten().tolist()]
        }
        trajectory["parameters"].append(param)

    return trajectory

def main():
    parser = argparse.ArgumentParser(description="Generate trajectory JSON from extrinsics and intrinsics.")
    parser.add_argument("--extrinsics", required=True, help="Path to extrinsic_matrices.npy")
    parser.add_argument("--intrinsics", required=True, help="Path to intrinsics YAML file")
    parser.add_argument("--output", default="trajectory.json", help="Output JSON file path")

    args = parser.parse_args()

    if not os.path.isfile(args.extrinsics):
        raise FileNotFoundError(f"Extrinsics file not found: {args.extrinsics}")
    if not os.path.isfile(args.intrinsics):
        raise FileNotFoundError(f"Intrinsics YAML file not found: {args.intrinsics}")

    extrinsics = np.load(args.extrinsics)
    if extrinsics.ndim != 3 or extrinsics.shape[1:] != (4, 4):
        raise ValueError("Extrinsics file must be a (N,4,4) numpy array")

    intrinsic_matrix, width, height = load_intrinsics(args.intrinsics)
    trajectory = build_trajectory_json(extrinsics, intrinsic_matrix, width, height)

    crater_name = os.path.basename(os.path.dirname(args.extrinsics))
    save_path = f"trajectory_{crater_name}.json"
    with open(save_path, "w") as f:
        json.dump(trajectory, f, indent=4)

    print(f"Saved trajectory with {len(extrinsics)} frames to {save_path}")

if __name__ == "__main__":
    main()