#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Extract camera intrinsics (fx, fy, cx, cy, width, height) from either a ROS bag or a live ROS topic (CameraInfo).

import os
import sys
import argparse
import yaml
import rosbag
import rospy
from sensor_msgs.msg import CameraInfo

# ---- CONFIG (defaults) ----
DEFAULT_BAG = "stationary_test3.bag"
DEFAULT_CAMINFO_TOPIC = "/airsim_node/drone0/down_custom/DepthPlanar/camera_info"
DEFAULT_OUTDIR = "intrinsics_output"

# ---- CONVERT CameraInfo TO DICT ----
def caminfo_to_intrinsics(msg: CameraInfo):
    return {
        "fx": float(msg.K[0]),
        "fy": float(msg.K[4]),
        "cx": float(msg.K[2]),
        "cy": float(msg.K[5]),
        "width": int(msg.width),
        "height": int(msg.height),
    }

# ---- SAVE INTRINSICS TO YAML ----
def save_yaml(path, data):
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)

# ---- BAG MODE ----
def run_bag_mode(bag_path, caminfo_topic, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    bag = rosbag.Bag(bag_path)
    for _, msg, _ in bag.read_messages(topics=[caminfo_topic]):
        intr = caminfo_to_intrinsics(msg)
        out_path = os.path.join(output_dir, "camera_intrinsics.yaml")
        save_yaml(out_path, intr)
        print(f"Saved intrinsics to: {out_path}")
        print(intr)
        bag.close()
        return
    bag.close()
    print(f"ERROR: No CameraInfo messages found for topic: {caminfo_topic}", file=sys.stderr)
    sys.exit(1)

# ---- LIVE MODE ----
def run_live_mode(caminfo_topic, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    rospy.loginfo(f"Waiting for CameraInfo on {caminfo_topic}...")
    msg = rospy.wait_for_message(caminfo_topic, CameraInfo)
    intr = caminfo_to_intrinsics(msg)
    out_path = os.path.join(output_dir, "camera_intrinsics.yaml")
    save_yaml(out_path, intr)
    print(f"Saved intrinsics to: {out_path}")
    print(intr)

# ---- MAIN ----
def main():
    p = argparse.ArgumentParser(description="Extract camera intrinsics from ROS CameraInfo messages (live or from rosbag)")
    p.add_argument("--bag", type=str, default=DEFAULT_BAG, help="Path to ROS bag (default: %(default)s)")
    p.add_argument("--live", action="store_true", help="Use live ROS topic instead of a bag")
    p.add_argument("--caminfo-topic", type=str, default=DEFAULT_CAMINFO_TOPIC, help="CameraInfo topic (default: %(default)s)")
    p.add_argument("--out", type=str, default=DEFAULT_OUTDIR, help="Output directory (default: %(default)s)")
    args = p.parse_args()

    if args.live:
        rospy.init_node("extract_intrinsics", anonymous=True)
        run_live_mode(args.caminfo_topic, args.out)
    else:
        run_bag_mode(args.bag, args.caminfo_topic, args.out)

if __name__ == "__main__":
    main()
