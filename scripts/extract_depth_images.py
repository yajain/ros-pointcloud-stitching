#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Extract depth imagery from either a ROS bag or a live ROS topic (with per-crater output folders)

import os
import sys
import argparse
import cv2
import numpy as np
np.float = np.float64  # temp fix for ros_numpy

import rosbag
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from std_srvs.srv import Trigger, TriggerResponse
from pointcloud_stitching.srv import StartCapture, StartCaptureResponse
import ros_numpy

# ---- CONFIG (defaults) ----
DEFAULT_BAG = "stationary_test3.bag"
DEFAULT_DEPTH_TOPIC = "/airsim_node/drone0/down_custom/DepthPlanar"
DEFAULT_OUT = "depth_output"

# ---- GLOBAL STATE ----
is_capturing = False
current_output_dir = None
frame_idx = 0
base_output_dir = DEFAULT_OUT
depth_timestamps = []  # NEW

# ---- CONVERT ROS IMAGE TO NUMPY ----
def ros_img_to_cv2(msg):
    return ros_numpy.image.image_to_numpy(msg)

# ---- BAG MODE (unchanged) ----
def run_bag_mode(bag_path, depth_topic, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(bag_path):
        print(f"ERROR: bag file not found: {bag_path}", file=sys.stderr)
        sys.exit(1)
    bag = rosbag.Bag(bag_path)
    for i, (_, msg, _) in enumerate(bag.read_messages(topics=[depth_topic])):
        depth_img = ros_img_to_cv2(msg)
        print(f"[Frame{i}] dtype: {depth_img.dtype}, min: {np.min(depth_img)}, max: {np.max(depth_img)}")
        raw_path = os.path.join(output_dir, f"depth_planar_{i:03d}.npy")
        np.save(raw_path, depth_img)
        depth_mm = (depth_img * 1000).astype(np.uint16)
        depth_path = os.path.join(output_dir, f"depth_fixed_{i:03d}.png")
        cv2.imwrite(depth_path, depth_mm)
        vis_img = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        vis_path = os.path.join(output_dir, f"depth_vis_{i:03d}.png")
        cv2.imwrite(vis_path, vis_img)
        print(f"Saved {raw_path} and {depth_path}")
        print(f"Saved {vis_path} and {raw_path}")
    bag.close()
    print(f"Done. Saved all images to: {output_dir}")

# ---- SERVICE CALLBACKS ----
def handle_start_capture(req):
    global is_capturing, current_output_dir, frame_idx, depth_timestamps
    crater_name = req.crater_name.strip()
    if not crater_name:
        rospy.logwarn("Crater name is empty. Capture not started.")
        return StartCaptureResponse(success=False, message = "Empty crater name")

    current_output_dir = os.path.join(base_output_dir, crater_name)
    os.makedirs(current_output_dir, exist_ok=True)
    frame_idx = 0
    depth_timestamps = []
    is_capturing = True
    rospy.loginfo(f"[START] Capturing started for crater: {crater_name}")
    return StartCaptureResponse(success=True, message=f"Depth capture started for {crater_name}")
    
def handle_stop_capture(req):
    global is_capturing
    is_capturing = False
    rospy.loginfo("[STOP] Capturing stopped")

    if current_output_dir and depth_timestamps:
        ts_path = os.path.join(current_output_dir, "depth_timestamps.npy")
        np.save(ts_path, np.array(depth_timestamps))
        rospy.loginfo(f"[STOP] Saved {len(depth_timestamps)} timestamps to {ts_path}")
    return TriggerResponse(success=True, message="Depth capture stopped")

# ---- LIVE MODE ----
def run_live_mode(depth_topic, output_dir):
    global base_output_dir
    base_output_dir = output_dir  # assign to global

    def cb(msg):
        global is_capturing, current_output_dir, frame_idx, depth_timestamps  # MODIFIED
        if not is_capturing or current_output_dir is None:
            return

        ts = msg.header.stamp.to_sec()  # NEW

        depth_img = ros_img_to_cv2(msg)
        print(f"[{current_output_dir}/Frame{frame_idx}] dtype: {depth_img.dtype}, min: {np.min(depth_img)}, max: {np.max(depth_img)}")

        raw_path = os.path.join(current_output_dir, f"depth_planar_{frame_idx:03d}.npy")
        np.save(raw_path, depth_img)

        depth_mm = (depth_img * 1000).astype(np.uint16)
        depth_path = os.path.join(current_output_dir, f"depth_fixed_{frame_idx:03d}.png")
        cv2.imwrite(depth_path, depth_mm)

        vis_img = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        vis_path = os.path.join(current_output_dir, f"depth_vis_{frame_idx:03d}.png")
        cv2.imwrite(vis_path, vis_img)

        depth_timestamps.append(ts)  # NEW
        print(f"Saved {raw_path}, {depth_path}, {vis_path}")
        frame_idx += 1

    rospy.init_node("depth_extractor_live", anonymous=True)
    rospy.Subscriber(depth_topic, Image, cb)
    rospy.Service("/start_capture_depth", StartCapture, handle_start_capture)
    rospy.Service("/stop_capture_depth", Trigger, handle_stop_capture)

    rospy.loginfo("Depth extractor running. Use /start_capture and /stop_capture to control saving.")
    rospy.spin()
    print("Node shutdown.")

# ---- CLI ----
def parse_args():
    p = argparse.ArgumentParser(description="Extract depth imagery from ROS bag or live topic.")
    p.add_argument("--bag", type=str, default=DEFAULT_BAG, help="Path to ROS bag (default: %(default)s)")
    p.add_argument("--live", action="store_true", help="Use live ROS topic instead of a bag")
    p.add_argument("--depth-topic", type=str, default=DEFAULT_DEPTH_TOPIC, help="Depth topic (default: %(default)s)")
    p.add_argument("--out", type=str, default=DEFAULT_OUT, help="Output base directory (default: %(default)s)")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.live:
        run_live_mode(args.depth_topic, args.out)
    else:
        run_bag_mode(args.bag, args.depth_topic, args.out)
