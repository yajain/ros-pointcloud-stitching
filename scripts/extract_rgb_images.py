#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Extract RGB imagery from either a ROS bag or a live ROS topic (now supports per-crater folders and service-triggered capture)

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
DEFAULT_RGB_TOPIC = "/airsim_node/drone0/down_custom2/Scene"
DEFAULT_OUT = "rgb_output"

# ---- GLOBAL STATE ----
is_capturing = False
current_output_dir = None
frame_idx = 0
base_output_dir = DEFAULT_OUT

# ---- CONVERT ROS IMAGE TO NUMPY ----
def ros_img_to_cv2(msg):
    return ros_numpy.image.image_to_numpy(msg)

# ---- BAG MODE (unchanged) ----
def run_bag_mode(bag_path, rgb_topic, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(bag_path):
        print(f"ERROR: bag file not found: {bag_path}", file=sys.stderr)
        sys.exit(1)
    bag = rosbag.Bag(bag_path)
    for i, (_, msg, _) in enumerate(bag.read_messages(topics=[rgb_topic])):
        rgb_img = ros_img_to_cv2(msg)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        out_path = os.path.join(output_dir, f"rgb_{i:03d}.png")
        cv2.imwrite(out_path, rgb_img)
        print(f"Saved: {out_path}")
    bag.close()
    print(f"Done. Saved all RGB images to: {output_dir}")

# ---- SERVICE CALLBACKS ----
def handle_start_capture(req):
    global is_capturing, current_output_dir, frame_idx
    crater_name = req.crater_name.strip()
    if not crater_name:
        rospy.logwarn("Crater name is empty. Capture not started.")
        return StartCaptureResponse(success=False, message="Empty crater name")

    current_output_dir = os.path.join(base_output_dir, crater_name)
    os.makedirs(current_output_dir, exist_ok=True)
    frame_idx = 0
    is_capturing = True
    rospy.loginfo(f"[START] RGB capturing started for crater: {crater_name}")
    return StartCaptureResponse(success=True, message=f"RGB capture started for {crater_name}")

def handle_stop_capture(req):
    global is_capturing
    is_capturing = False
    rospy.loginfo("[STOP] RGB capturing stopped")
    return TriggerResponse(success=True, message="RGB capture stopped")

# ---- LIVE MODE ----
def run_live_mode(rgb_topic, output_dir):
    global base_output_dir
    base_output_dir = output_dir  # assign to global

    def cb(msg):
        global is_capturing, current_output_dir, frame_idx
        if not is_capturing or current_output_dir is None:
            return
        rgb_img = ros_img_to_cv2(msg)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        out_path = os.path.join(current_output_dir, f"rgb_{frame_idx:03d}.png")
        cv2.imwrite(out_path, rgb_img)
        print(f"Saved: {out_path}")
        frame_idx += 1

    rospy.init_node("rgb_extractor_live", anonymous=True)
    rospy.Subscriber(rgb_topic, Image, cb)
    rospy.Service("/start_capture_rgb", StartCapture, handle_start_capture)
    rospy.Service("/stop_capture_rgb", Trigger, handle_stop_capture)

    rospy.loginfo("RGB extractor running. Use /start_capture and /stop_capture to control saving.")
    rospy.spin()
    print("Node shutdown.")

# ---- CLI ----
def parse_args():
    p = argparse.ArgumentParser(description="Extract RGB images from ROS bag or live topic.")
    p.add_argument("--bag", type=str, default=DEFAULT_BAG, help="Path to ROS bag (default: %(default)s)")
    p.add_argument("--live", action="store_true", help="Use live ROS topic instead of a bag")
    p.add_argument("--rgb-topic", type=str, default=DEFAULT_RGB_TOPIC, help="RGB topic (default: %(default)s)")
    p.add_argument("--out", type=str, default=DEFAULT_OUT, help="Output directory (default: %(default)s)")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.live:
        run_live_mode(args.rgb_topic, args.out)
    else:
        run_bag_mode(args.bag, args.rgb_topic, args.out)
