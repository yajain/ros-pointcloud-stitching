#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Extract drone pose messages (position + quaternion) from ROS bag or live topic, with per-crater output folders.

import os
import sys
import argparse
import numpy as np
import rosbag
import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from std_srvs.srv import Trigger, TriggerResponse
from pointcloud_stitching.srv import StartCapture, StartCaptureResponse

# ---- CONFIG (defaults) ----
DEFAULT_BAG = "stationary_test3.bag"
DEFAULT_POSE_TOPIC = "/drone1/mavros/global_position/local"
DEFAULT_OUT = "pose_output"

# ---- GLOBAL STATE ----
is_capturing = False
current_output_dir = None
base_output_dir = DEFAULT_OUT
pose_log = []

# ---- BAG MODE ----
def run_bag_mode(bag_path, pose_topic, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(bag_path):
        print(f"ERROR: bag file not found: {bag_path}", file=sys.stderr)
        sys.exit(1)

    bag = rosbag.Bag(bag_path)
    pose_data = []

    print(f"Reading pose messages from {bag_path} ...")
    for _, msg, _ in bag.read_messages(topics=[pose_topic]):
        ts = msg.header.stamp.to_sec()
        pos = msg.pose.pose.position
        quat = msg.pose.pose.orientation
        pose_data.append({
            "timestamp": ts,
            "position": np.array([pos.x, pos.y, pos.z]),
            "quaternion": np.array([quat.x, quat.y, quat.z, quat.w])
        })

    np.save(os.path.join(output_dir, "pose_log.npy"), np.array(pose_data, dtype=object))
    print(f"Saved {len(pose_data)} poses to: {output_dir}/pose_log.npy")
    bag.close()

# ---- SERVICE CALLBACKS ----
def handle_start_capture(req):
    global is_capturing, current_output_dir, pose_log
    crater_name = req.crater_name.strip()
    if not crater_name:
        rospy.logwarn("Crater name is empty. Capture not started.")
        return StartCaptureResponse(success=False, message="Empty crater name")
    current_output_dir = os.path.join(base_output_dir, crater_name)
    os.makedirs(current_output_dir, exist_ok=True)
    pose_log = []
    is_capturing = True
    rospy.loginfo(f"[START] Pose capture started for crater: {crater_name}")
    return StartCaptureResponse(success=True, message=f"Pose capture started for {crater_name}")

def handle_stop_capture(req):
    global is_capturing
    is_capturing = False
    rospy.loginfo("[STOP] Pose capture stopped")
    if current_output_dir and pose_log:
        out_path = os.path.join(current_output_dir, "pose_log.npy")
        np.save(out_path, np.array(pose_log, dtype=object))
        rospy.loginfo(f"[STOP] Saved {len(pose_log)} poses to {out_path}")
    return TriggerResponse(success=True, message="Pose capture stopped")

# ---- LIVE MODE ----
def run_live_mode(pose_topic, output_dir):
    global base_output_dir
    base_output_dir = output_dir

    def cb(msg):
        global is_capturing, current_output_dir, pose_log
        if not is_capturing or current_output_dir is None:
            return
        ts = msg.header.stamp.to_sec()
        pos = msg.pose.pose.position
        quat = msg.pose.pose.orientation
        pose_log.append({
            "timestamp": ts,
            "position": np.array([pos.x, pos.y, pos.z]),
            "quaternion": np.array([quat.x, quat.y, quat.z, quat.w])
        })

    rospy.init_node("pose_extractor", anonymous=True)
    rospy.Subscriber(pose_topic, Odometry, cb)
    rospy.Service("/start_capture_pose", StartCapture, handle_start_capture)
    rospy.Service("/stop_capture_pose",  Trigger, handle_stop_capture)

    rospy.loginfo("Pose extractor running. Use /start_capture and /stop_capture to control saving.")
    rospy.spin()
    print("Node shutdown.")

# ---- CLI ----
def parse_args():
    p = argparse.ArgumentParser(description="Extract pose messages from ROS bag or live topic.")
    p.add_argument("--bag", type=str, default=DEFAULT_BAG, help="Path to ROS bag (default: %(default)s)")
    p.add_argument("--live", action="store_true", help="Use live ROS topic instead of a bag")
    p.add_argument("--pose-topic", type=str, default=DEFAULT_POSE_TOPIC, help="Pose topic (default: %(default)s)")
    p.add_argument("--out", type=str, default=DEFAULT_OUT, help="Output base directory (default: %(default)s)")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.live:
        run_live_mode(args.pose_topic, args.out)
    else:
        run_bag_mode(args.bag, args.pose_topic, args.out)
