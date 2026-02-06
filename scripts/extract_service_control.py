#!/usr/bin/env python3
import rospy
from std_srvs.srv import Trigger, TriggerResponse
from pointcloud_stitching.srv import StartCapture, StartCaptureResponse

def handle_start(req):
    crater_name = req.crater_name.strip()
    if not crater_name:
        return StartCaptureResponse(success=False, message="Empty crater name")

    try:
        rospy.wait_for_service("/start_capture_rgb")
        rospy.wait_for_service("/start_capture_depth")
        rospy.wait_for_service("/start_capture_pose")

        start_rgb   = rospy.ServiceProxy("/start_capture_rgb",   StartCapture)
        start_depth = rospy.ServiceProxy("/start_capture_depth", StartCapture)
        start_pose  = rospy.ServiceProxy("/start_capture_pose",  StartCapture)

        r1 = start_rgb(crater_name)
        r2 = start_depth(crater_name)
        r3 = start_pose(crater_name)

        ok  = (r1.success and r2.success and r3.success)
        msg = f"RGB:{r1.success} DEPTH:{r2.success} POSE:{r3.success}"
        return StartCaptureResponse(success=ok, message=msg)
    except Exception as e:
        return StartCaptureResponse(success=False, message=str(e))

def handle_stop(req):
    try:
        rospy.wait_for_service("/stop_capture_rgb")
        rospy.wait_for_service("/stop_capture_depth")
        rospy.wait_for_service("/stop_capture_pose")

        stop_rgb   = rospy.ServiceProxy("/stop_capture_rgb",   Trigger)
        stop_depth = rospy.ServiceProxy("/stop_capture_depth", Trigger)
        stop_pose  = rospy.ServiceProxy("/stop_capture_pose",  Trigger)

        s1 = stop_rgb()
        s2 = stop_depth()
        s3 = stop_pose()

        ok  = (s1.success and s2.success and s3.success)
        msg = f"RGB:{s1.success} DEPTH:{s2.success} POSE:{s3.success}"
        return TriggerResponse(success=ok, message=msg)
    except Exception as e:
        return TriggerResponse(success=False, message=str(e))

if __name__ == "__main__":
    rospy.init_node("capture_controller")
    rospy.Service("/start_capture_all", StartCapture, handle_start)  # StartCapture so we can pass crater_name
    rospy.Service("/stop_capture_all",  Trigger,      handle_stop)
    rospy.loginfo("Ready: /start_capture_all and /stop_capture_all")
    rospy.spin()
