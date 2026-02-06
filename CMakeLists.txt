cmake_minimum_required(VERSION 3.0.2)
project(pointcloud_stitching)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  std_srvs
  message_generation
)

add_service_files(
  FILES
  StartCapture.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime std_msgs std_srvs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)