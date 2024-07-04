#!/bin/python3
# -*- coding: UTF-8 -*-
import rospy
from urdf_parser_py.urdf import URDF
from  pykdl_utils.kdl_parser import kdl_tree_from_urdf_model
from pykdl_utils.kdl_kinematics import KDLKinematics
import numpy as np
from sensor_msgs.msg import JointState
from std_msgs.msg import String,Float32MultiArray
from geometry_msgs.msg import PointStamped
from  RLPlanningNode.msg import detectResult
import time
# from inference import get_action
import json
import actionlib
import math
import tf
from getRefMotionFromFile import getRefMotionFromFile
from cloud_robot_msgs.msg import  ImageList,AnimationRequestGoal ,AnimationRequestAction,AnimationRequestActionFeedback
from cloud_robot_msgs.msg import RobotRTAnimation
import threading
from loguru import logger
from ultralytics import YOLO
from geometry3d import *
jointInOrder = ['knee', 'lumbar_yaw', 'lumbar_pitch','lumbar_roll', 'neck_yaw', 'neck_pitch', 'neck_roll', 'left_shoulder_pitch', 'left_shoulder_roll',
  'left_elbow_yaw', 'left_elbow_pitch', 'left_wrist_yaw', 'left_wrist_pitch', 'left_wrist_roll', 'right_shoulder_pitch',
  'right_shoulder_roll', 'right_elbow_yaw', 'right_elbow_pitch', 'right_wrist_yaw', 'right_wrist_pitch', 'right_wrist_roll', 
  'left_thumb', 'left_index', 'left_middle', 'left_ring', 'left_pinky', 'right_thumb', 'right_index', 'right_middle', 'right_ring', 'right_pinky']

def quat2matrix(q):
    x, y, z, w = q
    R = np.zeros([3, 3])
    R[0, 0] = 1 - 2 * y ** 2 - 2 * z ** 2
    R[0, 1] = 2 * x * y - 2 * z * w
    R[0, 2] = 2 * x * z + 2 * y * w
    R[1, 0] = 2 * x * y + 2 * z * w
    R[1, 1] = 1 - 2 * x ** 2 - 2 * z ** 2
    R[1, 2] = 2 * y * z - 2 * x * w
    R[2, 0] = 2 * x * z - 2 * y * w
    R[2, 1] = 2 * y * z + 2 * x * w
    R[2, 2] = 1 - 2 * x ** 2 - 2 * y ** 2
    return R

    
def main():
    pub = rospy.Publisher('/GraspStartWithConfig', String, queue_size=10)
    rospy.init_node('test', anonymous=True)
    test_string = String()
    test_string.data = '{"category":"cup"}'
    pub.publish(test_string)

    # test_float = Float32MultiArray()
    # test_float.data = np.concatenate((box[3],box[7]))
    # test_float.data = np.array([ -0.076792  , -0.093717   ,  0.47284,-0.0090824 ,  -0.055793   ,  0.59103])
    # test_float.data = test_float.data * 100
    # pub2.publish(test_float)
    print("already send test")
    # rospy.spin()

if __name__ == '__main__':
    main()