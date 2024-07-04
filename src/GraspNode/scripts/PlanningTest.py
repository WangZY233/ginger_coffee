#!/bin/python3
# -*- coding: UTF-8 -*-
import json
import math
import queue
import threading
import time
from datetime import datetime

import actionlib
import cv2
import numpy as np
import rospy
import tf
import torch
from cloud_robot_msgs.msg import (AnimationRequestAction, AnimationRequestGoal,
                                  ChassisMoveAction, ChassisMoveGoal,
                                  ImageList, RobotRTAnimation)
from geometry3d import *
from geometry_msgs.msg import PointStamped
from getRefMotionFromFile import getRefMotionFromFile
from inference import get_action, get_trajectory, get_trajectory_joint
from loguru import logger
from pykdl_utils.kdl_kinematics import KDLKinematics
from pykdl_utils.kdl_parser import kdl_tree_from_urdf_model
from RLPlanningNode.msg import detectResult
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float32MultiArray, String
from std_srvs.srv import SetBool, SetBoolRequest, SetBoolResponse, Trigger
from ultralytics import YOLO
from urdf_parser_py.urdf import URDF

logger.add(f'log/Plan_{datetime.now().strftime("%m%d_%H%M")}.log', format="{time} {level} {message}", level="DEBUG")

jointInOrder = ['knee', 'lumbar_yaw', 'lumbar_pitch','lumbar_roll', 'neck_yaw', 'neck_pitch', 'neck_roll', 
                'left_shoulder_pitch', 'left_shoulder_roll', 'left_elbow_yaw', 'left_elbow_pitch', 'left_wrist_yaw', 'left_wrist_pitch', 'left_wrist_roll', 
                'right_shoulder_pitch', 'right_shoulder_roll', 'right_elbow_yaw', 'right_elbow_pitch', 'right_wrist_yaw', 'right_wrist_pitch', 'right_wrist_roll', 
                'left_thumb', 'left_index', 'left_middle', 'left_ring', 'left_pinky', 
                'right_thumb', 'right_index', 'right_middle', 'right_ring', 'right_pinky']


class RLPlanning():
    def __init__(self):
        file = './src/ginger_description/urdf/ginger_description.urdf'
        robot = URDF.from_xml_file(file)
        self.jointstatemsg = None
        # self.cube = None
        self.cube = np.zeros((1,6))
        self.idlist =  [i for i in range(4,35)]
        self.current_jointstate=np.zeros(31)
        self.poslist=np.zeros(31)
        self.left_arm = np.zeros(7)
        self.right_arm = np.zeros(7)
        self.is_left = False
        self.is_both = False
        self.is_init = False
        self.is_cv_start=False
        self.chooseGraspArm=0  #0 means not init,1 means left arm ,2 means right arm
        self.tarJointPos = None
        self.action=np.zeros(7)
        tree = kdl_tree_from_urdf_model(robot)
        chain = tree.getChain("lumbar_roll_link","right_wrist_roll_link")
        chain_left = tree.getChain("lumbar_roll_link", "left_wrist_roll_link")
        self.listener = tf.TransformListener()
        
        self.isStartPlanning = False
        self.isStartGrasp = True
        # 抓杯子判断结束
        self.RL_Over = False
        # 按按钮判断结束
        self.RL_button_Over = False
        self.RL_Start = False
        self.startTime=0
        self.cup_stop_loc=0.0
        self.cup_stop_loc_x=0.0
        self.cup_stop_loc_y=0.0
        
        self.pose=None
        #debug for inverse kinematic
        self.frame = np.zeros((30,7))
        self.kdl_kin = KDLKinematics(robot,"lumbar_roll_link","right_wrist_roll_link")
        self.kdl_kin_left = KDLKinematics(robot, "lumbar_roll_link", "left_wrist_roll_link")
        self.rate = rospy.Rate(30) 
        self.rate_rl = rospy.Rate(15) 
        self.distance =0
        
        #选定的物品类别和跟踪id
        self.category_str = 'chunzhensuannai'
        self.category_list = 'chunzhensuannai'
        self.trackid=None
        #选定物品的3d坐标
        self.objPosInBackY=np.zeros(3)
        self.count=0
        self.step=0
        #for debug
        self.pub = rospy.Publisher("/point",PointStamped,queue_size=10)
        self.pub1 = rospy.Publisher("/joint_states", JointState, queue_size=10)
        self.control_pub = rospy.Publisher("/GraspStartWithConfig",String,queue_size=10)
        self.transPub = rospy.Publisher("/Float32MultiArray",Float32MultiArray, queue_size=10)

        # 运动控制
        self.pub2 = rospy.Publisher("/ginger/robot_animation/realtime_animation", RobotRTAnimation, queue_size=1)
        # 获取当前关节信息
        rospy.Subscriber("/ginger/joint_states", JointState, self.jointstate_callback)
        # 抓取开始
        rospy.Subscriber("/GraspStartWithConfig",String,self.graspStart_callback)
        # IK
        rospy.Subscriber("/LinearPlanning",String,self.KinematicInverse)
        
        # 打开摄像头服务
        self.K_client = rospy.ServiceProxy("/iris_camera/camera_parameters", Trigger)
        self.open_camera_client = rospy.ServiceProxy("/iris_camera/resume", Trigger)
        self.close_camera_client = rospy.ServiceProxy("/iris_camera/pause", Trigger)
        self.open_camera()
        
        rospy.spin()
        
    def open_camera(self):
        self.open_camera_client()
        response = self.K_client()
        intrinsics = json.loads(response.message)["rgb_camera"]["intrinsic"]
        self.K = np.array([[intrinsics["fx"], 0, intrinsics["cx"]],
                           [0, intrinsics["fy"], intrinsics["cy"]],
                           [0, 0, 1]])

    def close_camera(self):
        self.close_camera_client()
    
    #订阅机器人关节状态
    def jointstate_callback(self,data):
        self.jointstatemsg = data
        for i in range(len(jointInOrder)):
            for  j in range(len(data.name)):
                if jointInOrder[i] == data.name[j]:
                    self.current_jointstate[i] = data.position[j]
                    break
        # 计算末端位置
        if self.is_left:
            joint_temp_left = self.current_jointstate[7:14].copy()
            self.pose = self.kdl_kin_left.forward(joint_temp_left[:7])
            self.left_arm = joint_temp_left * 180 / 3.14159
            self.left_arm[2]*=-1
            self.left_arm[4]*=-1
            self.left_arm[6]*=-1
        else:
            joint_temp_right = self.current_jointstate[14:21].copy()
            self.pose = self.kdl_kin.forward(joint_temp_right[:7])
            self.right_arm = joint_temp_right * 180 / 3.14159
            self.right_arm[2]=-self.right_arm[2]
            self.right_arm[4]=-self.right_arm[4]
            self.right_arm[6]=-self.right_arm[6]
        self.current_jointstate[16]= -self.current_jointstate[16]
        self.current_jointstate[18]= -self.current_jointstate[18]
        self.current_jointstate[9]= -self.current_jointstate[9]
        self.current_jointstate[11]= -self.current_jointstate[11]
        handpose = np.array([self.pose[0,3],self.pose[1,3],self.pose[2,3]])
        self.distance = np.linalg.norm(handpose-self.objPosInBackY,ord=2)
        if self.distance<0.16 and self.isStartPlanning and (not self.isStartGrasp):
            self.isStartPlanning=False
            self.isStartGrasp = True
            print("hand close to object")
            self.graspAndHandUp()
    
    def start(self):
        
        rospy.sleep(3)
        self.startTime = int(round(time.time()*1000))
        self.isStartPlanning = True
        
        self.checkRobotStatus()
        # self.chassis_move(x = -0.05,y = -0.8,yaw = 0,t = 3)
        
        # FIXME
        self.HeadDown_button()
        # FIXME !!!!!!!!!
        # self.play_data('press_button_init')
        rospy.sleep(1)

        print("-------------")
        while not rospy.is_shutdown():
            print("开始点按钮")
            self.RL_Service_button(self.button_loc)         
            print("咖啡制作中...等待30s")
            self.checkRobotStatus()
            rospy.sleep(3)
            
            # 抓杯子
            print("开始抓杯子")
            self.HeadDown_cup()
            # rospy.sleep(0.5)
            self.cup_loc = self.cube
            self.RL_Service_cup(self.cup_loc)
            break
        
        self.isStartGrasp = False

    def checkRobotStatus(self):
        current = self.current_jointstate.copy()
        current[5] = 0
        if np.linalg.norm(self.poslist - current, ord=1) > 30.0:
            print("robot is not in idle state")
            self.Handmove2(rtarHandFrame=np.array([36,-40,40,-90,25,0,0],dtype=np.float32)*np.pi/180,
                       ltarHandFrame=np.array([36,40,-40,-90,-25,-0,-0],dtype=np.float32)*np.pi/180)
            self.Handmove2(np.zeros(7),np.zeros(7))
            self.resetState()
    
    # 发布实时控制消息, is_anim为True时，使用rate控制发布频率
    def realTimeAnimPub(self,pos_list,is_anim):
        if len(pos_list)!=31:
            rospy.logwarn("list length is not 31")
            return
        jointmsg = RobotRTAnimation()
        jointmsg.request_type = 0
        jointmsg.time = int(round(time.time()*1000)) - self.startTime
        jointmsg.id_list =self.idlist
        jointmsg.pos_list = pos_list.copy()
        if np.linalg.norm(pos_list - self.current_jointstate, ord=1) > 30.0:
            print("control step too large, ignore the action")
            return
        self.pub2.publish(jointmsg)
        self.poslist = pos_list.copy()
        if is_anim:
            self.rate.sleep()
        else:
            self.rate_rl.sleep()        
        
    '''开始关节控制的action回调'''
    def done_callback(self,state,res):
        rospy.loginfo(state)
        rospy.loginfo(res)

    def feedback_callback(self,fb):
        rospy.loginfo('feedback')
        print(fb.progress)
        if fb.progress==0:
            self.startTime = int(round(time.time()*1000))

            '''trajectory test'''
            self.back_to_button_joint(is_left=False)
            # time.sleep(1)
            # self.play_data('putdown_cup') # FIXME "grasp_coffee_init" putdown_cup
            
            # self.chassis_move(x = 0.5,y = 0,yaw = 0,t = 1)

            # 播放轨迹
            # self.FixedTrajectory()
            # 连续发送相同位置
            # self.joint_test()
            
    def resetState(self):
        self.is_both = False
        self.is_left = False #确定使用左/右手抓取
        self.is_init = False #防止多次同时触发抓取流程
        self.is_cv_start=False #确认CV服务已经启动
        self.isStartPlanning = False #进入抬手准备阶段
        self.isStartGrasp = True 
        self.isUpdateCVResult = True
        self.RL_Start = False
        self.RL_Over = False
        self.chooseGraspArm=0  #0 means not init,1 means left arm ,2 means right arm
        self.poslist = np.zeros(31)
        #send request_type=1 stop realtime control 
        jointmsg = RobotRTAnimation()
        jointmsg.request_type = 1
        self.pub2.publish(jointmsg)

    def JsonParseForStartGrasp(self,data):
        json_str = json.loads(data.data,strict=False)
        print("*****************")
        print(json_str)
        
        self.category_str =json_str['category']
        self.category_list = json_str['category'].split(" ")
        rospy.loginfo("category is %s",self.category_str)

    #开始示反馈抓取服务
    def graspStart_callback(self,data = None):
        if self.is_init:
            print("抓取任务中")
            return
        self.is_init = True
        # while not self.is_cv_start:
        #     time.sleep(0.05)
        print("grasp start")

        #save category（类别） from cloud instru          
        # 存储类别信息
        self.JsonParseForStartGrasp(data)
        self.isStartPlanning = True 

        # 请求机器人控制权
        client = actionlib.SimpleActionClient('/ginger/robot_animation/animation_request',AnimationRequestAction)
        print("wait for server!")
        client.wait_for_server()
        print("sercer connected!")

        goal = AnimationRequestGoal()
        goal.request.animation_type = 11
        goal.request.group = 'upper_limb_and_head'
        # goal.request.controller = 'upper_limb_and_head_complance_controller' # 柔顺控制
        goal.request.controller = 'upper_limb_and_head_pos_controller'  # 位置控制

        
        goal.request.priority = 'MEDIUM'
        print("send goal")
        client.send_goal(goal,self.done_callback,feedback_cb=self.feedback_callback)
        client.wait_for_result()#添加以解决有时不能进入feedback_cb的问题
    
    # 左右手运动函数
    def Handmove2(self,rtarHandFrame,ltarHandFrame,finger = False,openfinger = False, frame_num=90.0):
        frameNum = 0
        jointStep = self.current_jointstate.copy()
        resetHandFrameL = jointStep[7:14].copy()
        resetHandFrameR = jointStep[14:21].copy()
        startHandFrame = jointStep[26:31].copy()
        if finger:
            HandPos = np.array([50,0,50,50,50],dtype=np.float32)*np.pi/180
        elif openfinger:
            HandPos = np.array([0,0,0,0,0],dtype=np.float32)*np.pi/180
        else:
            HandPos = jointStep[26:31].copy()
        while not rospy.is_shutdown() and self.isStartPlanning and frameNum<=int(frame_num):
            tempFrameR = (rtarHandFrame-resetHandFrameR)*frameNum/frame_num + resetHandFrameR
            tempFrameL = (ltarHandFrame-resetHandFrameL)*frameNum/frame_num + resetHandFrameL
            tempFrame = (HandPos-startHandFrame)*frameNum/frame_num + startHandFrame
            for j in range(7):
                jointStep[7+j] = tempFrameL[j]
                jointStep[14+j] = tempFrameR[j]
                if j < 5:
                    jointStep[26+j] = tempFrame[j]
                    jointStep[21+j] = tempFrame[j]
            self.realTimeAnimPub(jointStep,True)
            # if finger:
            #     print(jointStep[26:31])
            frameNum+=1 
        
    def HeadDown_button(self):
        '''
            按按钮前期准备动作
        '''
        frameNum = 0
        # self.Handmove2(rtarHandFrame=np.array([-5 ,-30  , 6.30253575 ,-80  , 4.35550809 ,-10.47408872 ,-13.84015213])*np.pi/180,
        #                ltarHandFrame=np.array([-5 ,30  , -6.30253575 ,-80  , -4.35550809 ,-10.47408872 ,13.84015213])*np.pi/180)
        # self.Handmove2(np.zeros(7),np.zeros(7))
        jointStep = np.zeros(31)
        
        # 低头
        # while not rospy.is_shutdown() and self.isStartPlanning and frameNum<=15:
        #     jointStep[5] = 0
        #     self.realTimeAnimPub(jointStep,True)
        #     frameNum+=1
        
        self.Handmove2(rtarHandFrame=np.array([-5 ,-15  , 6.30253575 ,-80  , -4.35550809 ,-10.47408872 ,-13.84015213])*np.pi/180,
                       ltarHandFrame=np.zeros(7)*np.pi/180, frame_num=50)
        
        self.Handmove2(rtarHandFrame=np.array([-45.199, -28.6990, -3.0186257362365723, -94.602,  -30, 3.064, -3.41])*np.pi/180,
                       ltarHandFrame=np.zeros(7), finger=True, frame_num=30)
        
    def HeadDown_cup(self):
        '''
        抓杯子前期准备动作
        '''
        jointStep = self.current_jointstate
        frameNum=0
        
        # 低头
        while not rospy.is_shutdown() and self.isStartPlanning and frameNum<=45:
            jointStep[5] = frameNum*0.61/45
            self.realTimeAnimPub(jointStep,True)
            frameNum+=1

        # 抬手
        self.Handmove2(rtarHandFrame=np.array([-5 ,-15  , 6.30253575 ,-80  , -4.35550809 ,-10.47408872 ,-13.84015213])*np.pi/180,
                       ltarHandFrame=np.array([-5 ,30  , -6.30253575 ,-80  , -4.35550809 ,-10.47408872 ,13.84015213])*np.pi/180, frame_num=45)
        # 把手移动到初始点
        # self.Handmove2(rtarHandFrame=np.array([-0.6, -0.21972, -0.22041, -1.581, 0.11196, 0.34668, -0.23287]),
        #                ltarHandFrame=np.array([-0.6, 0.21972, 0.22041, -1.581, -0.11196, 0.34668, 0.23287]), frame_num=45)
        self.Handmove2(
            rtarHandFrame=np.array([-25.70, -7.86, 10.22, -99.24, 2.44, 30.0, 15.17])*np.pi/180,
            ltarHandFrame=np.array([-25.70, 7.86, -10.22, -99.24, -2.44, 30.0, -15.17])*np.pi/180, frame_num=30)

    def joint_test(self):
        """ TODO if use left, change self.is_left """

        # right_angle = [-10, -10, 10, 10, 10, 10, 10]
        right_angle = np.zeros(31)
        right_angle[14:21] = [-15, -10, 15, 5, 15, 10, 15]
        # left_angle = [-10, 10, 10, 10, 10, 10, 10]
        # right_angle = [0, 0, 0, 0, 0, 0, 0]
        # left_angle = [0, 0, 0, 0, 0, 0, 0]
        logger.debug(f"control right_angle:{np.round(np.array(right_angle), 2)}")
        for i in range(10):
            self.realTimeAnimPub(right_angle,True)
            logger.debug(f"right_arm:{np.round(np.array(self.right_arm), 2)}")

        print('joint_angel has been sent!')
        # self.Handmove2(ltarHandFrame=np.array(left_angle)*np.pi/180,
        #                rtarHandFrame = np.zeros(7)
        #                ,finger=False, frame_num=90)
        logger.debug(f"control right_angle:{np.round(np.array(right_angle), 2)}")
        # logger.debug(f"control left_angle:{np.round(np.array(left_angle), 2)}")

        time.sleep(1)
        logger.debug(f"right_arm:{np.round(np.array(self.right_arm), 2)}")
        # logger.debug(f"left_arm:{np.round(np.array(self.left_arm), 2)}")
        print('joint test done!')


    def back_to_button_joint(self, is_left=False):
        print("back_to_button_joint!")
        if is_left:
            self.Handmove2(
                ltarHandFrame=np.array([-5 ,30  , -6.30253575 ,-80  , -4.35550809 ,-10.47408872 ,13.84015213])*np.pi/180,
                rtarHandFrame = np.zeros(7),
                finger=True, frame_num=90)
            self.Handmove2(
                ltarHandFrame=np.array([-5 ,30  , -6.30253575 ,-80  , -4.35550809 ,-10.47408872 ,13.84015213])*np.pi/180,
                rtarHandFrame = np.zeros(7),
                frame_num=60)
            self.Handmove2(np.zeros(7),np.zeros(7), openfinger=True, frame_num=60)
        else:
            self.Handmove2(
                rtarHandFrame=np.array([-45.199, -28.6990, -3.0186257362365723, -94.602,  -30, 3.064, -3.41])*np.pi/180,
                ltarHandFrame = np.zeros(7),
                finger=True, frame_num=90)
            self.Handmove2(
                rtarHandFrame=np.array([-5 ,-30  , 6.30253575 ,-80  , 4.35550809 ,-10.47408872 ,-13.84015213])*np.pi/180,
                ltarHandFrame = np.zeros(7),
                frame_num=60)
            self.Handmove2(np.zeros(7),np.zeros(7), openfinger=True, frame_num=60)

    def back_to_init_joint(self):
        self.Handmove2(rtarHandFrame=np.array([-45.199, -28.6990, -3.0186257362365723, -94.602,  -30, 3.064, -3.41])*np.pi/180,
                       ltarHandFrame=np.array([-45.199, 28.6990, 3.0186257362365723, -94.602,  30, 3.064, 3.41])*np.pi/180, frame_num=90)
        self.Handmove2(rtarHandFrame=np.array([-5 ,-30  , 6.30253575 ,-80  , 4.35550809 ,-10.47408872 ,-13.84015213])*np.pi/180,
                       ltarHandFrame=np.array([-5 ,30  , -6.30253575 ,-80  , -4.35550809 ,-10.47408872 ,13.84015213])*np.pi/180,frame_num=60)
        self.Handmove2(np.zeros(7),np.zeros(7), openfinger=True, frame_num=60)

    def back_to_init_joint_low(self):
        self.Handmove2(rtarHandFrame=np.array([-5 ,-30  , 6.30253575 ,-80  , 4.35550809 ,-10.47408872 ,-13.84015213])*np.pi/180,
                       ltarHandFrame=np.array([-5 ,30  , -6.30253575 ,-80  , -4.35550809 ,-10.47408872 ,13.84015213])*np.pi/180,frame_num=90)
        self.Handmove2(np.zeros(7),np.zeros(7), openfinger=True, frame_num=60)

    def play_data(self, phase):
        print('-'*10, phase, '-'*10)
        if phase == 'press_button_init':
            fileName = 'src/GraspNode/data/press_button_init.data'
            self.is_both = True
        elif phase == 'grasp_coffee_init':
            fileName = 'src/GraspNode/data/grasp_coffee_init.data'
        elif phase == 'grasp_coffee_init01':
            fileName = 'src/GraspNode/data/grasp_coffee_init01.data'
            self.is_both = True
        elif phase == 'putdown_cup':
            fileName = 'src/GraspNode/data/put_down_coffee.data'
            self.is_both = True
        elif phase == 'back_to_zero':
            fileName = 'src/GraspNode/data/back_to_init.data'
            self.is_both = True
        
        self.PlayTrajectory(fileName)

    def PlayTrajectory(self, data_path):
        print("is left", self.is_left, "is both", self.is_both)
        data = getRefMotionFromFile(data_path, self.is_left, self.is_both)
        print("data parsed!", data_path)

        is_right = False
        if ~self.is_left or self.is_both:
            is_right = True
        if self.is_both:
            is_left = True
        
        self.startTime = int(round(time.time()*1000))
        frameNum=0
        while not rospy.is_shutdown() and len(data)>0:
            jointmsg = RobotRTAnimation()
            jointmsg.request_type = 0
            jointmsg.time = int(round(time.time() * 1000)) - self.startTime
            jointmsg.id_list = self.idlist
            jointStep = self.poslist
            jointStep[5]=0
            if self.is_left:
                for j in range(7):
                    jointStep[7 + j] = data[0][j]
                for m in range(5):
                    jointStep[21+m] = data[0][7+m]  
            if is_right:
                for j in range(7):
                    jointStep[14 + j] = data[0][j]
                for m in range(5):
                    jointStep[26+m] = data[0][7+m]
            if np.linalg.norm(self.poslist - jointStep, ord=1) > 30.0:
                print("control step too large, ignore the action")
                continue
          
            jointmsg.pos_list = jointStep
            frameNum += 1
            # FIXME
            self.pub2.publish(jointmsg)
            data = np.delete(data,0,axis=0)
            self.rate.sleep()
        
        print("Play trajectory DONE!")
        return
    
    def FixedTrajectory(self):
        self.HeadDown_button()
        rospy.sleep(3)
        jointStep = self.poslist
        for t in [0, 12, 0]: # 重复轨迹的0, 12, 0三个step的位置
            for i in range(150):  # 每个动作连续发5s
                action1 = get_trajectory_joint(t)
                localaction=action1*np.pi/180
                localaction[-1]*=-1
                for j in range(7):
                    jointStep[14 + j] = localaction[j]
                logger.debug(f"control joint: {np.round(np.array(jointStep[14:21]) / np.pi * 180, 2)}")
                logger.debug(f"right_arm:{np.round(np.array(self.right_arm), 2)}")
                self.realTimeAnimPub(jointStep,True)
        while True:
            sum = 0
            self.realTimeAnimPub(jointStep,True) # 持续发送最后的动作
            for j in range(7):
                sum += np.linalg.norm(jointStep[j + 14] - self.right_arm[j])
            logger.debug(f"right_arm_wait:{np.round(np.array(self.right_arm), 2)}")
            if sum < 0.1:
                break
        print("Play trajectory done!")
        
    def KinematicInverse(self):
        targetPose = self.pose
        targetPose[2,3] = self.pose[2,3]+0.01
        targetJoint = None
        joint = self.current_jointstate[14:21]
        joint[2] *= -1
        joint[4] *= -1
        targetJoint = self.kdl_kin.inverse(targetPose,joint)
        print(targetJoint)
        
        jointStep = self.current_jointstate
        for j in range(7):
            if j in [6]:
                jointStep[14+j] = -targetJoint[j]
            else:
                jointStep[14+j] = targetJoint[j]
        self.realTimeAnimPub(jointStep,True)
            
        if targetJoint is not None:
            currJoint = self.current_jointstate[14:21]
            for i in range(0,30):
                self.frame[i] = (targetJoint-currJoint)*i/30+currJoint
            # print(self.frame)
            while not rospy.is_shutdown() and len(self.frame)>0:
                jointmsgtemp = JointState()
                for i in range(0,len(self.jointstatemsg.position)):
                    jointmsgtemp.name.append(self.jointstatemsg.name[i])
                    jointmsgtemp.position.append(self.jointstatemsg.position[i])
                jointmsgtemp.header.seq = self.jointstatemsg.header.seq+1
                jointmsgtemp.header.stamp. secs=  rospy.get_rostime().secs
                jointmsgtemp.header.stamp. nsecs=  rospy.get_rostime().nsecs
                for i in range(0,7):
                    jointmsgtemp.position[i+22]=self.frame[0][i]
                self.pub1.publish(jointmsgtemp)
                self.frame = np.delete(self.frame,0,axis=0)
                self.rate.sleep()
            self.frame=np.zeros((30,7))
            print("inverse success")
        else:
            print("inverse failed")

    # 底盘移动
    def chassis_done_callback(self,state,res):
        rospy.loginfo(state)
        rospy.loginfo(res)
        # self.small_move_service = rospy.ServiceProxy('/ginger/chassis_teleop/small_range_move', SetBool)
        # req = SetBoolRequest()
        # req.data = False
        # res = self.small_move_service(req)
        # logger.info(res)
    
    def chassis_active_callback(self):
        pass
        
    def chassis_move_feedback_callback(self,fb):
        pass
    
    def chassis_move(self,x,y,yaw,t):
        print("moving")
        self.small_move_service = rospy.ServiceProxy('/ginger/chassis_teleop/small_range_move', SetBool)
        req = SetBoolRequest()
        req.data = True
        res = self.small_move_service(req)
        # logger.info(res)

        client = actionlib.SimpleActionClient('/ginger/ChassisMove', ChassisMoveAction)
        client.wait_for_server()

        goal = ChassisMoveGoal()
        goal.tar_pos_x = x
        goal.tar_pos_y = y
        goal.tar_pos_yaw = yaw
        goal.tar_pos_t = t
        logger.info("send goal")
        client.send_goal(goal, self.chassis_done_callback, self.chassis_active_callback, self.chassis_move_feedback_callback)
        client.wait_for_result(timeout=rospy.Duration(1))

if __name__ == '__main__':
    rospy.init_node('RLPlanning', anonymous=True)
    rl = RLPlanning()


