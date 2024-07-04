#!/bin/python3
# -*- coding: UTF-8 -*-
import numpy as np
import queue
import threading
import json
import cv2
from loguru import logger
import rospy
from std_srvs.srv import Trigger
from cloud_robot_msgs.msg import ImageList
import os
from ultralytics import YOLO

class AutoQueue(queue.Queue):
    def __init__(self, maxsize=0):
        self.lock = threading.Lock()
        self.maxsize=maxsize+1
        super().__init__(maxsize+1)

    def put(self, item, block=False, timeout=None):
        try:
            self.lock.acquire()
            super().put(item, block, timeout)
            if super().full():
                super().get() 
            self.lock.release()
        except KeyboardInterrupt:
            self.lock.release()
            raise KeyboardInterrupt


    def get(self, block=False, timeout=None):
        try:
            self.lock.acquire()
            if self.empty():
                self.lock.release()
                return None
            item=super().get(block=block, timeout=timeout)
            self.lock.release()
            return item
        except KeyboardInterrupt:
            self.lock.release()
            raise KeyboardInterrupt

class Demo():
    def __init__(self):
        super().__init__()
        self.data_queue = AutoQueue(1)
        self.stream_listener = rospy.Subscriber("/realsense2_cam/rgb_depth_stream", ImageList,self.stream_listener_callback,queue_size=1,buff_size=1024*1024)
        self.K_client = rospy.ServiceProxy("/iris_camera/camera_parameters", Trigger)
        self.open_camera_client = rospy.ServiceProxy("/iris_camera/resume", Trigger)
        self.close_camera_client = rospy.ServiceProxy("/iris_camera/pause", Trigger)
        # self.detect_timer= rospy.Timer(rospy.Duration(0.01), self.detect)
        self.img_bgr = None
        self.open_camera()
        model = YOLO('/home/wangzy/push_button_ws/src/RLPlanningNode/models/best(1).pt')


    def open_camera(self):

        self.open_camera_client()
        response = self.K_client()
        intrinsics = json.loads(response.message)["rgb_camera"]["intrinsic"]
        self.K = np.array([[intrinsics["fx"], 0, intrinsics["cx"]],
                           [0, intrinsics["fy"], intrinsics["cy"]],
                           [0, 0, 1]])

    def close_camera(self):
        self.close_camera_client()

    def stream_listener_callback(self, msgs):
        self.data_queue.put(msgs)

    def detect(self,e):
        logger.info("detecting")
        msgs = self.data_queue.get()
        if msgs is None:
            logger.warning("no data")
            return None,None
        bgr = cv2.imdecode(np.frombuffer(msgs.data[:msgs.length[0]], dtype=np.uint8), cv2.IMREAD_COLOR)
        # 把bgr转成rgb
        self.img_bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        
        depth = cv2.imdecode(np.frombuffer(msgs.data[msgs.length[0]:msgs.length[0] + msgs.length[1]], dtype=np.uint8),
                             cv2.IMREAD_ANYDEPTH)
        accel = np.array(
            msgs.data[msgs.length[0] + msgs.length[1]:msgs.length[0] + msgs.length[1] + msgs.length[2]].decode().split(
                ",")).astype(np.float32)
        scale = 0.001

        # do something here
        logger.info("bgr : %s"%str(bgr.shape))
        logger.info("depth : %s"%str(depth.shape))
        logger.info("accel : %s"%str(accel))
        logger.info("depth_scale : %.4f"%scale)
        # cv2.imshow("image",self.img_bgr)
        path = '/home/wangzy/Workspace/ginger/ginger_ws/src/photos'
        # if cv2.waitKey(1) & 0xFF == ord('s'): # 按s键保存
        #     cv2.imwrite(path+'.jpg',self.img_bgr)            
        
        #i=0
        # cv2.imwrite(str(i)+'.jpg',self.img_bgr)
        # i=i+1
        # print("save image")
        # result = model.predict(source = self.img_bgr)
        # annotated_frame = result[0].plot()
        # cv2.imshow('img',annotated_frame)
        return self.img_bgr,depth



if __name__ == '__main__':
    rospy.init_node('cv_demo')
    # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    node=Demo()
    i = 0
    model = YOLO('/home/wangzy/push_button_ws/src/RLPlanningNode/models/best(1).pt')
    while not rospy.is_shutdown():
        img,depth= node.detect(1)
        if (img is None) :
            continue
        rospy.sleep(0.01)
        result = model.predict(source = img)
        annotated_frame = result[0].plot()
        cv2.imshow(winname = 'YOLO',mat = annotated_frame)
        
        # model.predict(source=img,show=True)
        key = cv2.waitKey(1)
        if key == ord('s'):
            path = '/home/wangzy/lyf/ginger_ws/src/photos'
            cv2.imwrite(path+str(i)+'.jpg',img)
            cv2.imwrite(path+str(i)+'.png',depth)
            print("save image")
            i = i+1
        if key == ord('q'):
            break
        
        
        
        output_video = '/home/wangzy/lyf/ginger_ws/src/photos/output_depth.mp4' 
        output1 = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'),30,(640,480))
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # output2 = cv2.VideoWriter(output_video, fourcc, 30.0, (640, 480))
        # if depth.dtype != 'uint8':
            # depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
        # if output2 is None:
            # print("无法创建VideoWriter对象")
        while not rospy.is_shutdown():
            img,depth= node.detect(1)
            output1.write(img)
            # output2.write(depth)
            key2 = cv2.waitKey(1)
            if key == 27:
                break
    #     if img is None:
    #         continue
    #     cv2.imshow("image",img)
    # cv2.waitKey(1)
    
    # img_bgr = node.img_bgr
    # i = 0
    # while(1):
    #     cv2.imwrite(str(i)+'.jpg',img_bgr)
    #     i=i+1
    #     print("save image")
    
    rospy.spin()
    rospy.signal_shutdown("exit")