#!/bin/python3
# -*- coding: UTF-8 -*-
import os
os.environ['CUDA_VISIBLE_DEVICES'] ='0'
# os.environ['CUDA_LAUNCH_BLOCKING'] ='1'

import os.path
from datetime import datetime
import autoqueue
from loguru import logger
from std_srvs.srv import Trigger
import tf
from PIL import Image
from ultralytics import YOLO
from geometry3d import *
import cv2
import rospy

from cloud_robot_msgs.msg import AnimationRequestGoal ,AnimationRequestAction,ImageList
from std_msgs.msg import Float32MultiArray , Bool

logger.add(f'log/cup_{datetime.now().strftime("%m%d_%H%M")}.log', format="{time} {level} {message}", level="DEBUG")

data_queue = autoqueue.AutoQueue(1)
box3d = []
labels = ["redbutton","Latte","milk","Espresso","Long coffee","Hot water","Cappuccino","button"]

# 加载模型
K_button = np.loadtxt("src/GraspNode/config/cup.K")  #相机内参
grav_button = np.loadtxt("src/GraspNode/config/cup.grav")  #重力方向
model_button = YOLO('src/GraspNode/models/yolo/best_button_0119.pt')  # 载入自定义模型

K_hand = np.loadtxt("src/GraspNode/config/cup.K")
grav_hand = np.loadtxt("src/GraspNode/config/hand.grav")        
model_hand = YOLO('src/GraspNode/models/yolo/best_index_0119.pt')  #加载模型

K_cup = np.loadtxt("src/GraspNode/config/cup.K")  #相机内参
grav_cup = np.loadtxt("src/GraspNode/config/cup.grav")  #重力方向
model_cup = YOLO('src/GraspNode/models/yolo/yolov8m-seg2.pt')  # 载入自定义模型

K_thumb = np.loadtxt("src/GraspNode/config/cup.K")
# grav_thumb = np.loadtxt("/home/wangzy/Workspace/ginger/ginger_cv/example/hand.grav")        
model_thumb = YOLO('src/GraspNode/models/yolo/best_thumb.pt')  #加载模型

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

def point_select(box,direction = 'z'):
    '''
    冒泡排序，从小到大排列
    '''
    if direction == 'z':
        for i in range(7):
            for j in range(0,7-i):
                if -box[j][1] > -box[j+1][1]:
                    box[j],box[j+1] = box[j+1],box[j]
    elif direction == 'x':
        for i in range(7):
            for j in range(0,7-i):
                if box[j][1] > box[j+1][1]:
                    box[j],box[j+1] = box[j+1],box[j] 
    return box

def find_centroid(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
    if contours:
        # 计算重心
        M = cv2.moments(contours[0])
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            return cx, cy
    return None, None

def identify_right_hand_mask(mask1, mask2):
    cx1, _ = find_centroid(mask1)
    cx2, _ = find_centroid(mask2)

    # 假设重心更靠右的是右手
    if cx1 is not None and cx2 is not None:
        if cx1 > cx2:# 大于为右手，小于为左手
            return mask1
        else:
            return mask2
    else:
        logger.error("无法确定重心，无法判断哪个是右手")
        return None

def get_button_box(rgb,depth):
    '''
    使用yolo获取按钮的位置
    '''
    # 对一张图像进行预测
    results = model_button.predict(source = rgb,conf = 0.5)
    annotated_frame = results[0].plot()
    cv2.imshow(winname = 'button',mat = annotated_frame)
    key = cv2.waitKey(1)
    center_points = []
    for r in results:
        xywh = r.boxes.xywh.cpu().numpy() 
        class_idx = r.boxes.cls.cpu().numpy() 
        for i in range(len(class_idx)):
            idx = np.where(class_idx==int(class_idx[i]))[0][0]
            xywh_i = xywh[idx]
            center_x = xywh_i[0]
            center_y = xywh_i[1]
            center_points.append((center_x,center_y))

    center_point_xyz = []
    for point in center_points:
        center_x, center_y = point[0], point[1]
        arr = np.array(depth) 
        # depthValue =float(arr[int(center_y), int(center_x)])
        depthValue = arr[np.round(center_y).astype(int), np.round(center_x).astype(int)]
        coordinate = depth2xyz(center_x, center_y, depthValue,K_button)
        center_point_xyz.append(coordinate)

    return class_idx,center_point_xyz

def get_hand_box(rgb,depth):
    '''
    使用yolo获取机器人食指的包围盒
    '''
    results = model_hand(rgb)
    for r in results:
        if r.masks:
            masks_data = r.masks.data
            for index, mask in enumerate(masks_data):
                    hand_mask = mask.cpu().numpy().astype(np.uint8)
        else:
            cv2.imshow("index box",rgb)
            cv2.waitKey(1)
            return np.zeros([1,8]),False

    # 显示实例分割结果
    contours, _ = cv2.findContours(hand_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    res1=cv2.drawContours(rgb.copy(), contours=contours, contourIdx=-1, color=(64,224, 208), thickness=5)       
    # cv2.imshow("index mask",res1)       
    
    # 保存点云   
    # pts=mask2voxel(depth,K_hand,hand_mask,voxel_size=0.001)
    # o3d.io.write_point_cloud("/home/wangzy/lyf/ginger_ws/output/vis_hand.ply", pts)
    
    # 显示包围盒
    box3d = compute_oriented_3d_box(depth, hand_mask, K_hand, grav_hand, voxel_size=0.001)
    vis_rgb = vis_box3d(rgb, depth, K_hand, box3d, color=(0, 255, 0))
    cv2.imshow("index box",vis_rgb)
    cv2.waitKey(1)
    return box3d,True

def get_cup_box(rgb,depth):
    '''
    使用yolo获取杯子的位置
    '''
    results = model_cup(rgb)  # 对一张图像进行预测
    class_index = 41
    cup_mask = []
    have_cup = False
    for r in results:
        if r.masks:
            masks_data,boxes_cls = r.masks.data,r.boxes.cls
            for index, (mask, cls) in enumerate(zip(masks_data, boxes_cls)):
                if cls == class_index:
                    cup_mask = mask.cpu().numpy()
                    have_cup = True
        else:
            cv2.imshow("cup",rgb)
            cv2.waitKey(1)
            return np.zeros([1,8]),False
    if have_cup == False:
        return np.zeros([1,8]),False
    if len(cup_mask) != 0:
        cv2.imwrite("/home/wangzy/Workspace/ginger/ginger_cv/cv_demo_code/scripts/output/yolo_cup_mask.png" , cup_mask)
        mask = cv2.imread("/home/wangzy/Workspace/ginger/ginger_cv/cv_demo_code/scripts/output/yolo_cup_mask.png", -1)  #目标掩膜
    
    # 计算包围盒 
    box3d = compute_3d_box(depth, cup_mask, K_cup, grav_cup, voxel_size=0.01,cup=True)
    vis_rgb = vis_box3d(rgb, depth, K_cup, box3d, color=(0, 255, 0))
    cv2.imshow("cup",vis_rgb)
    cv2.waitKey(1)
    return box3d,True

def get_thumb_box(rgb,depth):
    '''
    使用yolo获取机器人大拇指的包围盒
    '''
    results = model_thumb(rgb)
    for r in results:
        if r.masks:
            masks_data = r.masks.data
            for index, mask in enumerate(masks_data):
                if len(masks_data)==2:
                    if index == 0:
                        hand_mask1 = mask.cpu().numpy().astype(np.uint8) 
                    elif index == 1:
                        hand_mask2 = mask.cpu().numpy().astype(np.uint8)
                        hand_mask = identify_right_hand_mask(hand_mask1,hand_mask2)
                else:
                    hand_mask = mask.cpu().numpy().astype(np.uint8)
        else:
            cv2.imshow("thumb box",rgb)
            cv2.waitKey(1)
            return np.zeros([1,8]),False

    # 显示实例分割结果
    contours, _ = cv2.findContours(hand_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    res1=cv2.drawContours(rgb.copy(), contours=contours, contourIdx=-1, color=(64,224, 208), thickness=5)       
    # cv2.imshow("thumb mask",res1)       
    
    # 显示包围盒
    box3d = compute_oriented_3d_box(depth, hand_mask, K_hand, grav_hand, voxel_size=0.001)
    vis_rgb = vis_box3d(rgb, depth, K_hand, box3d, color=(0, 255, 0))
    cv2.imshow("thumb box",vis_rgb)
    cv2.waitKey(1)
    return box3d,True

global stop_num
stop_num = 0
global stop_alert
stop_alert = False
global box_diag_robot
box_diag_robot = [0,0,0]

def stream_listener_callback(msgs):
    data_queue.put(msgs)
    msgs = data_queue.get()
    
    global box_diag_robot
    global stop_num
    global stop_alert
    stop_alert = False  # FIXME
    
    if msgs is None:
        logger.warning("no data")
        return None,None
    bgr = cv2.imdecode(np.frombuffer(msgs.data[:msgs.length[0]], dtype=np.uint8), cv2.IMREAD_COLOR)
    depth = cv2.imdecode(np.frombuffer(msgs.data[msgs.length[0]:msgs.length[0] + msgs.length[1]], dtype=np.uint8),
                            cv2.IMREAD_ANYDEPTH)
    
    if stop_alert == False:
        '''
            按按钮部分cv识别，识别按钮和大拇指坐标
        '''
        # 获取tf转换关系
        try:
            (trans,rot) = listener.lookupTransform('lumbar_roll_link', '/bowknot_camera_color_optical_frame', rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                pass
        R=quat2matrix([rot[0],rot[1],rot[2],rot[3]])
        ### 获取按钮包围盒
        button_coord = [0,0,0]
        button_classes,button_point = get_button_box(bgr,depth=depth) 
        # FIXME print button location
        for i in range(len(button_classes)):
            logger.debug(f"[{button_classes[i]}] {np.around(np.array(button_point[i]) * 100, 4)}")
            
        for i in range(len(button_classes)):
            if button_classes[i] == 6:
                coordinate = button_point[i]
                coordinate = np.array(coordinate)
                coordinate *= 100
                # FIXME choose right buttom !!!!!!!!!!!
                if coordinate[0] < 0:
                    continue
                logger.debug(f"latte origin camera: {coordinate}")
                # if coordinate[1] < 0:
                #     coordinate[1] += 3.5
                # logger.debug(f"latte fixed camera: {coordinate}")
                coordinate = (np.dot(coordinate,R.T)+np.array(trans)*100).tolist()
                # logger.debug("latte_coordinate",coordinate)
                latte_loc = coordinate
            if button_classes[i] == 0:
                stop_num += 1
        
        if len(button_classes) >= 3:
            stop_num = 0
        if stop_num >= 3:
            logger.info(f"stop_num: {stop_num}")
            stop_alert = True
            logger.info("按钮识别完成")
            cv2.destroyAllWindows()
                
        ### 获取食指包围盒
        box3d_hand,success_index = get_hand_box(bgr,depth=depth)
        handPos = [0,0,0]
        if success_index:
            # 摄像头坐标转换为机器人坐标
            box3d_hand_cam = box3d_hand*100
            hand_pos_cam = np.mean(box3d_hand_cam, axis=0)
            logger.debug(f"index_loc_cam: {hand_pos_cam}")
            hand_diag_robot = (np.dot(box3d_hand_cam,R.T)+np.array(trans)*100)
            # logger.debug("hand_diag_robot", len(hand_diag_robot), hand_diag_robot)
            handPos = ((hand_diag_robot[0]+hand_diag_robot[1]+hand_diag_robot[2]+hand_diag_robot[3]+hand_diag_robot[4]+hand_diag_robot[5]+hand_diag_robot[6]+hand_diag_robot[7])/8).tolist()
            logger.debug(f"index_loc_trans: {handPos}")
            
        #按钮坐标，for debug
        logger.debug(f"latte_loc[1]: {latte_loc[1]}")
        # FIXME
        # button_location = [latte_loc[0],latte_loc[2], abs(latte_loc[1])]
        # index_loc = [handPos[0],handPos[2]+0.5,-handPos[1]+1]

        button_location = [latte_loc[0],latte_loc[2], abs(latte_loc[1])+7]
        # button_location = [latte_loc[0],latte_loc[2], 50]
        # button_location = [latte_loc[0],45, 50]
        # index_loc = [handPos[0],handPos[2]+0.5,-handPos[1]+1]
        index_loc = [handPos[0] + 1, handPos[2] - 1.3, -handPos[1]+7] # FIXME +1
        # index_loc = [handPos[0],handPos[2],-handPos[1]]
        logger.debug(f"button_location fixed: {button_location}")
        logger.debug(f"index_loc_fixed: {index_loc}")
        
        # 发送信息到话题
        pub_msg_loc = Float32MultiArray()
        pub_msg_loc.data = button_location
        loc_publisher.publish(pub_msg_loc)
        
        pub_msg_loc = Float32MultiArray()
        pub_msg_loc.data = index_loc
        index_loc_publisher.publish(pub_msg_loc)
            
        
        ### 用于判断停止条件
        pub_msg_stop = Bool()
        pub_msg_stop.data = stop_alert
        stop_button_publisher.publish(pub_msg_stop)
    
    else:
        '''
            抓杯子部分cv识别，识别按钮和大拇指坐标
        '''
        # 获取tf转换关系
        try:
            (trans,rot) = listener.lookupTransform('lumbar_roll_link', '/bowknot_camera_color_optical_frame', rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                pass
        R=quat2matrix([rot[0],rot[1],rot[2],rot[3]])
        
        ### 获取杯子包围盒
        box3d_cup,success_cup = get_cup_box(bgr,depth=depth)
        if success_cup:
            # 摄像头坐标转换为机器人坐标
            box_diag_cam = (box3d_cup[0]+box3d_cup[7]+box3d_cup[1]+box3d_cup[2]+box3d_cup[3]+box3d_cup[4]+box3d_cup[5]+box3d_cup[6])*100/8
            box_diag_robot = (np.dot(box_diag_cam,R.T)+np.array(trans)*100).tolist()
            logger.debug(f"box_diag_robot: {box_diag_robot}")
            
            # 发送信息到话题
            pub_msg_cup = Float32MultiArray()
            pub_msg_cup.data = [box_diag_robot[0]-3,box_diag_robot[2]+1,36]
            logger.debug(f"box_diag_robot_fix: {pub_msg_cup.data}")
            cup_loc_publisher.publish(pub_msg_cup)

        ### 获取大拇指包围盒
        box3d_hand,success_trumb = get_thumb_box(bgr,depth=depth)
        if success_trumb:
            # 摄像头坐标转换为机器人坐标
            box3d_hand_cam = box3d_hand*100
            hand_diag_robot = (np.dot(box3d_hand_cam,R.T)+np.array(trans)*100)
            # 找到包围盒中x轴坐标最小的四个点，并计算手部坐标
            hand_diag_robot = point_select(hand_diag_robot,direction='x')
            handPos = ((hand_diag_robot[0]+hand_diag_robot[1]+hand_diag_robot[2]+hand_diag_robot[3])/4).tolist()
            # handPos = ((hand_diag_robot[0]+hand_diag_robot[1]+hand_diag_robot[2]+hand_diag_robot[3]+hand_diag_robot[4]+hand_diag_robot[5]+hand_diag_robot[6]+hand_diag_robot[7])/8).tolist()
            logger.debug(f"thumbPos : {handPos}")
            handPos = [handPos[0]-3.5,handPos[2]+1,-handPos[1]+3.5]
            logger.debug(f"thumbPos_fix : {handPos}")
            
            # 发送信息到话题
            pub_msg_hand = Float32MultiArray()
            pub_msg_hand.data = handPos
            hand_loc_publisher.publish(pub_msg_hand)
    
        ### 用于判断停止条件
        pub_msg_stop = Float32MultiArray()
        # cup_box = point_select(box3d_cup,direction='x')
        # cup_box_xmin = ((box3d_cup[0]+box3d_cup[5]+box3d_cup[2]+box3d_cup[3])/4).tolist()
        # cup_box_xmin = (np.dot(cup_box_xmin,R.T)+np.array(trans)*100).tolist()
        pub_msg_stop.data = [box_diag_robot[0]-3.5,box_diag_robot[2]]
        stop_cup_publisher.publish(pub_msg_stop)


if __name__=="__main__":
    rospy.init_node("cv_loc_publisher",anonymous=True)
    listener = tf.TransformListener()
    stream_listener = rospy.Subscriber("/realsense2_cam/rgb_depth_stream", ImageList,stream_listener_callback,queue_size=1,buff_size=1024*1024)
    
    # 按钮识别部分
    loc_publisher = rospy.Publisher("/cv/button_loc",Float32MultiArray,queue_size=10)
    index_loc_publisher = rospy.Publisher("/cv/index_loc",Float32MultiArray,queue_size=10)
    stop_button_publisher = rospy.Publisher("/cv/stop_button",Bool,queue_size=10)
    
    # 抓杯子部分
    cup_loc_publisher = rospy.Publisher("/cv/cup_loc",Float32MultiArray,queue_size=10)
    hand_loc_publisher = rospy.Publisher("/cv/hand_loc",Float32MultiArray,queue_size=10)
    stop_cup_publisher = rospy.Publisher("/cv/stop_cup",Float32MultiArray,queue_size=10)

    rospy.spin()
        

    