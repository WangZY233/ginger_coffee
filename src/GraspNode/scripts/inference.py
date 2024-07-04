#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import time
import re
# import logging
from loguru import logger
from datetime import datetime

import torch
import numpy as np

# logging.basicConfig(level=logging.INFO, filename='rl.log', format='%(asctime)s - %(levelname)s: %(message)s')
logger.add(f'log/RL_{datetime.now().strftime("%m%d_%H%M")}.log', format="{time} {level} {message}", level="DEBUG")


# FIXME 选择使用的模型
# model_path = '/home/wangzy/lyf/ginger_ws/src/RLPlanningNode/models/model1225.pt'

model_path_cup = 'src/GraspNode/models/model.pt'
# model_path_cup = 'src/GraspNode/models/grasp_0130_model.pt'
# model_path_button = 'src/GraspNode/models/model_button_cp_0126-1.pt'
# model_path_button = 'src/GraspNode/models/model_0204.pt'
# model_path_button = 'src/GraspNode/models/model_0204_1.pt'
model_path_button = 'src/GraspNode/models/model_0220_c.pt'


# FIXME 调整控制幅度
# 控制幅度 = kp * control_range
kp = 0.001 # FIXME 0.01

model_cup = torch.jit.load(model_path_cup)
# model_cup.cuda()
model_cup.eval()

model_button = torch.jit.load(model_path_button)
# model_button.cuda()
model_button.eval()

control_joints = ["RShoulder_X", "RShoulder_Z", "RElbow_X", "RElbow_Y", "RWrist_X", "RWrist_Y", "RWrist_Z"]

control_range = {
    "RShoulder_X": [-130.0, 160.0], # 手臂前后
    "RShoulder_Z": [-130.0, 16.0], # 手臂向外张开,正表示向内
    "RElbow_X": [-100.0, 100.0],
    "RElbow_Y": [-114.0, 10.0],
    "RWrist_X": [-100.0, 100.0], #翻手腕
    "RWrist_Y": [-30.0, 30.0],
    "RWrist_Z": [-34.0, 34.0]
}


# trajectory
trajectory_path = "/home/cloud/Downloads/0222_real.txt"
trajectory_list = []
f = open(trajectory_path, "r")
for line in f.readlines():
    if "action:" not in line:
        continue
    line = line.strip().split('[')[1].split(']')[0]
    COMBINE_WHITESPACE = re.compile(r"\s+")
    line = COMBINE_WHITESPACE.sub(" ", line).strip()
    tra = [float(num) for num in line.split(' ')]
    trajectory_list.append(tra.copy())
f.close()
# print(len(trajectory_list))
# for trajectory in trajectory_list:
#     print(trajectory)

def pre_processor(box, joints_angle):
    joints_angle = np.array(joints_angle, dtype='float32')
    # logging.debug(f'input angle: {joints_angle}')
    # FIXME 真机需要归一化
    for i in range(len(control_joints)):
        rg = control_range[control_joints[i]]
        joints_angle[i] = joints_angle[i] / (rg[1] - rg[0])
    obs = torch.cat([torch.tensor(box.flatten()), torch.tensor(joints_angle)]).unsqueeze(0)
    return obs


def post_processor(joints_angle, raw_action, step,cup = False):
    # logging.debug(f'raw action: {raw_action}')
    action = None
    assert len(raw_action) == len(control_range)
    for i in range(len(control_range)):
        rg = control_range[control_joints[i]]
        # if i not in [1,3]:
        if cup == False:
            max_step = 50
        else:
            max_step = 20

        if i < 5:
            final_angle = joints_angle[i] + raw_action[i] * (rg[1] - rg[0]) * max(kp * (1 - step / max_step), 0.005) # FIXME .005
        else:
            final_angle = joints_angle[i] + raw_action[i] * (rg[1] - rg[0]) * max(0.02 * (1 - step / max_step), 0.01) # FIXME .005
        
        # FIXME
        # if i == 1:
        #     final_angle -= 0.8

        # if i == 2:
        #     final_angle += 1.5
        #     logger.debug(f"raw_action[i]: {raw_action[i]} \tfinal_angle: {final_angle} \tjoints_angle[i]:{joints_angle[i]} \t[step]:{raw_action[i] * (rg[1] - rg[0]) * max(kp * (1 - step / max_step), 0.005)}")

        # else:
        # final_angle = joints_angle[i] + raw_action[i] * (rg[1] - rg[0]) * max(kp * (1 - step / 30), 0.01)
        # final_angle = joints_angle[i] + raw_action[i]
        new_action = np.clip(final_angle, rg[0], rg[1])
        if action is None:
            action = new_action
        else:
            action = np.append(action, new_action)
    # logging.log(f'control action: {action}')
    # print("action", action)
    return action

def get_trajectory(button_location, hand_loc, right, step=0, is_left=False):
    logger.info(f"{'-'*10}{step}{'-'*10}")
    logger.debug(f"button_location:{button_location}, hand_loc:{hand_loc}, joint:{right}")
    action = np.array(trajectory_list[step])
    return action

def get_trajectory_joint(step=0, is_left=False):
    logger.info(f"{'-'*10}{step}{'-'*10}")
    action = np.array(trajectory_list[step])
    return action

def get_action(box, joints_angle, step=0, is_left=False,cup = False):
    """
    input:
    box: numpy.array, box信息
    joint_angle: numpy.array, 7个关节角度
    is_left: Boolean, use right hand or not
    return：
    action: numpy.array, 7个增加转动后的关节角度
    """
    # pre
    logger.info(f"{'-'*10}{step}{'-'*10}")
    if is_left:
        joints_angle = adjust_left(joints_angle)
        box = box.reshape((1,6))
        # box[:, :, :] = box[::-1, :, :]  # flip values of 0 axis
    logger.debug(f'obs_raw is:{box}, {joints_angle}')
    obs = pre_processor(box, joints_angle)
    logger.debug(f'obs is: {np.around(obs.tolist(), 2)}')
    

    # inference
    with torch.no_grad():
        if cup == False:
            logits = model_button({"obs": obs}, [obs], obs)[0]
        else:
            logits = model_cup({"obs": obs}, [obs], obs)[0]
        # print("obs",obs)
        raw_action, log_std = torch.chunk(logits, 2, dim=1)
        clipped_actions = np.clip(raw_action.cpu().numpy()[0], -1, 1)

    # FIXME post 真机需要
    # logger.debug(f'action raw is: {np.around(clipped_actions, 2)}')
    action = post_processor(joints_angle, clipped_actions, step,cup=cup)
    # logger.debug(f'action is: {np.around(action, 2)}')
    # action = clipped_actions
    if is_left:
        action = adjust_left(action)
    return action


def adjust_left(joints_angle):
    """
    adjust left arm value
    """
    joints_angle[1] *= -1
    joints_angle[2] *= -1
    joints_angle[4] *= -1
    joints_angle[6] *= -1
    return joints_angle


def test():
    # 30 * 30 * 30
    box = np.random.randint(-1, 1, 1*6)
    # 7 joints
    joints_angle = np.random.randint(-30, 30, 7)
    for i in range(10):
        start = time.time()
        action = get_action(box, joints_angle, i)
        # print(f"inference used time is : {time.time() - start}")
        # print(f"control action is {action}")


if __name__ == '__main__':
    # test()
    print('done')
