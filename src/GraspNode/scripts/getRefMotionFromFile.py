#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np


def getRefMotionFromFile(fileName, is_left=False, is_both=False):
    f = open(fileName)
    line = f.readline()
    joint = np.zeros(1)

    is_right = False
    if ~is_left or is_both:
        is_right = True
    if is_both:
        is_left = True
    print("is_right", is_right, 'is_left', is_left)

    while line:
        frame = line.split(',')
        if is_left:
            for i in range(7,14):
                angle = float(frame[i])
                joint = np.append(joint, angle)
            for i in range(21,26):
                joint = np.append(joint,float(frame[i]))
                
        if is_right:
            for i in range(14,21):
                angle = float(frame[i])
                joint = np.append(joint,angle)
            for i in range(26,31):
                joint = np.append(joint,float(frame[i]))

        for i in range(4,7):
            joint = np.append(joint,float(frame[i]))  # head
        line = f.readline()
    f.close()
    
    joint = np.delete(joint,0)
    if is_both:
        data = joint.reshape(int(len(joint)/27),27)
    else:
        data = joint.reshape(int(len(joint)/15),15)

    return data


if __name__ == '__main__':
    data = np.zeros(7)
    fileName = 'src/RLPlanningNode/scripts/resource/disong_1m_right.data'
    getRefMotionFromFile(fileName,False)