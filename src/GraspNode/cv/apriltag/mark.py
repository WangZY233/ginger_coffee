#!/usr/bin/env python
# coding: UTF-8
import pupil_apriltags as apriltag
import cv2

img = cv2.imread("tag36h11_31.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 创建一个apriltag检测器
detector = apriltag.Detector()

# 进行apriltag检测，得到检测到的apriltag的列表
tags = detector.detect(gray)  #图片必须转换成灰度图

print(tags)


for tag in tags:
    cv2.circle(img, tuple(tag.corners[0].astype(int)), 4, (255, 0, 255), 2)  # left-top
    cv2.circle(img, tuple(tag.corners[1].astype(int)), 4, (255, 0, 255), 2)  # right-top
    cv2.circle(img, tuple(tag.corners[2].astype(int)), 4, (255, 0, 255), 2)  # right-bottom
    cv2.circle(img, tuple(tag.corners[3].astype(int)), 4, (255, 0, 255), 2)  # left-bottom
    cv2.circle(img, tuple(tag.center.astype(int)), 4, (255, 0, 255), 2)  # center

cv2.imshow("out_image", img)
cv2.imwrite('image.png', img)
cv2.waitKey()
