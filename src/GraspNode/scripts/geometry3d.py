#!/bin/python3
# -*- coding: UTF-8 -*-
import os
import sys
import open3d as o3d
import numpy as np
import cv2
import time
from collections import Counter
# import torch

def polygon2mask(polygon,W,H):
    '''

    Args:
        polygon: 多边形顶点 [n,2]
        W: 图像宽度
        H: 图像高度

    Returns:
        mask [H,W]
    '''
    mask = np.zeros((H,W), dtype=np.uint8)
    polygon = np.array(polygon).reshape(-1, 2).astype(np.int32)
    cv2.fillPoly(mask, [polygon], 1)
    return mask

# def points_filter(pts, eps=0.005,min_pts=20):
def points_filter(pts, eps=0.01,min_pts=20,cup = False):
    if pts.dimension() != 3:
        print('input shape error for points_filter, should be Nx3')
        return o3d.geometry.PointCloud()
    if len(pts.points) < min_pts:
        return o3d.geometry.PointCloud()
    index = pts.cluster_dbscan(eps, 3)
    index = np.array(index)
    counts=np.bincount(index[index>=0])
    inliers = o3d.geometry.PointCloud()
    if cup == False:
        max_cluster_label = np.argmax(counts)
        inliers += pts.select_by_index(np.flatnonzero(index == max_cluster_label))
    else :
        for j in range(len(counts)):
            if counts[j]>min_pts:
                inliers += pts.select_by_index(np.flatnonzero(index == j))
    return inliers

def compute_oriented_3d_box(depth,mask,K,grav,voxel_size=0.01):
    '''
        计算目标物的3d包围盒
        Args:
            depth: 原始深度图 [H,W]
            mask: 目标物的掩膜 [H,W]
            K: 相机内参 [3,3]
            grav: 重力方向 [3,]
            voxel_size: 下采样体素大小

        Returns:
            相机系下包围盒坐标 [8,3]
            顶点顺序如下
                   z
                 /
               /
             /
            o----------x
            |
            |      5-----4
            |     /|    /|
            |    3-|---6 |
            |    | |   | |
            |    | 2---|-7
            |    |/    |/
            |    0-----1
            y

    '''

    time_stat={}

    # depth2xyz
    t0=time.time()
    pts=mask2points(depth,K,mask)
    t1=time.time()
    time_stat["depth2points"] = (t1 - t0) * 1000
    if len(pts.points)==0:
        return

    # transform
    t0 = time.time()
    grav = grav / np.linalg.norm(grav)
    theta = np.arccos(np.inner(grav.squeeze(), np.array([0, 0, -1])))
    axis = np.cross(grav.squeeze(), np.array([0, 0, -1]))
    axis = axis / np.linalg.norm(axis)
    Rg = cv2.Rodrigues(axis * theta)[0]
    T=np.eye(4)
    T[:3,:3]=Rg
    # pts = pts.transform(T)
    t1 = time.time()
    time_stat["transform"] = (t1 - t0)*1000

    # cluster
    t0 = time.time()
    pts = pts.voxel_down_sample(voxel_size)
    pts_clustered = points_filter(pts, min_pts=int(len(pts.points)//10))
    num_points = len(pts_clustered.points)
    print("points num:",num_points)
    t1 = time.time()
    time_stat["cluster"] = (t1 - t0) * 1000

    # bbox
    t0=time.time()
    if num_points > 4:
        bounding_box=pts_clustered.get_oriented_bounding_box()
        box3d=np.array(bounding_box.get_box_points())
        # box3d=np.dot(box3d,Rg)
        # print(num_points)
    else:
        print("not enough points")
        box3d = None
    t1 = time.time()
    time_stat["bbox"] = (t1 - t0) * 1000
    return box3d

def compute_3d_box(depth,mask,K,grav,voxel_size=0.01,cup = False):
    '''
        计算目标物的3d包围盒
        Args:
            depth: 原始深度图 [H,W]
            mask: 目标物的掩膜 [H,W]
            K: 相机内参 [3,3]
            grav: 重力方向 [3,]
            voxel_size: 下采样体素大小

        Returns:
            相机系下包围盒坐标 [8,3]
            顶点顺序如下
                   z
                 /
               /
             /
            o----------x
            |
            |      5-----4
            |     /|    /|
            |    3-|---6 |
            |    | |   | |
            |    | 2---|-7
            |    |/    |/
            |    0-----1
            y

    '''

    time_stat={}

    # depth2xyz
    t0=time.time()
    pts=mask2points(depth,K,mask)
    t1=time.time()
    time_stat["depth2points"] = (t1 - t0) * 1000
    if len(pts.points)==0:
        return

    # transform
    t0 = time.time()
    grav = grav / np.linalg.norm(grav)
    theta = np.arccos(np.inner(grav.squeeze(), np.array([0, 0, -1])))
    axis = np.cross(grav.squeeze(), np.array([0, 0, -1]))
    axis = axis / np.linalg.norm(axis)
    Rg = cv2.Rodrigues(axis * theta)[0]
    T=np.eye(4)
    T[:3,:3]=Rg
    pts = pts.transform(T)
    t1 = time.time()
    time_stat["transform"] = (t1 - t0)*1000

    # cluster
    t0 = time.time()
    pts = pts.voxel_down_sample(voxel_size)
    pts_clustered = points_filter(pts, min_pts=int(len(pts.points)//10),cup=cup)
    t1 = time.time()
    time_stat["cluster"] = (t1 - t0) * 1000

    # bbox
    t0=time.time()
    bounding_box=pts_clustered.get_axis_aligned_bounding_box()
    box3d=np.array(bounding_box.get_box_points())
    box3d=np.dot(box3d,Rg)
    t1 = time.time()
    time_stat["bbox"] = (t1 - t0) * 1000
    return box3d

def draw_dotted_line(rgb, uv1, uv2, color=(255, 0, 0), thickness=2):
    N = np.linalg.norm(uv2 - uv1)
    uv = np.linspace(uv1, uv2, int(N / 10))
    for i in range(len(uv)):
        cv2.circle(rgb, tuple((int(uv[i, 0]), int(uv[i, 1]))),
                   1, color=color, thickness=-1, lineType=cv2.LINE_AA)

    return uv

def draw_box3d_sheltered(rgb, uv, depth, color=(255, 0, 0)):
    rgb=rgb.copy()
    max_depth = depth.max()
    ids = np.argmax(depth)
    uv = uv.astype(int)

    connect = [(0, 1), (1, 7), (7, 2), (2, 0),
               (3, 6), (6, 4),(4, 5), (5, 3),
               (0, 3), (1, 6), (7, 4), (2, 5)]

    for i in range(12):
        if ids in connect[i]:
            draw_dotted_line(rgb, uv[connect[i][0]], uv[connect[i][1]], color)
        else:
            # print(tuple(uv[connect[i][0]]))
            cv2.line(rgb, tuple(uv[connect[i][0]]), tuple(
                uv[connect[i][1]]), color, thickness=2)
    cv2.line(rgb, tuple((uv[0, 0], uv[0, 1])), tuple(
        (uv[7, 0], uv[7, 1])), (255, 0, 0), thickness=2)
    cv2.line(rgb, tuple((uv[1, 0], uv[1, 1])), tuple(
        (uv[2, 0], uv[2, 1])), (0, 255, 0), thickness=2) 
    return rgb

def vis_box3d(rgb,depth,K,box3d,color=(0,0,255)):
    '''
    将3d包围盒投影到图像上
    Args:
        rgb: [H,W,3]
        depth: [H,W]
        K: [3,3]
        box3d: [8,3]
        color: [3,] 0~255

    Returns: [H,W,3]
    '''
    uv, uv_depth = project_upright_camera_to_image(box3d, K)
    rgb=draw_box3d_sheltered(rgb, uv, depth, color=color)
    return rgb

def project_upright_camera_to_image(pc, K):
    '''
    将相机系下的三维坐标投影到图像坐标系中
    :param pc: point cloud with size N*3
    :param K: camera intrinsics
    :return: uv,depth
    '''
    uv = np.dot(pc, np.transpose(K))  # (n,3)
    # print("uv[:, 2]:",uv[:, 2])
    if uv[:, 2].all() != 0:
        uv[:, 0] /= uv[:, 2]
        uv[:, 1] /= uv[:, 2]
    else:
        print("----------skip----------")
    return uv[:, 0:2], pc[:, 2]

def mask2points(depth,K,mask):
    '''
    通过分割的mask和深度图，得到目标物的点云
    Args:
        depth: 深度图 [H,W]
        K: 相机内参 [3,3]
        mask: 目标掩膜 [H,W]

    Returns:
        pts :open3d.geometry.PointCloud

    '''
    depth2 = depth.copy()
    
    depth2[depth2 > 1200] = 0
    depth2[mask == 0] = 0
    # 转换相机内参为相机内参矩阵
    # K_matrix = np.array(K)
    # K_o3d = o3d.camera.PinholeCameraIntrinsic()
    # K_o3d.intrinsic_matrix = K_matrix
    
    
    # 提取相机内参的具体数值
    fx = K[0,0]
    # print("fx:",fx)
    fy = K[1,1]
    cx = K[0,2]
    cy = K[1,2]
    K_o3d = o3d.camera.PinholeCameraIntrinsic(depth.shape[1],depth.shape[0],fx,fy,cx,cy)

    # K_o3d = o3d.camera.PinholeCameraIntrinsic(depth.shape[1], depth.shape[0], K)
    depth2 = o3d.geometry.Image(depth2)
    pts = o3d.geometry.PointCloud.create_from_depth_image(depth2, K_o3d)
    return pts


def mask2voxel(depth,K,mask,voxel_size=0.01):
    '''
    通过分割的mask和深度图，得到目标物的点云,并且下采样至指定分辨率
    Args:
        depth: 深度图 [H,W]
        K: 相机内参 [3,3]
        mask: 目标掩膜 [H,W]
        voxel_size: 下采样体素大小

    Returns:
        pts :open3d.geometry.PointCloud

    '''
    pts = mask2points(depth, K, mask)
    pts=pts.voxel_down_sample(voxel_size)
    pts_clustered = points_filter(pts, min_pts=int(len(pts.points)//10))

    return pts_clustered

def depth2xyz(u, v, depthValue, intrinsic_matrix):
    fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
    cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]

    depth = depthValue * 0.001  # 将深度单位从毫米转换为米

    z = float(depth)
    x = float((u - cx) * z) / fx
    y = float((v - cy) * z) / fy

    result = [x, y, z]
    return result