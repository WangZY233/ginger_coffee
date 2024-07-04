import cv2
from PIL import Image, ImageDraw
import numpy as np
        # 载入一个模型
import matplotlib.pyplot as plt
import pupil_apriltags as apriltag


intrinsic_file = 'camera.txt'
K = np.loadtxt(intrinsic_file)

img_d = cv2.imread("20231204-162554.png", -1)  #深度图
rgb_image = cv2.imread("tag36h11_31.png")

def get_n_channel(img):
    if img.ndim == 2:
        print("通道数：1")
        return 1
    else:
        print("图像包含多个通道")
        return img.shape[2]

def position(img):
    """

    :rtype: object
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detector = apriltag.Detector()
    result = detector.detect(gray)
    return result

def depth2xyz(u, v, depthValue, intrinsic_matrix):
    fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
    cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]

    depth = depthValue * 0.001  # 将深度单位从毫米转换为米

    z = float(depth)
    x = float((u - cx) * z) / fx
    y = float((v - cy) * z) / fy

    result = [x, y, z]
    return result


if get_n_channel(img_d) == 1:
    depth = img_d
else:
    depth = cv2.split(img_d)[0]


center_point_xyz = []
points = position(rgb_image)
print(points[0].center)

center = points[0].center

center_x, center_y = center[0], center[1]
print(center_x)
arr = np.array(depth)

depthValue =float(arr[int(center_y), int(center_x)])
coordinate = depth2xyz(center_x, center_y, depthValue,K)
center_point_xyz.append(coordinate)
print(center_point_xyz)


# 投影三维坐标点到图像平面并在 RGB 图像上标记
def plot_3d_points_on_rgb(image, points_3d, intrinsic_matrix, save_path=None):
    fig, ax = plt.subplots(1)
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    intrinsic_inv = np.linalg.inv(intrinsic_matrix)
    points_2d = np.dot(points_3d, intrinsic_inv.T)
    points_2d /= points_2d[:, 2].reshape(-1, 1)

    ax.scatter(points_2d[:, 0], points_2d[:, 1], s=20, c='r', marker='o')

    if save_path:
        plt.savefig(save_path)  # 保存图像
    else:
        plt.show()

# 将三维坐标点投影到 RGB 图像平面并在图像上标记
plot_3d_points_on_rgb(rgb_image, np.array(center_point_xyz), K, save_path="result.jpg")
