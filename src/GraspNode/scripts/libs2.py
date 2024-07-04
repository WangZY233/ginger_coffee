from xpinyin import Pinyin
from scipy.spatial import ConvexHull
from sklearn.cluster import dbscan
from collections import Counter
import threading
import multiprocessing
from src.config import *
import sanic
import yaml
import numpy as np
import random
#import cupy as cp
import cv2
import torch
import time
import sys
import build._ext as _ext
import logging
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)

def read_data(msg,request):
    '''
    parse data from request
    :return:bgr,bgr_buf, depth,depth_buf, K, scale,truncation
    '''
    try:
        bgr_buf = request.files.get("imagefile").body
        bgr = cv2.imdecode(
            np.frombuffer(
                bgr_buf,
                dtype=np.uint8),
            cv2.IMREAD_COLOR)
    except BaseException:
        logging.error('---read_data---no bgr data')
        msg["code"] = "1"
        raise RuntimeError("---read_data---no bgr data received")

    try:
        depth_buf = request.files.get("depthfile").body
        depth = cv2.imdecode(
            np.frombuffer(
                depth_buf,
                np.uint8),
            cv2.IMREAD_ANYDEPTH)
    except BaseException:
        logging.error('---read_data---no depth data')
        msg["code"] = "1"
        raise RuntimeError("---read_data---no depth data received")

    try:
        param = request.form["param"][0]
        param = param.split(",")
        K = str2K(",".join(param[:4]))
    except BaseException:
        logging.error("---read_data---no intrinsics")
        msg["code"] = "1"
        raise RuntimeError("---read_data---no intrinsics received")

    try:
        scale = float(request.form["scale"][0])
    except BaseException:
        logging.error("---read_data---no depth scale data")
        msg["code"] = "1"
        raise RuntimeError("---read_data---no depth scale data received")

    try:
        grav = request.form["accel"][0]
        grav = -np.array(grav.split(","), dtype=np.float32)
    except BaseException:
        logging.error("---read_data---no grav")
        msg["code"] = "1"
        raise RuntimeError("---read_data---no grav data received")

    try:
        truncation = request.form.items()["truncation"]
        truncation = float(truncation)
    except BaseException:
        truncation = DEPTH_TRUNCATION
        logging.warning("---read_data---no depth truncation data,use default %.2f"%DEPTH_TRUNCATION)

    depth = depth * scale
    depth[depth > truncation] = 0
    depth = depth.astype(np.float32)
    points_total=depth2xyz(depth,K,retain_zeros=True)
    points=points_total[points_total[:,2]>0]
    logging.info("---read_data---all data parsed successfully")
    return bgr, bgr_buf, depth, depth_buf, K, scale, grav, truncation,points_total,points


def pca(x):
    '''
    Use pca to compute prime directions in 3D space
    :param x: points [n,3]
    :return: eigen value,sorted transformation matrix
    '''
    x = x.clone()
    x -= x.mean(0)
    corr = np.dot(x.T, x) / x.shape[0]
    v, d = np.linalg.eig(corr)
    if np.linalg.det(d) < 0:
        d[:, 0] *= -1
    index = v.argsort()[::-1]
    D = d.T
    D = D[index]
    D = D.T
    return v[index], D

def matrix2quat(M):
    w = (np.trace(M) + 1) ** 0.5 / 2
    x = (M[2, 1] - M[1, 2]) / 4 / w
    y = (M[0, 2] - M[2, 0]) / 4 / w
    z = (M[1, 0] - M[0, 1]) / 4 / w
    return [x, y, z, w]

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

def depth2xyz(depth, depth_cam_matrix, flatten=True,retain_zeros=False):
    """
    Convert depth image to points
    :param depth_map: depth image
    :param depth_cam_matrix: camera intrinsics
    :param flatten:points format,default=True
    :return: points of [nx3] if flatten=true else [h,w,3]
    """
    H,W=depth.shape
    depth_map=torch.from_numpy(depth)
    fx, fy = depth_cam_matrix[0, 0], depth_cam_matrix[1, 1]
    cx, cy = depth_cam_matrix[0, 2], depth_cam_matrix[1, 2]
    h, w =torch.meshgrid(torch.arange(H),torch.arange(W))
    z = depth_map
    if retain_zeros:
        x = ((w - cx) * z / fx)
        y = ((h - cy) * z / fy)
    else:
        index = z > 0
        x = ((w - cx) * z / fx)[index]
        y = ((h - cy) * z / fy)[index]
        z = z[index]
    xyz = torch.dstack((x, y, z)) if flatten == False else torch.dstack(
        (x, y, z)).view([-1, 3])
    return xyz

def depth2xyz_cuda(depth_map, depth_cam_matrix, flatten=True,retain_zeros=False):
    """
    Convert depth image to points
    :param depth_map: depth image
    :param depth_cam_matrix: camera intrinsics
    :param flatten:points format,default=True
    :return: points of [nx3] if flatten=true else [h,w,3]
    """
    fx, fy = depth_cam_matrix[0, 0], depth_cam_matrix[1, 1]
    cx, cy = depth_cam_matrix[0, 2], depth_cam_matrix[1, 2]
    h, w = cp.mgrid[0:depth_map.shape[0], 0:depth_map.shape[1]]
    z = depth_map
    if retain_zeros:
        x = ((w - cx) * z / fx)
        y = ((h - cy) * z / fy)
    else:
        index = z > 0
        x = ((w - cx) * z / fx)[index]
        y = ((h - cy) * z / fy)[index]
        z = z[index]
    xyz = cp.dstack((x, y, z)) if flatten == False else cp.dstack(
        (x, y, z)).reshape(-1, 3)
    return xyz.get()


def project_upright_camera_to_image(pc, K):
    '''
    Project camera coordinate to image coordinate
    :param pc: point cloud with size N*3
    :param K: camera intrinsics
    :return: uv,depth
    '''
    K=torch.tensor(K).float().to(TORCH_DEVICE)
    uv = torch.matmul(pc, K.T)  # (n,3)
    uv[:, 0] /= uv[:, 2]
    uv[:, 1] /= uv[:, 2]
    return uv[:, 0:2], pc[:, 2]


def str2K(param, scale=1):
    '''
    :param param:"fx,fy,cx,cy"
    :return: 3x3 array
    '''
    fx, fy, cx, cy = param.split(",")
    K = np.eye(3,dtype=np.float32)
    K[0, 0] = float(fx) // scale
    K[1, 1] = float(fy) // scale
    K[0, 2] = float(cx) // scale
    K[1, 2] = float(cy) // scale
    return K

def Rodrigues(r):
    t=torch.linalg.norm(r)
    n=r/t
    n1,n2,n3=n
    R=torch.cos(t).to(TORCH_DEVICE)*torch.eye(3).to(TORCH_DEVICE)+(1-torch.cos(t).to(TORCH_DEVICE))*torch.outer(n.T,n)+torch.sin(t).to(TORCH_DEVICE)*torch.tensor([[0,-n3,n2],[n3,0,-n1],[-n2,n1,0]]).to(TORCH_DEVICE)
    return R

def letter_box(img):
    if len(img.shape) == 2:
        h, w = img.shape
        s = max(h, w)
        d = s - min(h, w)
        if h > w:
            img = np.pad(img, ((0, 0), (d // 2, d // 2)),
                         mode="constant", constant_values=0)
        elif h < w:
            img = np.pad(img, ((d // 2, d // 2), (0, 0)),
                         mode="constant", constant_values=0)
    elif len(img.shape) == 3:
        h, w, c = img.shape
        s = max(h, w)
        d = s - min(h, w)
        if h > w:
            img = np.pad(img, ((0, 0), (d // 2, d // 2), (0, 0)),
                         mode="constant", constant_values=0)
        elif h < w:
            img = np.pad(img, ((d // 2, d // 2), (0, 0), (0, 0)),
                         mode="constant", constant_values=0)

    return img



def kalman_filter(s, x):
    Xt = np.concatenate([s["trajectory"][-1], s["velocity"]]).reshape([-1, 1])
    Zt = np.concatenate([x, np.zeros(3)]).reshape([-1, 1])
    Xt = np.dot(s["F"], Xt)
    Pt = np.dot(np.dot(s["F"], s["P"]), s["F"].T) + s["Q"]
    Kt = np.dot(
        np.dot(
            Pt, s["H"].T), np.linalg.inv(
            np.dot(
                np.dot(
                    s["H"], Pt), s["H"].T) + s["R"]))
    Xt = Xt + np.dot(Kt, (Zt - np.dot(s["H"], Xt)))
    Pt = np.dot(np.eye(len(x) * 2) - np.dot(Kt, s["H"]), Pt)
    s["P"] = Pt
    return Xt[:3].squeeze(), Xt[3:].squeeze()


def kalman_filter_grav(s, x):
    x = np.array(x)
    if len(s["trajectory"]) == 0:
        Xt = x
    else:
        Xt = s["trajectory"][-1].reshape([-1, 1])
        Zt = x.reshape([-1, 1])
        Xt = np.dot(s["F"], Xt)
        Pt = np.dot(np.dot(s["F"], s["P"]), s["F"].T) + s["Q"]
        Kt = np.dot(
            np.dot(
                Pt, s["H"].T), np.linalg.inv(
                np.dot(
                    np.dot(
                        s["H"], Pt), s["H"].T) + s["R"]))
        Xt = Xt + np.dot(Kt, (Zt - np.dot(s["H"], Xt)))
        Pt = np.dot(np.eye(len(x)) - np.dot(Kt, s["H"]), Pt)
        s["P"] = Pt
        Xt = Xt.squeeze()
    s["trajectory"].append(Xt)
    return Xt





def find_line(uv, angle_thres=10):
    ind = uv.argmin(0)[1]
    if ind < len(uv) - 1:
        p = ind + 1
    else:
        p = 0
    v = uv[p] - uv[ind]
    lines = [[[p, ind, v]]]
    while p != ind:
        if p == len(uv) - 1:
            v = uv[0] - uv[p]
            theta = np.arccos(np.dot(
                lines[-1][-1][2], v) / np.linalg.norm(lines[-1][-1][2]) / np.linalg.norm(v))
            if theta < angle_thres / RADIAN2DEGREE:
                lines[-1].append([0, p, v])
            else:
                lines.append([[0, p, v]])
            p = 0
        else:
            v = uv[p + 1] - uv[p]
            theta = np.arccos(np.dot(
                lines[-1][-1][2], v) / np.linalg.norm(lines[-1][-1][2]) / np.linalg.norm(v))
            if theta < angle_thres / RADIAN2DEGREE:
                lines[-1].append([p + 1, p, v])
            else:
                lines.append([[p + 1, p, v]])
            p = p + 1
    return lines



def project_point_to_line(point, pos, dire):
    x0, y0, z0 = np.array(pos).squeeze()
    m, n, p = np.array(dire).squeeze()
    x1, y1, z1 = np.array(point).squeeze()
    A = np.array([[n, -m, 0],
                  [p, 0, -m],
                  [m, n, p]])
    b = np.array([[n * x0 - m * y0],
                  [p * x0 - m * z0],
                  [m * x1 + n * y1 + p * z1]])
    res = np.dot(np.linalg.pinv(A), b).squeeze()
    return res


def fitting_line(vert, line):
    inds = np.unique(np.ravel([[x[0], x[1]] for x in line]))
    points = vert[inds]
    if len(points) == 2:
        return points
    v, D = pca(points)
    dire = D[:, 0]
    pos = points.mean(0)
    start_point = vert[line[0][1]]
    end_point = vert[line[-1][0]]
    p1 = project_point_to_line(start_point, pos, dire)
    p2 = project_point_to_line(end_point, pos, dire)
    return p1, p2



def get_points_index(W,H,x1,y1,x2=None,y2=None):
    assert 0<=x1<W
    assert 0<=y1<H
    if x2 is not None:
        assert 0 <= x2 <=W
    if y2 is not None:
        assert 0<=y2<=H
    if x2 is None:
        x2=x1+1
    if y2 is None :
        y2=y1+1
    rows = torch.arange(y1,y2).to(TORCH_DEVICE)
    cols = torch.arange(x1,x2).to(TORCH_DEVICE)
    index=(rows.reshape([-1,1])*W+cols.reshape([1,-1])).ravel()
    return index


def find_neibour_points(roi,points_total,W,H):
    """
    Find surrounding points of obj and an approximate xyz of the obj.
    :param depth: depth image [H,W]
    :param roi: [x1,y1,w,h]
    :param K: 3x3 np.ndarray
    :return: neibour_points, target_xyz
    """
    try:
        x1, y1, x2, y2 = roi
        w=x2-x1
        h=y2-y1
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        offset = min(w, h) // 4
        x3 = max(x1 - offset, 0)
        x4 = min(x2 + offset,W)
        y3 = max(y1 - offset, 0)
        y4 = min(y2 + offset, H)
        #index1 = get_points_index(W, H, x3, y3, x4, y1)
        index2 = get_points_index(W, H, x3, y1, x1, y2)
        index3 = get_points_index(W, H, x2, y1, x4, y2)
        index4 = get_points_index(W, H, x3, y2, x4, y4)

        index=torch.cat([index2,index3,index4]).to(TORCH_DEVICE)
        plane_points = points_total[index]
        plane_points = plane_points[plane_points[:, 2] > 0]
        #logging.info("---find_neibour_points---find_neibour_points done")

        return plane_points

    except BaseException:
        logging.error("---find_neibour_points---error in find_neibour_points")
        return [],[]


def RANSAC_plane_fitting(plane_points,grav,nums=RANSAC_POINTS, r=0.01, M=RANSAC_M):
    """
    Fitting the parametric equation of the plane
    :param plane_points: candidate points of shape [n,3]
    :param r: points within distance r is considered as inliers
    :param N: number of samples
    :return: parameter of the plane
    """
    try:
        if len(plane_points)>nums:
            index=torch.randperm(len(plane_points))[:nums].to(TORCH_DEVICE)
            plane_points = plane_points[index]
        x, y, z = plane_points.T
        maxnum = 0

        for k in range(10):
            # M samples
            for i in range(M):
                inds = torch.randperm(len(x))[:3].to(TORCH_DEVICE)
                x_ = x[inds]
                y_ = y[inds]
                z_ = z[inds]

                # solve equations
                A = torch.stack([x_, y_, z_, torch.ones_like(x_).to(TORCH_DEVICE)]).T
                v, d = torch.linalg.eigh(torch.matmul(A.T, A))
                solve = d[:, v.argmin()]
                # choose inliers
                B = torch.stack([x, y, z, torch.ones_like(x)]).T
                dist = torch.abs(torch.matmul(B, solve.reshape([4, 1]))) / (solve[0] ** 2 + solve[1] ** 2 + solve[2] ** 2)
                num = torch.sum(dist < r)
                inds = torch.where(dist < r)[0]
                if num > maxnum:
                    maxnum = num
                    bestinds = inds
                    bestsolve = solve

            bestsolve /=torch.linalg.norm(bestsolve[:3])
            inner = torch.abs(torch.inner(bestsolve[:3], torch.tensor(grav).float().to(TORCH_DEVICE)))
            angle = torch.arccos(inner) * RADIAN2DEGREE
            #logging.info('---RANSAC_plane_fitting---bestsolve:[%.6f,%.6f,%.6f], inner:%.6f, angle:%.2f'%(bestsolve[0],bestsolve[1],bestsolve[2], inner, angle))
            if angle < PLANE_GRAV_DEGREE_THRES * 1.2:
                break

        # refine with inliers
        x_ = x[bestinds]
        y_ = y[bestinds]
        z_ = z[bestinds]
        A = torch.stack([x_, y_, z_, torch.ones_like(x_).to(TORCH_DEVICE)]).T
        v, d = torch.linalg.eigh(torch.matmul(A.T, A))
        bestsolve = d[:, v.argmin()]
        norm = torch.linalg.norm(bestsolve[:3])
        bestsolve /= norm
        #logging.info("---RANSAC_plane_fitting---RANSAC_plane_fitting done")
        return bestsolve
    except BaseException:
        logging.error("---RANSAC_plane_fitting---Error in RANSAC_plane_fitting")
        return []


def choose_inliers(bestsolve, points_total,thres=0.005,nums=2048):
    try:
        t1=time.time()
        if len(points_total)>nums:
            index = torch.randperm(len(points_total))[:nums].to(TORCH_DEVICE)
            points = points_total[index]
        t2=time.time()
        x, y, z = points.T
        B = torch.stack([x, y, z, torch.ones_like(x).to(TORCH_DEVICE)]).T
        dist = torch.abs(torch.matmul(B, bestsolve.reshape([4, 1]))) / (bestsolve[0] ** 2 + bestsolve[1] ** 2 + bestsolve[2] ** 2)
        dist = dist.squeeze()
        #np.savetxt("points.txt",points)
        points = points[dist <= thres]
        t3=time.time()
        p = torch.nn.functional.pad(points,(0,1),"constant",-1).cpu().numpy()
        res = _ext.dbscan(p, 1, 5, 0.1)
        res = torch.tensor(res).to(TORCH_DEVICE)
        # dbscan = DBSCAN(eps=0.1, min_samples=5,verbose=True)
        # dbscan.fit(points.numpy())
        t4=time.time()
        inds = res[:, 3]
        most_ind, nums =torch.mode(inds)
        plane = points[inds == most_ind]
        # counts = Counter(dbscan.labels_)
        # most_ind, nums = counts.most_common()[0]
        # plane = points[dbscan.labels_ == most_ind]
        t5=time.time()
        print("---choose_inliers---random downsample:%.2fms"%((t2-t1)*1000))
        print("---choose_inliers---points in plane:%.2fms"%((t3-t2)*1000))
        print("---choose_inliers---dbscan:%.2fms"%((t4-t3)*1000))
        print("---choose_inliers---collection:%.2fms"%((t5-t4)*1000))
        print("---choose_inliers---choose_inliers done")
        return plane
    except BaseException:
        logging.error("---choose_inliers---choose_inlierserror in choose_inliers")
        return []


def find_rect(plane, grav,res):
    """
    Find 2D convex hull of points
    :param plane: points [n,3]
    :return: 3D vertex points of the hull [m,3]
    """
    try:
        grav=torch.tensor(grav).float().to(TORCH_DEVICE)
        grav = grav / torch.linalg.norm(grav)
        theta = torch.arccos(torch.inner(grav.squeeze(), torch.tensor([0, 0, -1]).float().to(TORCH_DEVICE)))
        axis = torch.cross(grav.squeeze(), torch.tensor([0, 0, -1]).float().to(TORCH_DEVICE))
        axis = axis / torch.linalg.norm(axis)
        Rg = Rodrigues(axis * theta)
        res["Rg"]=Rg
        points2 = torch.matmul(plane, Rg.T)
        p2 = points2[:, :2]
        hull = ConvexHull(p2.cpu())
        area1=hull.volume
        vert = points2[hull.vertices]
        h = vert.mean(0)[2]
        rect = minimum_bounding_rectangle(vert[:, :2])
        rect = torch.nn.functional.pad(rect, (0,1),"constant",h)
        rect = torch.matmul(rect, Rg)
        area2=torch.norm(torch.cross(rect[1]-rect[0],rect[2]-rect[0]))
        if area1/area2>0.85:
            table_shape="rect"
        else:
            table_shape="circle"
        res["table"]["table_shape"]=table_shape
        #logging.info("---find_rect---find_hull done")
        return rect
    except BaseException:
        logging.error("---find_rect---error in find_rect")
        return []



def parse_detection(result2d):
    '''

    :param result2d: {["bbox":[x1,y1,x2,y2],'rec_docs':str,'rec_scores':float],...}
    :return:[[id,x1,y1,w,h,conf]]
    '''
    try:
        if len(result2d) == 0:
            #logging.warning("---parse_detection---No objects detected")
            return {"result":[]}
        res = {"result":[]}
        #result2d = sorted(result2d, key=lambda x: x['bbox'][0])
        for i in range(len(result2d)):
            try:
                if result2d[i]["rec_docs"] in ["table","hand"]:
                    item = {}
                    item["category"] = result2d[i]["rec_docs"]
                    item["box2d"] = [result2d[i]["bbox"][0],
                                     result2d[i]["bbox"][1],
                                     result2d[i]["bbox"][2],
                                     result2d[i]["bbox"][3]]
                    item["score"] = result2d[i]["rec_scores"]
                    item["mask"] = result2d[i]["mask"]
                    res[item["category"]]=item
                else:
                    item={}
                    item["category"]=result2d[i]["rec_docs"]
                    item["box2d"]=[result2d[i]["bbox"][0],
                                result2d[i]["bbox"][1],
                                result2d[i]["bbox"][2],
                                result2d[i]["bbox"][3]]
                    item["score"]=result2d[i]["rec_scores"]
                    #item["track_id"] = int(result2d[i]["track_id"])
                    item["track_id"]=int(i)
                    item["mask"]=result2d[i]["mask"]
                    res["result"].append(item)
            except BaseException:
                logging.error("---parse_detection---error in parse_detection")
                logging.error("---parse_detection---error item is:")
                logging.error(result2d[i])
                continue
        #logging.info("---parse_detection---%d objects detected" % (i + 1))
        return res
    except BaseException:
        #logging.error("---parse_detection---error in parse_detection return []")
        return []


def process_result(msg,K,res):
    try:
        try:
            if "table" in res:
                table_shape=res["table"]["table_shape"]
                rect=res["table"]["rect"]
                a=rect[1]-rect[0]
                b=rect[3]-rect[0]
                a,b=max(torch.norm(a),torch.norm(b)),min(torch.norm(a),torch.norm(b))
                if table_shape=="circle":
                    r=max(a,b)
                    uv, uvd = project_upright_camera_to_image(rect, K)
                    msg["result"].append({"category": "table",
                                          "shape":"circle",
                                          "id": "0",
                                          "box2d": uv.ravel().tolist(),
                                          "box3d": rect.ravel().tolist(),
                                          "position": rect.mean(0).tolist(),
                                          "radius":r.tolist()})
                else:
                    uv, uvd = project_upright_camera_to_image(rect, K)
                    msg["result"].append({"category": "table",
                                          "shape":"rectagle",
                                          "id": "0",
                                          "box2d": uv.ravel().tolist(),
                                          "box3d": rect.ravel().tolist(),
                                          "position": rect.mean(0).tolist()})
        except BaseException:
            logging.error("---process_result---error in process rect")
            logging.error(res["rect"])

        try:
            if "hand" in res:
                msg["result"].append({"category": "hand",
                                      "id": "0",
                                      "box2d":[
                                        float(res["hand"]["box2d"][0]),
                                        float(res["hand"]["box2d"][1]),
                                        float(res["hand"]["box2d"][2]),
                                        float(res["hand"]["box2d"][3])] ,
                                      "box3d":res["hand"]["box3d"].ravel().tolist(),
                                      "position": res["hand"]["position"].tolist()})
        except:
            logging.error("---process_result---error in process hand,hand is")
            logging.error(res["hand"])

        for i,item in enumerate(res["result"]):
            try:
                if item is not None:
                    msg["result"].append(
                        {
                            "category": item["category"],
                            "id": item["track_id"],
                            "box2d": [
                                float(item["box2d"][0]),
                                float(item["box2d"][1]),
                                float(item["box2d"][2]),
                                float(item["box2d"][3])],
                            "box3d": item["box3d"].ravel().tolist(),
                            "position": item["box3d"].mean(0).tolist(),
                            "score": item["score"].tolist()
                        }
                    )
            except BaseException:
                logging.error("---process_result---error in process results")
        #logging.info("---process_result---process result done")
        return msg
    except BaseException:
        logging.error("---process_result---error in process result")
        return msg


def minimum_bounding_rectangle(points):
    """
    Find the smallest bounding rectangle for a set of points.
    Returns a set of points representing the corners of the bounding box.

    :param points: an nx2 matrix of coordinates
    :rval: an nx2 matrix of coordinates
    """

    pi2 = np.pi / 2.

    # get the convex hull for the points
    hull_points=points
    # calculate edge angles

    edges = hull_points[1:] - hull_points[:-1]
    angles = torch.arctan(edges[:, 1]/edges[:, 0])

    angles = torch.abs(angles%pi2)
    angles = torch.unique(angles)

    # find rotation matrices
    rotations = torch.stack([
        torch.cos(angles),
        torch.cos(angles - pi2),
        torch.cos(angles + pi2),
        torch.cos(angles)]).T
    rotations = rotations.reshape((-1, 2, 2))

    # apply rotations to the hull
    rot_points = torch.matmul(rotations, hull_points.T)

    # find the bounding points
    min_x = torch.min(rot_points[:, 0], dim=1)[0]
    max_x = torch.max(rot_points[:, 0], dim=1)[0]
    min_y = torch.min(rot_points[:, 1], dim=1)[0]
    max_y = torch.max(rot_points[:, 1], dim=1)[0]

    # find the box with the best area
    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = torch.argmin(areas)

    # return the best box
    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]
    r = rotations[best_idx]

    rval = torch.zeros((4, 2)).to(TORCH_DEVICE)
    rval[0] = torch.matmul(torch.stack([x1, y2]), r)
    rval[1] = torch.matmul(torch.stack([x2, y2]), r)
    rval[1] = torch.matmul(torch.stack([x2, y2]), r)
    rval[2] = torch.matmul(torch.stack([x2, y1]), r)
    rval[3] = torch.matmul(torch.stack([x1, y1]), r)

    return rval


def table_detection_threading(points_total,points, grav, res,W,H):
    try:
        if "table" in res:
            t1=time.time()
            mask=res["table"]["mask"]
            v,u=torch.where(mask>0)
            index=v*W+u
            plane_points = points_total[index]
            plane_points = plane_points[plane_points[:, 2] > 0]
            bestsolve = RANSAC_plane_fitting(
                plane_points, grav, nums=RANSAC_POINTS, r=PLANE_DIST_THRES, M=RANSAC_M)
            t2=time.time()
            if len(bestsolve)> 0:
                if bestsolve[3] < 0:
                    bestsolve *= -1
                res["table"]["bestsolve"] = bestsolve
                t3 = time.time()
                plane = choose_inliers(bestsolve, points, thres=PLANE_DIST_THRES,nums=PLANE_DOWNSAMPLE_NUMS)
                res["table"]["points"] = plane
                t4 = time.time()
                if len(plane) < PLANE_POINTS_THRES:
                    res["table"]["points"] = None
                    res["table"]["rect"] = None
                else:
                    rect = find_rect(plane, grav, res)
                    res["table"]["rect"] = rect
                t5 = time.time()
                print("---table_detection_threading---table_detection_total:%.2fms" % ((t5 - t1) * 1000))
                print("---table_detection_threading---solve_plane:%.2fms" % ((t2 - t1) * 1000))
                print("---table_detection_threading---aggregate:%.2fms" % ((t3 - t2) * 1000))
                print("---table_detection_threading---choose_inlier: %.2fms" % ((t4 - t3) * 1000))
                print("---table_detection_threading---find_rect:%.2fms" % ((t5 - t4) * 1000))
                print("---table_detection_threading---table rect detected")
                return

        res["table"]={}
        res["table"]["b"] = []
        res["table"]["xyz"] = []
        threads=[]
        t1=time.time()
        for i in range(len(res["result"])):
            roi = res["result"][i]["box2d"]
            threads.append(threading.Thread(target=table_detection_single,
                                            args=(grav,roi, points_total,W,H,res,i)))
        for i in range(len(threads)):
            threads[i].start()
        for i in range(len(threads)):
            threads[i].join()
        t2=time.time()
        b = torch.stack(res["table"]["b"])
        inner = torch.abs(torch.inner(b[:, :3], torch.tensor(grav).float().to(TORCH_DEVICE)))
        angle = torch.arccos(inner) * RADIAN2DEGREE
        inds = torch.where(angle < PLANE_GRAV_DEGREE_THRES)[0]

        b=b[inds]
        mindist = b[:,3].min()
        inds = torch.where(b[:, 3] - mindist < MIN_DIST_THRES)[0]
        b = b[inds]

        bestsolve = b.mean(0)
        res["table"]["bestsolve"]=bestsolve
        t3=time.time()
        plane = choose_inliers(bestsolve,points,thres=PLANE_DIST_THRES)
        res["table"]["points"]=plane
        t4=time.time()
        if len(plane) < PLANE_POINTS_THRES:
            res["table"]["points"]=None
            res["table"]["rect"] = None
        else:
            rect=find_rect(plane, grav,res)
            res["table"]["rect"]=rect
        t5=time.time()
        #logging.info("---table_detection_threading---table_detection_total:%.2fms"%((t5-t1)*1000))
        #logging.info("---table_detection_threading---solve_plane:%.2fms"% ((t2-t1)*1000))
        #logging.info("---table_detection_threading---aggregate:%.2fms"%((t3-t2)*1000))
        #logging.info("---table_detection_threading---choose_inlier: %.2fms"%((t4-t3)*1000))
        #logging.info("---table_detection_threading---find_rect:%.2fms"%((t5-t4)*1000))
        #logging.info("Table rect detected")
    except BaseException:
        #logging.warning("No table rect detected")
        res["table"]["rect"] = None



def table_detection_single(grav,roi,points_total,W,H,res,i):
    t1=time.time()
    plane_points= find_neibour_points(roi, points_total,W,H)
    if len(plane_points) < PLANE_POINTS_THRES:
        return
    t2=time.time()
    bestsolve = RANSAC_plane_fitting(
        plane_points, grav,nums=RANSAC_POINTS,r=PLANE_DIST_THRES, M=RANSAC_M)
    if len(bestsolve)==0:
        return
    t3=time.time()
    if bestsolve[3] < 0:
        bestsolve *= -1
    res["result"][i]["bestsolve"]=bestsolve
    res["table"]["b"].append(bestsolve)
    #logging.info("---table_detection_single---find_neibour_points:%.2fms"%((t2-t1)*1000))
    #logging.info("---table_detection_single---RANSAC_plane_fitting:%.2fms"%((t3-t2)*1000))


def compute_9d_box_threading(points_total,grav,res,W,H):

    threads=[]
    t1=time.time()
    for i in range(len(res["result"])):
        threads.append(threading.Thread(target=compute_9d_box_sigle, args=(points_total,grav,res,i,W,H)))
    for i in range(len(threads)):
        threads[i].start()
    for i in range(len(threads)):
        threads[i].join()
    t2=time.time()
    #logging.info("---compute_9d_box_threading---compute_9d_box_threading:%.2fms"%((t2-t1)*1000))



def compute_9d_box_sigle(points_total, grav,res,i,W,H):
    try:
        if "mask" in res["result"][i]:
            time_stat = {}
            t1=time.time()
            mask = res["result"][i]["mask"]
            v, u = torch.where(mask > 0)
            index = v * W + u
            points = points_total[index]
            points = points[points[:, 2] > 0]
            t2 = time.time()
            if "depth2points" not in time_stat:
                time_stat["depth2points"] = (t2 - t1) * 1000
            else:
                time_stat["depth2points"] += (t2 - t1) * 1000
            grav = torch.tensor(grav).float().to(TORCH_DEVICE)
            grav = grav / torch.linalg.norm(grav)
            theta = torch.arccos(torch.inner(grav.squeeze(), torch.tensor([0, 0, -1]).float().to(TORCH_DEVICE)))
            axis = torch.cross(grav.squeeze(), torch.tensor([0, 0, -1]).float().to(TORCH_DEVICE))
            axis = axis / torch.linalg.norm(axis)
            Rg = Rodrigues(axis * theta)
            points = torch.matmul(points, Rg.T)
            t1 = time.time()
            if "calib" not in time_stat:
                time_stat["calib"] = (t1 - t2)*1000
            else:
                time_stat["calib"] += (t1 - t2)*1000
            if "bestsolve" in res["result"][i]:
                current_solve=res["result"][i]["bestsolve"]
                inner = torch.abs(torch.inner(current_solve[:3], grav))
                angle = torch.arccos(inner) * RADIAN2DEGREE
                if angle > ANGLE_THRES:
                    current_solve = res["table"]["bestsolve"]
            else:
                current_solve=res["table"]["bestsolve"]

            FLOOR_H = torch.matmul(torch.tensor([0, 0, -current_solve[3] / current_solve[2]]).to(TORCH_DEVICE), Rg.T)[2]
            z0 = FLOOR_H
            points = points[points[:, 2] > z0 + OBJ_ABOVE_TABLE_THRES]
            t2 = time.time()
            if "filter" not in time_stat:
                time_stat["filter"] = (t2 - t1) * 1000
            else:
                time_stat["filter"] += (t2 - t1) * 1000

            # cluster
            t1 = time.time()
            if len(points) > OBJ_SAMPLE_POINTS:
                index = torch.randperm(len(points))[:OBJ_SAMPLE_POINTS].to(TORCH_DEVICE)
                points = points[index]
            t2 = time.time()
            # cores, index = dbscan(points, eps=0.05)
            p = torch.nn.functional.pad(points, (0, 1), "constant", -1)
            cluster_res = _ext.dbscan(p.cpu().numpy(), 1, 5, OBJ_DBSCAN_MARGIN)
            cluster_res = torch.tensor(cluster_res).to(TORCH_DEVICE)
            if len(cluster_res) == 0:
                t3 = time.time()
                return
            index = cluster_res[:, 3]
            index = cluster_res[:, 3]
            t3 = time.time()
            inds, nums = torch.mode(index)
            points = points[index == inds]
            obj = points
            obj = torch.matmul(obj, Rg)
            res["result"][i]["points"] = obj
            h = points[:, 2].max() - FLOOR_H
            t4 = time.time()

            if "cluster" not in time_stat:
                time_stat["cluster"] = (t4 - t1) * 1000
                time_stat["sample"] = (t2 - t1) * 1000
                time_stat["dbscan"] = (t3 - t2) * 1000
                time_stat["collction"] = (t4 - t3) * 1000
            else:
                time_stat["cluster"] += (t4 - t1) * 1000
                time_stat["sample"] += (t2 - t1) * 1000
                time_stat["dbscan"] += (t3 - t2) * 1000
                time_stat["collction"] += (t4 - t3) * 1000

            # estimate d
            t1 = time.time()
            sample = points[:, :2]
            dist = torch.linalg.norm(sample.reshape([-1, 1, 2]) - sample.reshape([1, -1, 2]), axis=-1)
            d = dist.max()
            t2 = time.time()
            if "estimate_d" not in time_stat:
                time_stat["estimate_d"] = (t2 - t1) * 1000
            else:
                time_stat["estimate_d"] += (t2 - t1) * 1000

            # make box
            t1 = time.time()
            z1 = z0 + h
            dx = points[:, 0].max() - points[:, 0].min()
            if dx < d:
                x0 = points[:, 0].min() - (d - dx) / 2
                x1 = points[:, 0].max() + (d - dx) / 2
            else:
                x0 = points[:, 0].min()
                x1 = points[:, 0].max()
            dy = points[:, 1].max() - points[:, 1].min()
            if dy < d:
                y0 = points[:, 1].min()
                y1 = points[:, 1].min() + d
            else:
                y0 = points[:, 1].min()[1]
                y1 = points[:, 1].max()[1]
            center = torch.tensor([(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2]).to(TORCH_DEVICE)
            box3d = center + torch.tensor([[-d / 2, -d / 2, -h / 2],
                                           [-d / 2, -d / 2, h / 2],
                                           [-d / 2, d / 2, -h / 2],
                                           [-d / 2, d / 2, h / 2],
                                           [d / 2, -d / 2, -h / 2],
                                           [d / 2, -d / 2, h / 2],
                                           [d / 2, d / 2, -h / 2],
                                           [d / 2, d / 2, h / 2],
                                           ]).to(TORCH_DEVICE)
            #print("---compute_9d_box_sigle---compute_9d_box_single time:")
            #for k, v in time_stat.items():
            #    print("---compute_9d_box_sigle---compute_9d_box_single:%s : %.2fms" % (k, v))
            res["result"][i]["box3d"] = torch.matmul(box3d, Rg)

        else:
            time_stat = {}
            box2d = res["result"][i]["box2d"]
            score = res["result"][i]["score"]
            u1, v1, u2, v2 = box2d
            w=u2-u1
            h=v2-v1
            t1 = time.time()
            index = get_points_index(W, H, u1, v1, u2, v2)
            points = points_total[index]
            points = points[points[:, 2] > 0]
            # depth2 = np.ascontiguousarray(depth[v1:v2, u1:u2])
            # res = _ext.depth2xyz(depth2.T.ravel(), K.ravel(), u2 - u1, v2 - v1, u1, v1, u2, v2)
            # points = np.array(res).reshape([-1, 3])
            if len(points)==0:
                return
            t2 = time.time()
            if "depth2points" not in time_stat:
                time_stat["depth2points"] = (t2 - t1)*1000
            else:
                time_stat["depth2points"] += (t2 - t1)*1000

            # calib
            t1 = time.time()
            grav = torch.tensor(grav).float().to(TORCH_DEVICE)
            grav = grav / torch.linalg.norm(grav)
            theta = torch.arccos(torch.inner(grav.squeeze(), torch.tensor([0, 0, -1]).float().to(TORCH_DEVICE)))
            axis = torch.cross(grav.squeeze(), torch.tensor([0, 0, -1]).float().to(TORCH_DEVICE))
            axis = axis / torch.linalg.norm(axis)
            Rg = Rodrigues(axis * theta)
            points = torch.matmul(points, Rg.T)
            t2 = time.time()
            if "calib" not in time_stat:
                time_stat["calib"] = (t2 - t1)*1000
            else:
                time_stat["calib"] += (t2 - t1)*1000

            # filter
            t1 = time.time()

            # plane_points, target_xyz = find_neibour_points(box2d,points_total,W,H)
            # current_solve = RANSAC_plane_fitting(
            #     plane_points,grav,nums=RANSAC_POINTS, r=PLANE_DIST_THRES, M=RANSAC_M)

            if "bestsolve" in res["result"][i]:
                current_solve=res["result"][i]["bestsolve"]
                inner = torch.abs(torch.inner(current_solve[:3], grav))
                angle = torch.arccos(inner) * RADIAN2DEGREE
                if angle > ANGLE_THRES:
                    current_solve = res["table"]["bestsolve"]
            else:
                current_solve=res["table"]["bestsolve"]

            FLOOR_H = torch.matmul(torch.tensor([0, 0, -current_solve[3] / current_solve[2]]).to(TORCH_DEVICE), Rg.T)[2]
            z0 = FLOOR_H
            points = points[points[:, 2] > z0 + OBJ_ABOVE_TABLE_THRES]
            t2 = time.time()
            if "filter" not in time_stat:
                time_stat["filter"] = (t2 - t1)*1000
            else:
                time_stat["filter"] += (t2 - t1)*1000

            # cluster
            t1 = time.time()
            if len(points) > OBJ_SAMPLE_POINTS:
                index=torch.randperm(len(points))[:OBJ_SAMPLE_POINTS].to(TORCH_DEVICE)
                points = points[index]
            t2 = time.time()
            #cores, index = dbscan(points, eps=0.05)
            p = torch.nn.functional.pad(points, (0, 1),"constant", -1)
            cluster_res = _ext.dbscan(p.cpu().numpy(), 1, 5, OBJ_DBSCAN_MARGIN)
            cluster_res = torch.tensor(cluster_res).to(TORCH_DEVICE)
            if len(cluster_res) == 0:
                t3 = time.time()
                return
            index = cluster_res[:, 3]
            index = cluster_res[:, 3]
            t3 = time.time()
            inds, nums = torch.mode(index)
            points = points[index == inds]
            obj= points
            obj=torch.matmul(obj,Rg)
            res["result"][i]["points"]=obj
            h = points[:,2].max()-FLOOR_H
            t4 = time.time()


            if "cluster" not in time_stat:
                time_stat["cluster"] = (t4 - t1)*1000
                time_stat["sample"] = (t2 - t1)*1000
                time_stat["dbscan"] = (t3 - t2)*1000
                time_stat["collction"] = (t4 - t3)*1000
            else:
                time_stat["cluster"] += (t4 - t1)*1000
                time_stat["sample"] += (t2 - t1)*1000
                time_stat["dbscan"] += (t3 - t2)*1000
                time_stat["collction"] += (t4 - t3)*1000

            # estimate d
            t1 = time.time()
            sample = points[:, :2]
            dist = torch.linalg.norm(sample.reshape([-1, 1, 2]) - sample.reshape([1, -1, 2]), axis=-1)
            d = dist.max()
            t2 = time.time()
            if "estimate_d" not in time_stat:
                time_stat["estimate_d"] = (t2 - t1)*1000
            else:
                time_stat["estimate_d"] += (t2 - t1)*1000

            # make box
            t1 = time.time()
            z1 = z0 + h
            dx = points[:,0].max() - points[:,0].min()
            if dx < d:
                x0 = points[:,0].min() - (d - dx) / 2
                x1 = points[:,0].max() + (d - dx) / 2
            else:
                x0 = points[:,0].min()
                x1 = points[:,0].max()
            dy = points[:,1].max() - points[:,1].min()
            if dy < d:
                y0 = points[:,1].min()
                y1 = points[:,1].min() + d
            else:
                y0 = points[:,1].min()[1]
                y1 = points[:,1].max()[1]
            center = torch.tensor([(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2]).to(TORCH_DEVICE)
            box3d = center + torch.tensor([[-d / 2, -d / 2, -h / 2],
                                       [-d / 2, -d / 2, h / 2],
                                       [-d / 2, d / 2, -h / 2],
                                       [-d / 2, d / 2, h / 2],
                                       [d / 2, -d / 2, -h / 2],
                                       [d / 2, -d / 2, h / 2],
                                       [d / 2, d / 2, -h / 2],
                                       [d / 2, d / 2, h / 2],
                                       ]).to(TORCH_DEVICE)
            #print("---compute_9d_box_sigle---compute_9d_box_single time:")
            #for k,v in time_stat.items():
            #    print("---compute_9d_box_sigle---compute_9d_box_single:%s : %.2fms"%(k,v))
            res["result"][i]["box3d"]= torch.matmul(box3d, Rg)

    except BaseException:
        logging.error("---compute_9d_box_sigle---error in compute_9d_box_single")


def compute_hand(points_total,res,W,H):
    try:
        if "hand" in res:
            t1 = time.time()
            mask = res["hand"]["mask"]
            v, u = torch.where(mask > 0)
            index = v * W + u
            hand_points = points_total[index]
            hand_points = hand_points[hand_points[:, 2] > 0]
            t2=time.time()
            if len(hand_points) >HAND_SAMPLE_POINTS:
                index = torch.randperm(len(hand_points))[:HAND_SAMPLE_POINTS].to(TORCH_DEVICE)
                points = hand_points[index]
            t3 = time.time()
            # cores, index = dbscan(points, eps=0.05)
            p = torch.nn.functional.pad(points, (0, 1), "constant", -1)
            cluster_res = _ext.dbscan(p.cpu().numpy(), 1, 5, HAND_DBSCAN_MARGIN)
            cluster_res = torch.tensor(cluster_res).to(TORCH_DEVICE)
            if len(cluster_res) == 0:
                return
            index = cluster_res[:, 3]
            inds, nums = torch.mode(index)
            points = points[index == inds]
            t4 = time.time()
            R=compute_R(points)
            box3d=compute_box3d(points,R)
            t5=time.time()
            res["hand"]["points"]=points
            res["hand"]["box3d"]=box3d
            res["hand"]["position"]=box3d.mean(0)
            #print("---compute_hand---total:%.2fms"%((t5-t1)*1000))
            #print("---compute_hand---mask2points:%.2fms" % ((t2 - t1) * 1000))
            #print("---compute_hand---downsample:%.2fms" % ((t3 - t2) * 1000))
            #print("---compute_hand---dbscan:%.2fms" % ((t4 - t3) * 1000))
            #print("---compute_hand---compute_box3d:%.2fms" % ((t5 - t4) * 1000))
            #print("---compute_hand---hand detection done")
        else:
            logging.warning("---compute_hand---no hand detected")
    except:
        logging.warning("---compute_hand---error in compute_hand")


def compute_R(points):
    """
    use pca to compute rotation of a string
    :param string: nx3 points of a string
    :return: 3x3 rotation
    """
    v, D= pca(points)
    R = np.zeros([3, 3])
    R[:, 0] = D[:, 1]
    R[:, 1] = D[:, 0]
    R[:, 2] = D[:, 2]
    if R[:, 0][abs(R[:, 0]).argmax()] < 0:
        R[:, 0] *= -1
    if R[:, 1][abs(R[:, 1]).argmax()] < 0:
        R[:, 1] *= -1
    if R[:, 2][abs(R[:, 2]).argmax()] < 0:
        R[:, 2] *= -1
    if np.linalg.det(R) < 0:
        R = R[:, ::-1]
    return R

def compute_box3d(points, R):
    points2 = np.dot(points, R)
    center = (points2.max(0) + points2.min(0)) / 2
    points2 -= center
    l, h, w = points2.max(0) - points2.min(0)
    box3d = np.array([[-l / 2, -h / 2, -w / 2],
                      [-l / 2, -h / 2, w / 2],
                      [-l / 2, h / 2, -w / 2],
                      [-l / 2, h / 2, w / 2],
                      [l / 2, -h / 2, -w / 2],
                      [l / 2, -h / 2, w / 2],
                      [l / 2, h / 2, -w / 2],
                      [l / 2, h / 2, w / 2]])
    box3d += center
    box3d = np.dot(box3d, R.T)
    return box3d

def make_grid(res):
    grid_info=res["grid"]
    grid_in_ginger=grid_info["grid_in_ginger"]
    grid_dim=grid_info["grid_dim"]
    resolution=grid_info["resolution"]
    grid=np.zeros(np.array(grid_dim),dtype=np.float32)
    try:
        if "table" in res:
            make_grid_single(grid,res["table"],grid_info)
        if "hand" in res:
            make_grid_single(grid,res["hand"],grid_info)
        for i,item in enumerate(res["result"]):
            if "box3d" in item:
                make_grid_single(grid,item,grid_info)
        #logging.info("---make_grid---make_grid done")
        return grid
        
    except:
        #logging.error("---make_grid---error in make_grid")
        return grid

def make_grid_single(grid,item,grid_info):
    try:
        #print(grid_info)
        t1=time.time()
        grid_in_ginger=grid_info["grid_in_ginger"]
        grid_dim=grid_info["grid_dim"]
        resolution=grid_info["resolution"]
        tf=grid_info["tf"]
        R=tf[:3,:3]
        t=tf[:3,3]
        target=grid_info["target_category"]
        points = np.array(item["points"])
        points_in_ginger = np.dot(points,R.T)+t
        points_in_grid = (points_in_ginger - grid_in_ginger) // resolution
        points_in_grid=points_in_grid[np.logical_and(points_in_grid[:,0]<grid_dim[0],points_in_grid[:,0]>0)]
        points_in_grid=points_in_grid[np.logical_and(points_in_grid[:,1]<0,points_in_grid[:,1]>-grid_dim[2])]
        points_in_grid=points_in_grid[np.logical_and(points_in_grid[:,2]<grid_dim[1],points_in_grid[:,2]>0)]
        x,y,z=points_in_grid.astype(int).T
        y,z=z,-y
        #logging.info(grid_info)
        #logging.info("---make_grid_single---points in ginger:%d"%len(points_in_ginger))
        #logging.info(points_in_ginger)
        #logging.info("---make_grid_single---points in grid:%d"%len(points_in_grid))
        #logging.info("---make_grid_single---category:%s"%item["category"])
        #logging.info("---make_grid_single---target:%s"%target)
        if item["category"]=="table":
            grid[x,y,z]=-1
        elif item["category"]=="hand":
            grid[x,y,z]=0.5
        elif item["category"]==target:
            grid[x,y,z]=1
        else:
            grid[x,y,z]=-0.5
        t2=time.time()
        #print("---make_grid_single---%s:%.2fms"%(item["category"],(t2-t1)*1000))
    except:
        logging.info(item)
        logging.info(grid_info)



