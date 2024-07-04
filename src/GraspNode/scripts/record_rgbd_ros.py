import cv2
import os
import sys
import numpy as np
import threading
import queue
import rospy
import time
import json
from cloud_robot_msgs.msg import ImageList
from std_srvs.srv import Trigger
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

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


    def full(self):
        return len(self.queue)==self.maxsize-1


    def empty(self):
        return super().empty()


    def size(self):
        return len(self.queue)
        
        
    def __repr__(self):
        return self.queue.__repr__()
        
        
            
class Timer():
    def __init__(self,callback,rate):
        self.rate=rate
        self.callback=callback
        self.lock=threading.Lock()
        self.thread=threading.Thread(target=self.run)
        self.thread.setDaemon(True)
        self.thread.start()
    def run(self):
        try:
            while True:
                t1=time.time()
                self.callback()
                t2=time.time()
                dt=t2-t1
                if dt<1/self.rate:
                    time.sleep(1/self.rate-dt)
        except:
            self.lock.acquire()
            self.thread._stop()
            self.lock.release()     
            
               
            
class RsCapture():
    def __init__(self):
        super().__init__()

        self.datapath=sys.argv[1]
        self.check_path()
        self.record=False

        self.data_queue=AutoQueue(1)
        self.open_camera_client = rospy.ServiceProxy("/iris_camera/resume", Trigger)
        self.close_camera_client = rospy.ServiceProxy("/iris_camera/pause", Trigger)
        self.K_client = rospy.ServiceProxy("/iris_camera/camera_parameters", Trigger)
        response = self.K_client()
        intrinsics = json.loads(response.message)["rgb_camera"]["intrinsic"]
        self.K = np.array([[intrinsics["fx"], 0, intrinsics["cx"]],
                       [0, intrinsics["fy"], intrinsics["cy"]],
                       [0, 0, 1]])
        self.cv_bridge = CvBridge()
        
        self.stream_listener = rospy.Subscriber("/realsense2_cam/rgb_depth_stream", ImageList,self.stream_listener_callback,queue_size=1,buff_size=1024*1024)
        self.open_camera_client()
        
        
    def check_path(self):
        if not os.path.exists(self.datapath):
            os.makedirs(os.path.join(self.datapath,"rgb"))
            os.makedirs(os.path.join(self.datapath,"depth"))
            os.makedirs(os.path.join(self.datapath,"intrinsics"))
            os.makedirs(os.path.join(self.datapath,"grav"))


    def stream_listener_callback(self, msgs):
        stamp=time.time()
        bgr = cv2.imdecode(np.frombuffer(msgs.data[:msgs.length[0]],dtype=np.uint8),cv2.IMREAD_COLOR)
        depth=cv2.imdecode(np.frombuffer(msgs.data[msgs.length[0]:msgs.length[0]+msgs.length[1]],dtype=np.uint8),cv2.IMREAD_ANYDEPTH)
        grav=np.array(msgs.data[msgs.length[0]+msgs.length[1]:msgs.length[0]+msgs.length[1]+msgs.length[2]].decode().split(",")).astype(np.float32)
        if grav[1]<0:
            grav=-grav
        self.data_queue.put((stamp,bgr,depth,grav))
        
        
    def capture(self):
        if self.data_queue.empty():
            time.sleep(0.01)
            return
        timestamp,bgr,depth,grav=self.data_queue.get()
        bgr_show = bgr.copy()
        if self.record:
            cv2.putText(bgr_show, "rec", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
            cv2.circle(bgr_show, (40, 45), 5, (0, 0, 255), thickness=-1)
        else:
            cv2.putText(bgr_show, "ready", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
            cv2.circle(bgr_show, (40, 45), 5, (0, 255, 0), thickness=-1)
        cv2.imshow('rgb' , bgr_show)
        c = cv2.waitKey(1)
        if c == ord("r"):
            self.record = ~self.record
        if self.record:
            bgr_save = bgr.copy()
            ts = time.time()
            cv2.imwrite(os.path.join(self.datapath, 'rgb/%.4f.jpg' % ts), bgr)
            cv2.imwrite(os.path.join(self.datapath, 'depth/%.4f.png' % ts), depth)
            np.savetxt(os.path.join(self.datapath,'grav/%.4f.txt'%ts),grav)
            np.savetxt(os.path.join(self.datapath, 'intrinsics/%.4f.txt' % ts), self.K, fmt="%.6f")
            return
        # print(c)
        if c == ord('q'):
            cv2.destroyAllWindows()
            self.stop=True
        elif c == ord('s'):
            ts = time.time()
            cv2.imwrite(os.path.join(self.datapath, 'rgb/%.4f.jpg' % ts), bgr)
            cv2.imwrite(os.path.join(self.datapath, 'depth/%.4f.png' % ts), depth)
            np.savetxt(os.path.join(self.datapath, 'grav/%.4f.txt' % ts), grav)
            np.savetxt(os.path.join(self.datapath, 'intrinsics/%.4f.txt' % ts), self.K, fmt="%.6f")
            bgr_show =bgr.copy()
            cv2.putText(bgr_show, "rec", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
            cv2.circle(bgr_show, (40, 45), 5, (0, 0, 255), thickness=-1)
            cv2.imshow('rgb', bgr_show)
            cv2.waitKey(1)
        
    
if __name__ == '__main__':
    rospy.init_node('rs_capture')
    node=RsCapture()
    while True:
        node.capture()
    rospy.signal_shutdown("exit")
        
