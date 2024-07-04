import queue
import threading

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
     