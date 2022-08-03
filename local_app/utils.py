import time
import signal
import os
import numpy as np

# decode numpy array byte stream
class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

# every certain amount of time, send interrupt signal to PID
def clock(interval, pid, images, batch_size):
    while True:
        global request = random.sample(images, batch_size) # images to infer
        if handler_busy:
            # drop frame
            global drop_cnt += 1
        else:
            os.kill(pid, singal.SIGPOLL)
        time.sleep(interval)

# confidence level based offloader
# offload when edge max confidence < threshold
class Basic_Offloader:
    def __init__(self, conf_thres, top_n):
        self.thres = conf_thres
        self.top_n = top_n

    # based on edge prediction, return offload True/False
    def decision(self, pred):
        # pred: xyxyn format
        '''        
        columns: x1 y1 x2 y2 conf class
        '''
        # take 5th column and top n rows
        conf = pred[:self.top_n, 4]
        mean_conf = np.mean(conf)
        if mean_conf >= self.thres:
            return False
        else:
            return True



