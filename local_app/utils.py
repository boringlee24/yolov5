import time
import signal
import os
import numpy as np
import requests
import json

# decode numpy array byte stream
class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

# every certain amount of time, send interrupt signal to PID
def clock(interval, pid, images, batch_size):
    while True:
        global request_list = random.sample(images, batch_size) # images to infer
        if handler_busy:
            # drop frame
            global drop_cnt += 1
        else:
            os.kill(pid, singal.SIGPOLL)
        time.sleep(interval)

def query_cloud(url, images):
    file_list = []
    for i in range(len(images)):
        file_list.append(('img', open(files[i], 'rb')))
    test_response = requests.post(url, files=file_list)
    if test_response.ok:
        return test_response
    else:
        raise RuntimeError('did not receive valid response from cloud')

# confidence level based offloader
# offload when edge max confidence < threshold
class Basic_Agent:
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

class Offload_Object:
    def __init__(self):
        self.index = []
        self.requests = []
    def add(self, index, request):
        self.index.append(index)
        self.requests.append(request)
    def query(self, url):
        response = query_cloud(url, self.requests)
        decoded = json.loads(response._content.decode('utf-8'))['bbox']
        decoded = [np.asarray(json.loads(x)) for x in decoded]
        return decoded



