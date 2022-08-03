import requests
import argparse
import signal
from utils import NumpyArrayEncoder, clock, Basic_Offloader
import threading
import _thread
import os
import torch
import glob
import pdb
import random

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

def init_model(model_name):
    model = torch.hub.load(repo_or_dir=f'{ROOT}', model='custom', 
        source='local', path=f'{ROOT}/{model_name}.pt', force_reload=True)#, pretrained=True)
    if torch.cuda.is_available() and half:
        model.half()
    # warm up the model
    model('/GIT/datasets/coco/images/val2017/000000000139.jpg', size=640)
    return model

#### decide process the frames ####
# take global variables from parsed args, and model
def frame_handler(signalNumber, frame)
    global handler_busy = True
    # first run on the edge
    preds = model(request, size=640)
    offload_req = []
    for pred in preds:
        pred_array = pred.xyxyn.cpu().numpy()    #detach()
        decision = offloader.decision()
        if decision:
            # offload to cloud
        # TODO: store prediction results

    global handler_busy = False

def run(args):
    model = init_model(args.local_model)

    signal.signal(signal.SIGPOLL, frame_handler)

    x = threading.Thread(target=utils.clock, args=(INTERVAL, PID), daemon=True)
    x.start()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--fps', type=int, default=8, help='frames per seconds')
    parser.add_argument('--batch', type=int, default=4, help='inference batch size')
    parser.add_argument('--local_model', type=str, default='yolov5s', help='local yolo model')
    parser.add_argument('--conf_thres', type=float, default=0.7, help='offload confidence threshold')
    parser.add_argument('--top_n', type=int, default=1, help='compare top n confidences vs threshold')    

    args = parser.parse_args()

    INTERVAL = 1 / (args.fps / args.batch)
    PID = os.getpid()

    images = glob.glob('/GIT/datasets/coco/images/val2017/*.jpg')
    offloader = Basic_Offloader(args.conf_thres, args.top_n)

    drop_cnt = 0
    request = [] # a subset of all images
    handler_busy = False


