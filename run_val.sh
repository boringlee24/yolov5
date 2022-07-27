#!/bin/bash

python val.py --weights yolov5x.pt --data coco.yaml --img 640 --half --conf-thres 0.5
