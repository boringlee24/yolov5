#!/bin/bash

for CONF in 0.01 0.5 0.6 0.7 0.8
do 
	echo running yolov5x confidence $CONF
	python val.py --weights yolov5x.pt --data coco.yaml --img 1280 --half --conf-thres $CONF > runs/exp_v5x_1280p_${CONF}conf.log &&
	sleep 10 
	# echo running yolov5s confidence $CONF
	# python val.py --weights yolov5s.pt --data coco.yaml --img 640 --half --conf-thres $CONF > runs/exp_v5s_${CONF}conf.log &&
	# sleep 10
done

