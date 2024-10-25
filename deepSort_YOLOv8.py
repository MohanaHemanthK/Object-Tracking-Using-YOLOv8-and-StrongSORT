# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 01:51:07 2024

@author: mohan
"""

import cv2
from ultralytics import YOLO
import torch
import numpy as np

from deep_sort.deep_sort import DeepSort


import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

model = YOLO('yolov8x.pt') 
class_names = model.names
class_id_mapping = {idx: name for idx, name in enumerate(class_names)}


deep_sort_wts = "D:\Academics\RAwork\ObjectTracking\deep_sort\deep\checkpoint\ckpt.t7"
tracker = DeepSort(model_path=deep_sort_wts, max_age= 1000)

#camera = cv2.VideoCapture('WIN_20240212_11_29_30_Pro.mp4')
#camera = cv2.VideoCapture('ObjTra4pwBunker.mp4')
camera = cv2.VideoCapture(1)

while True:
    success,frame = camera.read()
    
    if not success:
        print('Error reading Camera!')
        break
    
    results = model(frame,conf=0.8, show = True, classes = 0)
    
    bboxes_xywh = []
    confs = []
    classes = []
    
    for result in results:
        boxes = result.boxes.xywh.tolist()
        class_ids = result.boxes.cls
        confidences = np.array(result.boxes.conf.tolist())
        class_labels = [class_id_mapping[int(class_id)] for class_id in class_ids]
        
        bboxes_xywh.extend(boxes)
        confs.extend(confidences)
        classes.extend(class_labels)
        
    bboxes_xywh = np.array(bboxes_xywh).reshape(-1, 4)
    
    tracks = tracker.update(bboxes_xywh, confs, frame)
    
    for track in tracker.tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        bbox_tlwh = track.to_tlwh()
        x1,y1,w,h = bbox_tlwh
        track_id = track.track_id
        #class_id = track.class_name
        #conf = track.conf
        
        color = (255, 0, 0)
            

        cv2.rectangle(frame, (int(x1), int(y1)), (int(w+x1), int(h+y1)), color, 2)
        # cv2.putText(frame, f"{class_names[class_id.item()]}: ID: {track_id} | conf: {conf}", (int(x1), int(y1 - 5)),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(frame, f" ID: {track_id}", (int(x1), int(y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.imshow('video', frame)
        
    if cv2.waitKey(1) == ord("q"):
        break
        
camera.release()
cv2.destroyAllWindows()