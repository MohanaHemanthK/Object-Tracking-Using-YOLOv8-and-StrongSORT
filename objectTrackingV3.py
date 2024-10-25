# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 22:07:47 2024

@author: mohan
"""

import cv2
from ultralytics import YOLO
import numpy as np
import torch

#from strong_sort.utils.parser import get_config
from strong_sort.strong_sort import StrongSORT
from strong_sort.sort.tracker import Tracker
# from torchreid import models
#from torchreid.utils import feature_extractor
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = YOLO('yolov8x.pt')
class_names = model.names
#class_id_mapping = {name: idx for idx, name in enumerate(class_names)}
class_id_mapping = {idx: name for idx, name in enumerate(class_names)}


COLORS = [(255, 0, 0), (255, 0, 255), (0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

#strong_sort_wts = 'osnet_x0_25_msmt17.pt'
strong_sort_wts = 'osnet_x1_0_imagenet.pt'
#strong_sort_wts = "osnet_x1_0_market.pt"
#strong_sort_wts = "resnet50_msmt17.pt"

#tracker = StrongSORT(model_weights = strong_sort_wts, device = device, fp16=True, max_age= 1000)
#tracker = StrongSORT(model_weights = strong_sort_wts, mc_lambda=0.999, ema_alpha=0.05, max_dist=0.35, max_iou_distance=0.9 , device = device, fp16=True, max_age= 1000)
# tracker = StrongSORT(model_weights = strong_sort_wts, device=device, fp16 = True, 
#                       ema_alpha=0.89, max_age = 10000, max_dist = 0.16, max_iou_distance = 0.543, mc_lambda = 0.995)

#Below one worked, for pranahith and me moving out of frame with out shuffling (even max_age = 1000 works)
# tracker = StrongSORT(model_weights = strong_sort_wts, device=device, fp16 = True, 
#                       ema_alpha=0.8, max_age = 10000000000000, max_dist = 0.15, max_iou_distance = 0.016, mc_lambda = 0.9999)

#Below one worked so good with banner and 2 persons
# tracker = StrongSORT(model_weights = strong_sort_wts, device=device, fp16 = True, 
#                       ema_alpha=0.08, max_age = 1000, max_dist = 0.15, max_iou_distance = 0.16, mc_lambda = 0.009999)

tracker = StrongSORT(model_weights = strong_sort_wts, device=device, fp16 = False, 
                      ema_alpha=0.8, max_age = 10000, max_dist = 0.15, max_iou_distance = 0.999, mc_lambda = 0.9999)

#Below tracker is working fine with pre-recorded video
# tracker = StrongSORT(model_weights = strong_sort_wts, device=device, fp16 = True, 
#                       ema_alpha=0.89, max_age = 70, max_dist = 0.20, max_iou_distance = 0.543, mc_lambda = 0.999)

#camera = cv2.VideoCapture('object_tracking.mp4')
#camera = cv2.VideoCapture('objectTrackingP5.mp4')
#camera = cv2.VideoCapture('objectTrackingP3.mp4')
#camera = cv2.VideoCapture('ObjectTrackingP3x2.mp4')
#camera = cv2.VideoCapture(1)
#camera = cv2.VideoCapture("2poffScrn.mp4")
#camera = cv2.VideoCapture("2pwbunker.mp4")
#camera = cv2.VideoCapture('ObjectTrackingwBunker.mp4')
#camera = cv2.VideoCapture("http://192.168.0.61:8080/video")
#camera = cv2.VideoCapture('ObjTra2PRbunker.mp4')
#camera = cv2.VideoCapture('ObjTra1pRbunker.mp4')
#camera = cv2.VideoCapture('WIN_20240212_11_29_30_Pro.mp4')
#camera = cv2.VideoCapture('WIN_20240212_11_47_36_Pro.mp4')
#camera = cv2.VideoCapture('ObjTra4pwBunker.mp4')
#camera = cv2.VideoCapture('YTsampleOutside.mp4')
camera = cv2.VideoCapture(0)
if camera.set(cv2.CAP_PROP_FRAME_WIDTH, 720) & camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1280):
    print("Resolution set successfully.")
else:
    print("Setting resolution failed. Camera might not support it.")

highlight_track_id = None

while True:
    success, frame = camera.read()
    if not success:
        print('Error reading camera!')
        break
    
    #og_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #frame = og_frame.copy()
    og_frame=frame.copy()
    results = model(frame, conf=0.75, show=True ,classes = 0)
    
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
        
    #x1,y1,x2,y2 = bboxes_xywh[0]
    bboxes_xywh = np.array(bboxes_xywh)
    confs = torch.tensor(confs)  # Convert confs to PyTorch tensor
    classes = torch.tensor(classes)  # Convert to PyTorch tensor
    
    tracks = tracker.update(bboxes_xywh, confs, classes, og_frame)
    
    #tracks=tracker.update(bboxes_xywh, confs, classes, og_frame)
    
    for track in tracker.tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
    
        bbox_tlwh = track.to_tlwh()
        x1,y1,w,h = bbox_tlwh
        #x1,y1,x2,y2=bbox_tlwh
        #x1, y1, x2, y2 = tracker._tlwh_to_xyxy(bbox_tlwh)
        # w=y2-y1
        # h=y2-y1
        
        
        track_id = track.track_id
        class_id = track.class_id
        conf = track.conf
        

        #color = COLORS[track_id % len(COLORS)]
        color = (0, 255, 0) if track_id == highlight_track_id else (255, 0, 0)
        

        cv2.rectangle(og_frame, (int(x1), int(y1)), (int(w+x1), int(h+y1)), color, 2)
        cv2.putText(og_frame, f"{class_names[class_id.item()]}: ID: {track_id} | conf: {conf}", (int(x1), int(y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        # cv2.rectangle(og_frame, (int(x1+(x2)/2),int(y1+(y2)/2)),(int(x2),int(y2)), color,2)
        # cv2.putText(og_frame, f"{class_names[class_id.item()]}: ID: {track_id}", (int(x1+(x2)/2), int(y1+(y2)/2) - 5),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.circle(og_frame, (0,0), 5, (255,0,0))

    
    cv2.imshow('video', og_frame)
    
    # Press q to close the camera and end the code.
    if cv2.waitKey(1) == ord("q"):
        break
    elif cv2.waitKey(1) == ord('t'):
        highlight_track_id = int(input("Enter the tracking ID to highlight in green: "))


camera.release()
cv2.destroyAllWindows()

