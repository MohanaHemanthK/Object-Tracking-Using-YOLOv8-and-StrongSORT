# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 11:39:50 2024

@author: mohan
"""

import cv2
from ultralytics import YOLO
import supervision as sv
from supervision.detection.line_counter import LineZone, LineZoneAnnotator
#from supervision.geometry.dataclasses import Point
#from supervision.detection.line_counter import LineZone, 
from supervision.detection.core import Detections
#from supervision.detection import line_counter
import numpy as np
from strong_sort.strong_sort import StrongSORT
import torch
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
strong_sort_wts = "resnet50_msmt17.pt"
#tracker = StrongSORT(model_weights = strong_sort_wts, device=device, fp16 = False, ema_alpha=0.8, max_age = 1000, max_dist = 0.15, max_iou_distance = 0.999, mc_lambda = 0.9999)
tracker = StrongSORT(model_weights = strong_sort_wts, device=device, fp16 = True, 
                      ema_alpha=0.89, max_age = 10000, max_dist = 0.16, max_iou_distance = 0.543, mc_lambda = 0.995)

LINE_START = sv.Point(320, 470)
LINE_END = sv.Point(320, 10)

# LINE_START = sv.Point(640, 710)
# LINE_END = sv.Point(640, 10)

# LINE_START = sv.Point(1100, 710)
# LINE_END = sv.Point(1100, 10)

#camera = cv2.VideoCapture('ObjTraCount2Pbunker.mp4')
# camera = cv2.VideoCapture('ObjTraCount4PBunker.mp4')
camera = cv2.VideoCapture(2)
#camera = cv2.VideoCapture(1)
#qcamera = cv2.VideoCapture(0)
# if camera.set(cv2.CAP_PROP_FRAME_WIDTH, 720) & camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1280):
#     print("Resolution set successfully.")
# else:
#     print("Setting resolution failed. Camera might not support it.")


model = YOLO('yolov8l.pt')

CLASS_NAMES_DICT = model.model.names

class_id_mapping = {idx: name for idx, name in enumerate(CLASS_NAMES_DICT)}
# class_ids of interest - car, motorcycle, bus and truck
CLASS_ID = [0]

line_counter = LineZone(start=LINE_START, end=LINE_END)

line_annotator = LineZoneAnnotator(thickness=2,
                                   text_thickness=2,
                                   text_scale=1)

#line_annotator = sv.LineZoneAnnotator(thickness=4, text_thickness=4, text_scale=2)
box_annotator = sv.BoxAnnotator(thickness =2,
                                text_thickness = 2,
                                text_scale = 1)




while True:
    success,frame = camera.read()
    
    if not success:
        print('error reading the camera!')
        break
    

    results = model(source = frame, conf = 0.75, show = True, classes = 0)
    
    bboxes_xywh = []
    confs = []
    classes = []
    
    #detections = sv.Detections.from_yolov8(result)
    for result in results:
        boxes1 = result.boxes.xywh.tolist()
        class_ids = result.boxes.cls
        confidences = np.array(result.boxes.conf.tolist())
        class_labels = [class_id_mapping[int(class_id)] for class_id in class_ids]      
        bboxes_xywh.extend(boxes1)
        confs.extend(confidences)
        classes.extend(class_labels)
    
    bboxes_xywh = np.array(bboxes_xywh)
    confs = torch.tensor(confs)  # Convert confs to PyTorch tensor
    classes = torch.tensor(classes) 
    
    tracks = tracker.update(bboxes_xywh, confs, classes, frame)

    track_confs = [track.conf for track in tracker.tracker.tracks if track.is_confirmed() and track.time_since_update <= 1]
    track_classes = [track.class_id for track in tracker.tracker.tracks if track.is_confirmed() and track.time_since_update <= 1]
    track_ids = [track.track_id for track in tracker.tracker.tracks if track.is_confirmed() and track.time_since_update <= 1]
    track_dets = [track.to_tlbr() for track in tracker.tracker.tracks if track.is_confirmed() and track.time_since_update <= 1]
    
    
    
    #print(track_confs)
    
    if len(track_dets) > 0:
        
        detections = Detections(
                        xyxy = np.array(track_dets),
                        confidence = np.array(track_confs),
                        class_id   = np.array(track_classes).astype(int),
                        tracker_id = np.array(track_ids)
                    )

    # detections = sv.Detections.from_yolov8()    
    
    #mask = np.array([class_id in CLASS_ID for class_id in detections.class_id], dtype=bool)
    #detections.filter(mask=mask, inplace=True)

        labels = [f'{model.model.names[class_id]} ID: {track_ID}|{confidence:0.2f}' for _, confidence, class_id, track_ID in detections]
        frame =box_annotator.annotate(
            scene = frame,
            detections = detections,
            labels = labels
            )

        line_counter.trigger(detections=detections)
    line_annotator.annotate(frame, line_counter)
    cv2.imshow('video',frame)
    
    if cv2.waitKey(1) == ord("q"):
        break
        
camera.release()
cv2.destroyAllWindows()