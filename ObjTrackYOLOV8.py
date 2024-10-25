# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 01:03:51 2024

@author: mohan
"""

# from ultralytics import YOLO
# import cv2

# model = YOLO('yolov8s.pt')

# camera = cv2.VideoCapture(0)

# while True:
#     success, frame = camera.read()
    
#     if not success:
#         print("Error reading camera frame!")
#         break
    
#     #results = model(source=frame, conf=0.4,show=True)
#     results = model.track(frame,show=True, tracker="bytetrack.yaml")
    
#     if cv2.waitKey(1) == ord("q"):
#         break

# # Release the camera and close the window
# camera.release()
# cv2.destroyAllWindows()




# #from deep_sort.utils.parser import get_config
# from deep_sort.deep_sort import DeepSort
# from deep_sort.sort.tracker import Tracker
# import cv2
# from ultralytics import YOLO
# import random
# import numpy as np
# import torch

# colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]

# torch.cuda.set_device(0)


# deep_sort_weights = 'deep_sort/deep/checkpoint/ckpt.t7'
# tracker = DeepSort(model_path=deep_sort_weights, max_age=70)

# model = YOLO('yolov8s.pt')

# camera = cv2.VideoCapture(0)

# class_names = model.names


# while True:
#     success, frame = camera.read()
    
#     if not success:
#         print('Error reading camera!')
#         break
    
#     results = model(frame, conf=0.8)
    
#     # for result in results:
#     #     detections = []
#         # for r in result.boxes.data.tolist():
#         #     x1,y1,x2,y2, score, class_id = r
#         #     x1=int(x1)
#         #     x2=int(x2)
#         #     y1=int(y1)
#         #     y2=int(y2)
#         #     class_id = int(class_id)
#         #     detections.append([x1,y1,x2,y2])
        
#     for result in results:
#         boxes = result.boxes  # Boxes object for bbox outputs
#         probs = result.probs  # Class probabilities for classification outputs
#         cls = boxes.cls.tolist()  # Convert tensor to list
#         xyxy = boxes.xyxy
#         conf = boxes.conf
#         xywh = boxes.xywh  # box with xywh format, (N, 4)
#         for class_index in cls:
#             class_name = class_names[int(class_index)]
#             #print("Class:", class_name)

#     pred_cls = np.array(cls)
#     conf = conf.detach().cpu().numpy()
#     xyxy = xyxy.detach().cpu().numpy()
#     bboxes_xywh = xywh
#     bboxes_xywh = xywh.cpu().numpy()
#     bboxes_xywh = np.array(bboxes_xywh, dtype=float)
    
#     tracks = tracker.update(bboxes_xywh, conf, frame)
#         #tracker.update(detections,conf=0.8,)
#     for track in tracks:
#         bbox = track.bbox
#         x1, y1, x2, y2 = bbox
#         track_id = track.track_id

#         cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (colors[track_id % len(colors)]), 3)
#         cv2.putText(frame, f"{class_name}: {track_id}", (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
#                     (colors[track_id % len(colors)]), 3)
            
#     cv2.imshow("YOLOv8 Camera Detections", frame)

# # Press q to close the camera and end the code.
#     if cv2.waitKey(1) == ord("q"):
#         break

# camera.release()
# cv2.destroyAllWindows()




# the bellow code worked just fine but bounding box issue


# import cv2
# from ultralytics import YOLO
# import numpy as np
# import torch

# #from strong_sort.utils.parser import get_config
# from strong_sort.strong_sort import StrongSORT
# from strong_sort.sort.tracker import Tracker
# import os

# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# model = YOLO('yolov8s.pt')
# class_names = model.names
# #class_id_mapping = {name: idx for idx, name in enumerate(class_names)}
# class_id_mapping = {idx: name for idx, name in enumerate(class_names)}


# COLORS = [(255, 0, 0), (255, 0, 255), (0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

# strong_sort_wts = 'osnet_x0_25_msmt17.pt'
# tracker = StrongSORT(model_weights = strong_sort_wts,device = device,fp16=False, max_age= 70)

# camera = cv2.VideoCapture(0)

# while True:
#     success, frame = camera.read()
#     if not success:
#         print('Error reading camera!')
#         break
    
#     og_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
#     frame = og_frame.copy()
    
#     results = model(frame, conf=0.8, show= True)
    
#     # for result in results:
#     #     boxes = result.boxes.xyxy.tolist()
#     #     class_ids=result.boxes.cls
#     #     confs= np.array(result.boxes.conf.tolist())
#     #     #classes = [class_id_mapping[class_names[int(class_id)]] for class_id in class_ids]
#     #     #classes = np.array(classes)
#     #     classes = [class_id_mapping[int(class_id)] for class_id in class_ids]
#     for result in results:
#         boxes = result.boxes.xyxy.tolist()
#         class_ids = result.boxes.cls
#         confs = np.array(result.boxes.conf.tolist())
#         classes = [class_id_mapping[int(class_id)] for class_id in class_ids]
#         classes = torch.tensor(classes)  # Convert to PyTorch tensor
#         confs = torch.tensor(confs)  # Convert confs to PyTorch tensor
        
#     bboxes_xywh=[]
#     for box in boxes:
#         x1, y1, x2, y2 = box
#         w = x2-x1
#         h = y2-y1
#         bbox_xywh = [x1,y1,w,h]
#         bboxes_xywh.append(bbox_xywh)
    
#     bboxes_xywh=np.array(bboxes_xywh)
    
#     tracks=tracker.update(bboxes_xywh, confs, classes, og_frame)
    
#     for track in tracker.tracker.tracks:
#         track_id = track.track_id
#         hits = track.hits
#         #class_id=track.class_id
#         bbox_tlwh = track.to_tlwh()
#         x1,y1,x2,y2 = tracker._tlwh_to_xyxy(bbox_tlwh)
#         #x2=x1+w
#         #y2=y1+h
#         bbox_xywh = (x1,y1,w,h)
#         color = COLORS[track_id % len(COLORS)]
        
#         cv2.rectangle(og_frame, (int(x2),int(y2)), (int(x1),int(y1)), color, 2)
#         cv2.putText(og_frame, f"class_names[class_id]: ID: {track_id}", (int(x1), int(y1) - 5),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
#         cv2.imshow('video',og_frame)
    
#     # Press q to close the camera and end the code.
#     if cv2.waitKey(1) == ord("q"):
#         break

# camera.release()
# cv2.destroyAllWindows()







import cv2
from ultralytics import YOLO
import numpy as np
import torch

#from strong_sort.utils.parser import get_config
from strong_sort.strong_sort import StrongSORT
from strong_sort.sort.tracker import Tracker
# from torchreid import models
# from torchreid.utils import feature_extractor
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = YOLO('yolov8l.pt')
class_names = model.names
#class_id_mapping = {name: idx for idx, name in enumerate(class_names)}
class_id_mapping = {idx: name for idx, name in enumerate(class_names)}


COLORS = [(255, 0, 0), (255, 0, 255), (0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

strong_sort_wts = 'osnet_x0_25_msmt17.pt'
#strong_sort_wts = 'osnet_x1_0_imagenet.pt'
tracker = StrongSORT(model_weights = strong_sort_wts,device = device,fp16=False, max_age= 1000)

camera = cv2.VideoCapture(0)
#camera = cv2.VideoCapture("http://192.168.0.61:8080/video")

highlight_track_id = None

while True:
    success, frame = camera.read()
    if not success:
        print('Error reading camera!')
        break
    
    #og_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #frame = og_frame.copy()
    og_frame=frame.copy()
    results = model(frame, conf=0.5, show=True)
    
    bboxes_xywh = []
    confs = []
    classes = []
    
    for result in results:
        boxes = result.boxes.xyxy.tolist()
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
        x1,y1,x2,y2=bbox_tlwh
        #x1, y1, x2, y2 = tracker._tlwh_to_xyxy(bbox_tlwh)
        w=y2-y1
        h=y2-y1
        
        
        track_id = track.track_id
        class_id = track.class_id
        conf = track.conf

        #color = COLORS[track_id % len(COLORS)]
        color = (0, 255, 0) if track_id == highlight_track_id else (255, 0, 0)

        #cv2.rectangle(og_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.rectangle(og_frame, (int(x1+(x2)/2),int(y1+(y2)/2)),(int(x2),int(y2)), color,2)
        cv2.putText(og_frame, f"{class_names[class_id.item()]}: ID: {track_id}", (int(x1+(x2)/2), int(y1+(y2)/2) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    cv2.imshow('video', og_frame)
    
    # Press q to close the camera and end the code.
    if cv2.waitKey(1) == ord("q"):
        break
    elif cv2.waitKey(1) == ord('t'):
        highlight_track_id = int(input("Enter the tracking ID to highlight in green: "))


camera.release()
cv2.destroyAllWindows()


        

# import cv2
# from ultralytics import YOLO
# import numpy as np
# import torch

# from strong_sort.strong_sort import StrongSORT

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# model = YOLO('yolov8s.pt')
# class_names = model.names
# class_id_mapping = {idx: name for idx, name in enumerate(class_names)}

# COLORS = [(255, 0, 0), (255, 0, 255), (0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

# strong_sort_wts = 'osnet_x0_25_msmt17.pt'
# tracker = StrongSORT(model_weights=strong_sort_wts, device=device, fp16=False, max_age=70)

# camera = cv2.VideoCapture(0)

# while True:
#     success, frame = camera.read()
#     if not success:
#         print('Error reading camera!')
#         break

#     og_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     frame = og_frame.copy()

#     results = model(frame, conf=0.8, show=True)

#     bboxes_xywh = []
#     confs = []
#     classes = []

#     for result in results:
#         boxes = result.boxes.xyxy.tolist()
#         class_ids = result.boxes.cls
#         confidences = np.array(result.boxes.conf.tolist())
#         class_labels = [class_id_mapping[int(class_id)] for class_id in class_ids]

#         bboxes_xywh.extend(boxes)
#         confs.extend(confidences)
#         classes.extend(class_labels)

#     bboxes_xywh = np.array(bboxes_xywh)
#     confs = torch.tensor(confs)
#     classes = torch.tensor(classes)

#     tracks = tracker.update(bboxes_xywh, confs, classes, og_frame)

#     for track in tracker.tracker.tracks:
#         if not track.is_confirmed() or track.time_since_update > 1:
#             continue

#         bbox_tlwh = track.to_tlwh()
#         x1, y1, w, h = map(int, bbox_tlwh)
#         x2, y2 = x1 + w, y1 + h

#         track_id = track.track_id
#         class_id = track.class_id
#         conf = track.conf

#         color = COLORS[track_id % len(COLORS)]

#         cv2.rectangle(og_frame, (x1, y1), (x2, y2), color, 2)
#         cv2.putText(og_frame, f"{class_names[class_id.item()]}: ID: {track_id}", (x1, y1 - 5),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#     cv2.imshow('video', og_frame)

#     if cv2.waitKey(1) == ord("q"):
#         break

# camera.release()
# cv2.destroyAllWindows()
        
        


