from ultralytics import YOLO
import cv2
import numpy as np

# Load a model
# model = YOLO("yolo11n.pt")  # load an official model
model = YOLO(r"D:\project\ultralytics-main\ultralytics\runs\obb\obb_boxes2\weights\best.pt")  # load a custom model

# Predict with the model
# results = model(r"D:\edgedownoad\11\images\val\WIN_20250304_17_48_41_Pro.jpg", save=True)  # predict on an image
# print(type(results))
# results = model.predict(source=r'D:\edgedownoad\11\images\val',
#               project='runs/detect',
#               name='exp',
#               save=True,)

results = model(r"D:\edgedownoad\11\images\val\WIN_20250304_17_48_39_Pro.jpg", mode="predict", save=True)

# box = results.

print(type(results))


