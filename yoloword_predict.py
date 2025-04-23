from ultralytics import YOLOWorld

# Initialize a YOLO-World model
model = YOLOWorld("D:\edgedownoad\yolo_world_v2_s_obj365v1_goldg_pretrain-55b943ea.pth")  # or select yolov8m/l-world.pt for different sizes

model.set_classes(["shoes",])
# Execute inference with the YOLOv8s-world model on the specified image
results = model.predict("ultralytics/assets/bus.jpg")

# Show results
results[0].show()