from ultralytics import YOLO

# Load a model
model = YOLO("yolo11s-obb.pt")  # load a pretrained model (recommended for training)

# Train the model with MPS
model.train(data="boxes.yaml", epochs=5,batch=4, imgsz=640, device="mps", name='obb_boxes')
model.val()