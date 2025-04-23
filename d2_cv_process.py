import cv2
import numpy as np
import torch
from ultralytics import YOLO

def detect_objects(image_or_path, threshold:float=0.2)->[]:
    """
    Detect objects with YOLO-World
    image_or_path: can be a file path (str) or a numpy array (image).
    Returns: (list of bboxes in xyxyxyxy format, detected classes list, visualization image)
    """
    model = YOLO(r"ultralytics/runs/obb/obb_boxes2/weights/best.pt")
    # YOLOv8 的 predict 可同时处理 文件路径(str) 或 图像数组(np.ndarray)
    '''
    xyxyxyxy: four corner point pixel location. inverse direction sort.
    xyxyxyxyn: four corner normalization point pixel location.
    xyxy: the minimum and maximum point of th max circumscribe(外接) rectangle   
    xywhr: center point of the detection rectangle and distance between the center and sides.
    '''
    results = model.predict(image_or_path)

    obbs = results[0].obb
    vis_img = results[0].plot()  # Get visualized detection results

    # Extract valid detections
    max_conf_box = {}
    max_conf = 0
    for obb in obbs:
        if obb.conf.item() > threshold:  # Confidence threshold
            if max_conf < obb.conf.item():
                max_conf = obb.conf.item()
                max_conf_box["xyxyxyxy"] = obb.xyxyxyxy[0].tolist()
                max_conf_box["conf"] = obb.conf.item()
                max_conf_box["cls"] = results[0].names[obb.cls.item()]
                max_conf_box["center"] = results[0].xywhr[0].tolist()[:2]
            # valid_boxes.append({
            #     "xyxyxyxy": obb.xyxyxyxy[0].tolist(),
            #     "conf": obb.conf.item(),
            #     "cls": results[0].names[obb.cls.item()]
            # })
    x1, y1, _, _, x3, y3, _, _ = max_conf_box["xyxyxyxy"]
    max_conf_box["angle"] = compute_angle((x1, y1), (x3, y3)) # left bottom and right top.
    return max_conf_box, vis_img

def show_max_conf_box(max_conf_box, vis_img, camera_xyz):
    for k, v in max_conf_box.items():
        print(f"{k}: {v}")

    ux, uy = max_conf_box["center"]
    cv2.circle(vis_img, (ux, uy), 4, (255, 255, 255), 5)
    cv2.putText(vis_img, str(camera_xyz), (ux + 20, uy + 10), 0, 0.5, [225, 255, 255], thickness=1,
                lineType=cv2.LINE_AA)
    cv2.namedWindow('detection', flags=cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
    cv2.resizeWindow('detection', 640, 480)
    cv2.imshow('detection', vis_img)
    cv2.waitKey(1)

def compute_angle(point1, point2):
    # point1, point2 = coord[0], coord[1]
    # assume camera x-axis is forward, return angle with x-axis.
    # 计算向量
    vector = point2 - point1
    print("向量:", vector)

    vector_axis = np.array([0., 1.0])

    dot_product = np.dot(vector, vector_axis)  # 计算点积
    norm_a = np.linalg.norm(vector)  # 计算向量a的模长
    norm_b = np.linalg.norm(vector_axis)  # 计算向量b的模长
    cos_theta = dot_product / (norm_a * norm_b)
    # 由于计算误差可能导致cos_theta稍微超过[-1, 1]的范围，这里进行裁剪
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)  # 计算弧度值
    return theta



if __name__ == '__main__':
    detect = detect_objects(r'example/color.png')
    print(detect)
    # print("Segmentation result mask shape:", .shape if seg_mask is not None else None)
