import cv2

from ultralytics import YOLO

# 加载模型
model = YOLO(model=r"D:\project\ultralytics-main\ultralytics\runs\obb\obb_boxes2\weights\best.pt")

# 摄像头编号
camera_no = 0

# 打开摄像头
cap = cv2.VideoCapture(camera_no)

while cap.isOpened():
    # 获取图像
    res, frame = cap.read()
    # 如果读取成功
    if res:
        # 正向推理
        results = model(frame)

        # 检查是否有检测结果
        if hasattr(results[0], 'obb') and hasattr(results[0].obb, 'xyxyxyxy') and len(results[0].obb.xyxyxyxy) > 0:
            # 遍历所有检测框
            for i, coord in enumerate(results[0].obb.xyxyxyxy):
                # 计算中心点坐标
                center_x = int(sum(point[0] for point in coord) / 4)
                center_y = int(sum(point[1] for point in coord) / 4)
                center = (center_x, center_y)
                conf = results[0].obb.conf[i]

                # 绘制中心点
                cv2.circle(frame, center, 3, (0, 0, 255), 3)

                # 在图像上添加文本
                cv2.putText(frame, f"Center: {center} {conf:.3f}", (center[0] + 10, center[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

                # 打印详细信息
                print("四个顶点的坐标为：")
                for i, (x, y) in enumerate(coord):
                    print(f"顶点{i + 1}的坐标为：({x}, {y})")
                print(f"中心点的坐标为：({center_x}, {center_y})")
        else:
            print("未检测到任何目标")
            # 可以选择在图像上添加提示文本
            cv2.putText(frame, "No targets detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2,
                        lineType=cv2.LINE_AA)


        # annotated_frame = results[0].plot()

        # 显示图像
        cv2.imshow(winname="YOLOV11obb", mat=frame)

        # 按ESC退出
        if cv2.waitKey(1) == 27:
            break

    else:
        break

# 释放链接
cap.release()
# 销毁所有窗口
cv2.destroyAllWindows()

