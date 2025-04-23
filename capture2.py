import cv2
import numpy as np
from ultralytics import YOLO
import time
import os
import yaml


class CameraProcessor:
    def __init__(self, camera_params_path, model_path, yaml_path):
        """初始化相机处理器"""
        # 加载相机参数
        self.load_camera_params(camera_params_path)

        # 加载类别信息
        self.load_class_names(yaml_path)
        print(f"类别名称: {self.class_names}")

        # 初始化YOLO模型
        try:
            self.model = YOLO(model_path)
            print("YOLO模型加载成功")
        except Exception as e:
            raise Exception(f"YOLO模型加载失败: {str(e)}")

        # 为不同类别设置不同的颜色
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(len(self.class_names), 3)).tolist()

        # 棋盘格参数
        self.CHECKERBOARD = (6, 9)  # 根据实际标定板修改
        self.square_size = 30.0  # 实际方格尺寸(mm)

        self.latest_results = []

    def load_camera_params(self, path):
        """加载相机标定参数"""
        try:
            data = np.load(path)
            self.camera_matrix = data['camera_matrix']
            self.dist_coeffs = data['dist_coeffs']
            print("相机参数加载成功")
        except Exception as e:
            raise Exception(f"加载相机参数失败: {str(e)}")

    def load_class_names(self, yaml_path):
        """从YAML文件加载类别名称"""
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                if 'names' in data:
                    self.class_names = data['names']
                    print(f"成功加载 {len(self.class_names)} 个类别:")
                    for i, name in enumerate(self.class_names):
                        print(f"  {i}: {name}")
                else:
                    raise Exception("YAML文件中未找到'names'字段")
        except Exception as e:
            raise Exception(f"加载类别信息失败: {str(e)}")

    def undistort_image(self, img):
        """图像去畸变"""
        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h))

        # 去畸变
        dst = cv2.undistort(img, self.camera_matrix, self.dist_coeffs, None, newcameramtx)

        # 裁剪图像
        x, y, w, h = roi
        if roi != (0, 0, 0, 0):  # 确保ROI有效
            dst = dst[y:y + h, x:x + w]
        return dst, newcameramtx

    def get_perspective_transform(self, img):
        """获取透视变换矩阵"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray,
                                                 (self.CHECKERBOARD[0] - 1, self.CHECKERBOARD[1] - 1), None)

        if ret:
            # 提高角点精度
            # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            # corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            # 重新组织角点
            # corners_reshaped = corners2.reshape(-1, 2)
            corners_reshaped = corners.reshape(-1, 2)
            # 选取棋盘四个顶角
            p1 = corners_reshaped[0]  # 左上角
            p2 = corners_reshaped[self.CHECKERBOARD[0] - 2]  # 右上角
            p3 = corners_reshaped[-(self.CHECKERBOARD[0] - 1)]  # 左下角
            p4 = corners_reshaped[-1]  # 右下角

            # 图像中的四个角点
            image_points = np.float32([p1, p2, p3, p4])

            # 世界坐标（整个棋盘的四个角）
            world_points = np.float32([
                [0, 0],
                [(self.CHECKERBOARD[0] - 2) * self.square_size, 0],
                [0, (self.CHECKERBOARD[1] - 2) * self.square_size],
                [(self.CHECKERBOARD[0] - 2) * self.square_size, (self.CHECKERBOARD[1] - 2) * self.square_size]
            ])

            # 计算透视变换矩阵
            M = cv2.getPerspectiveTransform(image_points, world_points)

            # np.save('perspective_matrix.npy', M)
            # print("透视矩阵已保存为 'perspective_matrix.npy'")
            # 画出选取的四个点
            img_display = img.copy()
            cv2.drawChessboardCorners(img_display,
                                      (self.CHECKERBOARD[0] - 1, self.CHECKERBOARD[1] - 1),
                                      corners, ret)

            # 画角点连线
            cv2.line(img_display, tuple(map(int, p1)), tuple(map(int, p2)), (0, 0, 255), 2)
            cv2.line(img_display, tuple(map(int, p1)), tuple(map(int, p3)), (0, 255, 0), 2)
            cv2.line(img_display, tuple(map(int, p2)), tuple(map(int, p4)), (128, 128, 128), 2)
            cv2.line(img_display, tuple(map(int, p3)), tuple(map(int, p4)), (128, 128, 128), 2)

            # 标注坐标系
            cv2.circle(img_display, tuple(map(int, p1)), 5, (255, 0, 0), -1)
            cv2.putText(img_display, "O(0,0)", (int(p1[0]) - 30, int(p1[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            cv2.putText(img_display, "X", (int(p2[0]) + 10, int(p2[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.putText(img_display, "Y", (int(p3[0]), int(p3[1]) + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow('Corner Detection', img_display)
            cv2.waitKey(0)  # 确保窗口不会一闪而过
            cv2.destroyAllWindows()

            return M, True

        return None, False

    def load_perspective_transform(self, filepath='perspective_matrix.npy'):
        """加载已保存的透视变换矩阵"""
        try:
            M = np.load(filepath)
            print(f"透视矩阵已从 '{filepath}' 成功加载")
            return M, True
        except FileNotFoundError:
            print(f"文件 '{filepath}' 未找到，请先进行标定")
            return None, False

    def get_or_load_perspective_transform(self, img, force_recalibrate=False):
        """
        如果不强制重标定，则优先从文件加载透视矩阵；
        否则重新标定并保存。
        """
        if not force_recalibrate:
            M, status = self.load_perspective_transform()
            if status:
                return M, True

        # 如果需要标定或加载失败，则执行标定
        M, status = self.get_perspective_transform(img)
        if status:
            np.save('perspective_matrix.npy', M)
            print("透视矩阵已保存为 'perspective_matrix.npy'")
        return M, status

    def draw_world_coordinate_system(self, img, M):
        """
        在实时检测画面中绘制世界坐标系
        """
        # 定义世界坐标系中的点（单位：mm）
        axis_length = 150  # 增加长度使坐标轴更明显
        world_points = np.float32([
            [0, 0],  # 原点
            [axis_length, 0],  # X轴端点（水平向右）
            [0, axis_length]  # Y轴端点（垂直向下）
        ])

        # 使用透视变换矩阵的逆矩阵将世界坐标转换为图像坐标
        M_inv = np.linalg.inv(M)

        # 转换为齐次坐标
        world_points_h = np.vstack((world_points.T, np.ones(3)))
        image_points_h = M_inv @ world_points_h

        # 转换回非齐次坐标
        image_points = (image_points_h[:2] / image_points_h[2]).T
        points = image_points.astype(np.int32)

        # 绘制坐标轴
        origin = tuple(points[0])
        x_end = tuple(points[1])
        y_end = tuple(points[2])

        # 创建半透明叠加层
        overlay = img.copy()

        # 绘制网格线（可选）
        grid_size = 30  # 30mm网格
        for i in range(1, 10):  # 绘制10条网格线
            # 水平网格线的世界坐标
            h_grid = np.float32([[0, i * grid_size], [axis_length, i * grid_size]])
            # 垂直网格线的世界坐标
            v_grid = np.float32([[i * grid_size, 0], [i * grid_size, axis_length]])

            # 转换到图像坐标
            h_grid_h = np.vstack((h_grid.T, np.ones(2)))
            v_grid_h = np.vstack((v_grid.T, np.ones(2)))

            h_img = (M_inv @ h_grid_h)[:2] / (M_inv @ h_grid_h)[2]
            v_img = (M_inv @ v_grid_h)[:2] / (M_inv @ v_grid_h)[2]

            # 绘制网格线
            cv2.line(overlay, tuple(h_img[:, 0].astype(int)), tuple(h_img[:, 1].astype(int)),
                     (128, 128, 128), 1)
            cv2.line(overlay, tuple(v_img[:, 0].astype(int)), tuple(v_img[:, 1].astype(int)),
                     (128, 128, 128), 1)

        # X轴（红色）
        cv2.arrowedLine(overlay, origin, x_end, (0, 0, 255), 2, tipLength=0.1)
        # Y轴（绿色）
        cv2.arrowedLine(overlay, origin, y_end, (0, 255, 0), 2, tipLength=0.1)

        # 标注坐标轴
        cv2.putText(overlay, "X",
                    (x_end[0] + 10, x_end[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(overlay, "Y",
                    (y_end[0], y_end[1] + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 标注原点
        cv2.circle(overlay, origin, 5, (255, 0, 0), -1)
        cv2.putText(overlay, "O(0,0)",
                    (origin[0] - 30, origin[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # 添加网格刻度（每30mm一个刻度）
        for i in range(1, 6):
            # X轴刻度
            x_tick_world = np.float32([[i * grid_size, 0], [i * grid_size, -10]])
            x_tick_h = np.vstack((x_tick_world.T, np.ones(2)))
            x_tick_img = (M_inv @ x_tick_h)[:2] / (M_inv @ x_tick_h)[2]
            x_tick_pos = tuple(x_tick_img[:, 0].astype(int))
            cv2.putText(overlay, f"{i * grid_size}",
                        (x_tick_pos[0] - 10, x_tick_pos[1] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            # Y轴刻度
            y_tick_world = np.float32([[0, i * grid_size], [-10, i * grid_size]])
            y_tick_h = np.vstack((y_tick_world.T, np.ones(2)))
            y_tick_img = (M_inv @ y_tick_h)[:2] / (M_inv @ y_tick_h)[2]
            y_tick_pos = tuple(y_tick_img[:, 0].astype(int))
            cv2.putText(overlay, f"{i * grid_size}",
                        (y_tick_pos[0] - 25, y_tick_pos[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # 将坐标系以半透明方式添加到原图
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        return img

    def detect_objects(self, img):
        """目标检测"""
        try:
            results = self.model(img)
            detections = []

            # 确保 results 不为空
            if results and len(results) > 0:
                result = results[0]  # 获取第一个结果
                if hasattr(results[0], 'obb') and hasattr(results[0].obb, 'xyxyxyxy') and len(
                        results[0].obb.xyxyxyxy) > 0:
                    for i, coord in enumerate(results[0].obb.xyxyxyxy):
                        # for box in boxes:
                        try:
                            # x1, y1, x2, y2 = map(int, box.xyxy[0])
                            # confidence = float(box.conf[0])
                            # class_id = int(box.cls[0])
                            # acquire angle between axis and one side. from left bottom begin.
                            # point1, point2 = coord[0], coord[1]
                            #
                            # # 计算向量
                            # vector = point2 - point1
                            # print("向量:", vector)
                            #
                            # vector_axis = np.array([1.0, 0.])
                            #
                            # dot_product = np.dot(vector, vector_axis)  # 计算点积
                            # norm_a = np.linalg.norm(vector)  # 计算向量a的模长
                            # norm_b = np.linalg.norm(vector_axis)  # 计算向量b的模长
                            # cos_theta = dot_product / (norm_a * norm_b)
                            # # 由于计算误差可能导致cos_theta稍微超过[-1, 1]的范围，这里进行裁剪
                            # cos_theta = np.clip(cos_theta, -1.0, 1.0)
                            # theta = np.arccos(cos_theta)  # 计算弧度值

                            # 确保 class_id 在有效范围内
                            # if 0 <= class_id < len(self.class_names):
                            # h_camera = 73
                            # h_object = 15
                            center_x = int(sum(point[0] for point in coord) / 4)
                            center_y = int(sum(point[1] for point in coord) / 4)
                            # center_x = int(center_x * ( (h_camera - h_object) / h_camera))
                            # center_y = int(center_y * ( (h_camera - h_object) / h_camera))
                            conf = results[0].obb.conf[i]
                            box = results[0].obb.xyxyxyxy[i]
                            # angle = result.obb.theta[0]
                            detections.append({
                                'box': (box[0], box[1], box[2], box[3]),
                                'center': (center_x, center_y),
                                'confidence': conf,
                                'class_name': self.class_names[0],
                                'color': self.colors[0],
                                # 'angle': theta  # 弧度制
                            })
                        except Exception as e:
                            print(f"处理检测框时出错: {str(e)}")
                            continue
            # self.latest_results = detections  # 保存检测结果
            return detections
        except Exception as e:
            print(f"目标检测过程出错: {str(e)}")
            return []

    def compute_angle(self, point1, point2):
        # point1, point2 = coord[0], coord[1]

        # 计算向量
        vector = point2 - point1
        print("向量:", vector)

        vector_axis = np.array([1.0, 0.])

        dot_product = np.dot(vector, vector_axis)  # 计算点积
        norm_a = np.linalg.norm(vector)  # 计算向量a的模长
        norm_b = np.linalg.norm(vector_axis)  # 计算向量b的模长
        cos_theta = dot_product / (norm_a * norm_b)
        # 由于计算误差可能导致cos_theta稍微超过[-1, 1]的范围，这里进行裁剪
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)  # 计算弧度值
        return theta

    def pixel_to_world(self, pixel_point, M):
        """像素坐标转换为世界坐标"""
        try:
            if M is None:
                raise ValueError("透视变换矩阵未初始化")

            # 转换为齐次坐标
            pixel_h = np.array([pixel_point[0], pixel_point[1], 1])

            # 使用透视变换矩阵进行转换
            world_h = M @ pixel_h

            # 转换回非齐次坐标
            if world_h[2] != 0:
                world_point = world_h[:2] / world_h[2]

                h_camera = 74
                h_object = 18

                world_point = world_point * ((h_camera - h_object) / h_camera)
                # self.latest_results['center'] = world_point
                # center_x = int(sum(point[0] for point in coord) / 4)
                # center_y = int(sum(point[1] for point in coord) / 4)
                # center_x = int(center_x * ( (h_camera - h_object) / h_camera))
                # center_y = int(center_y * ( (h_camera - h_object) / h_camera))
                return world_point
            else:
                raise ValueError("透视变换除零错误")
        except Exception as e:
            print(f"坐标转换错误: {str(e)}")
            return np.array([0, 0])

    def get_object_coordinates(self, img, M):
        """获取物体的世界坐标"""
        try:
            if M is None:
                print("警告: 透视变换矩阵未初始化")
                return []

            # 目标检测
            detections = self.detect_objects(img)
            if not detections:
                return []

            # 计算世界坐标
            result = []
            for det in detections:
                if det['confidence'] > 0.5:
                    try:
                        world_pos = self.pixel_to_world(det['center'], M)
                        point1 = self.pixel_to_world(det['box'][0], M)
                        point2 = self.pixel_to_world(det['box'][1], M)
                        theta = self.compute_angle(point1, point2)
                        det['world_pos'] = world_pos
                        det['angle'] = theta
                        print("confidence > 0.5")
                        result.append(det)
                    except Exception as e:
                        print(f"计算世界坐标时出错: {str(e)}")
                        continue
            self.latest_results = result
            return result
        except Exception as e:
            print(f"获取物体坐标时出错: {str(e)}")
            return []


def init_camera():
    """初始化相机"""
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise Exception("无法打开相机！")

    # 设置分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    return cap


def draw_detection_info(frame, detection, world_pos):
    """绘制检测信息"""
    # x1, y1, x2, y2 = detection['box']
    # color = detection['color']
    # class_name = detection['class_name']
    # confidence = detection['confidence']
    center = detection['center']
    cv2.circle(frame, center, 3, (0, 0, 255), 3)
    conf = detection['confidence']
    # # 绘制边界框
    # cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # 准备显示文本
    # 在图像上添加文本
    cv2.putText(frame, f"Center: {center} {conf:.3f}", (center[0] + 10, center[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

    # 打印详细信息
    # print("四个顶点的坐标为：")
    # for i, (x, y) in enumerate(coord):
    #     print(f"顶点{i + 1}的坐标为：({x}, {y})")
    print(f"中心点的坐标为：({center[0]}, {center[1]})")

    # # 计算文本大小
    # class_text_size = cv2.getTextSize(class_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
    # pos_text_size = cv2.getTextSize(pos_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
    #
    # # 绘制文本背景
    # cv2.rectangle(frame,
    #               (x1, y1 - 45),
    #               (x1 + max(class_text_size[0], pos_text_size[0]), y1),
    #               color, -1)
    #
    # # 绘制文本
    # cv2.putText(frame, class_text, (x1, y1 - 25),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    # cv2.putText(frame, pos_text, (x1, y1 - 5),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


# user_choice = '3'
def main():
    try:
        # 设置路径
        camera_params_path = 'camera_params.npz'
        model_path = 'box.pt'
        yaml_path = 'box_data.yaml'

        # 初始化相机处理器
        processor = CameraProcessor(camera_params_path, model_path, yaml_path)

        # try:
        #     # 初始化相机
        cap = init_camera()
        # except Error as e:
        #     print(f"取不到相机{str(e)}")
        # 等待相机稳定
        print("等待相机稳定...")
        time.sleep(2)

        # 获取透视变换矩阵
        # print("\n=== 透视变换矩阵获取步骤 ===")
        # print("1. 将标定板平放在工作平面上")
        # print("2. 确保标定板完整可见")
        # print("3. 等待系统检测角点...")
        # print("注意：左上角为坐标原点(0,0)")
        # print("警告：此后不要移动相机位置！")
        print("\n=== 透视矩阵设置 ===")
        print("请选择：")
        print("1. 重新标定 (需要放置棋盘)")
        print("2. 使用上次标定结果 (跳过棋盘)")
        # if user_choice == '3':
        user_choice = input("请输入序号 (1 或 2): ").strip()
        if user_choice == "1":
            force_recalibrate = True
            print("您选择了重新标定，请确保棋盘放置好...")
        elif user_choice == "2":
            force_recalibrate = False
            print("您选择了使用上次标定结果")
        else:
            print("无效输入，默认重新标定")
            force_recalibrate = True
            # user_choice == '3'
        # elif user_choice == "1":
        #     force_recalibrate = True
        #     print("您选择了重新标定，请确保棋盘放置好...")
        # force_recalibrate = False
        M = None
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            # 去畸变
            # undist_frame, _ = processor.undistort_image(frame)
            undist_frame = frame

            # 获取透视变换矩阵
            # M, found = processor.get_perspective_transform(undist_frame.copy())
            M, found = processor.get_or_load_perspective_transform(undist_frame, force_recalibrate)
            if found:
                print("\n✓ 成功获取透视变换矩阵！")
                print("现在可以：")
                print("1. 移除标定板")
                print("2. 在工作区域内放置待检测物体")
                print("3. 按空格键进行检测")
                print("4. 按ESC键退出程序")
                cv2.waitKey(0)
                break

            cv2.putText(undist_frame, "Place chessboard in view", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Setup', undist_frame)

            if cv2.waitKey(1) & 0xFF == 27:
                cap.release()
                cv2.destroyAllWindows()
                return

        print("\n=== 开始检测 ===")
        print("- 空格键：进行检测")
        print("- ESC键：退出程序")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("无法读取摄像头画面")
                continue

            try:
                # 去畸变
                # undist_frame, _ = processor.undistort_image(frame)
                # if undist_frame is None:
                #     print("图像去畸变失败")
                #     continue
                undist_frame = frame.copy()
                # 显示实时画面
                display_frame = undist_frame.copy()

                # 绘制世界坐标系（半透明）
                if M is not None:
                    display_frame = processor.draw_world_coordinate_system(display_frame, M)

                cv2.putText(display_frame, "Press SPACE to detect", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # 检查按键
                key = cv2.waitKey(1) & 0xFF

                # 空格键触发检测
                if key == 32:
                    try:
                        # 获取目标坐标
                        object_coordinates = processor.get_object_coordinates(undist_frame, M)

                        if object_coordinates and len(object_coordinates) > 0:
                            # 绘制检测结果
                            for coord in object_coordinates:
                                try:
                                    # 绘制检测信息
                                    draw_detection_info(display_frame, coord, coord['world_pos'])

                                    # 打印详细信息
                                    print(f"\n检测到物体:")
                                    print(f"- 类型: {coord['class_name']}")
                                    print(
                                        f"- 世界坐标: X={coord['world_pos'][0]:.1f}mm, Y={coord['world_pos'][1]:.1f}mm")
                                    print(f"- 置信度: {coord['confidence']:.2f}, 与x轴的夹角：{coord['angle']}")
                                except Exception as e:
                                    print(f"绘制检测结果时出错: {str(e)}")
                                    continue
                        else:
                            print("\n未检测到物体")
                            cv2.putText(display_frame, "No objects detected", (10, 60),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    except Exception as e:
                        print(f"处理检测结果时出错: {str(e)}")

                # ESC键退出
                elif key == 27:
                    break

                # 显示结果
                cv2.imshow('Detection', display_frame)

            except Exception as e:
                print(f"主循环处理帧时出错: {str(e)}")
                continue

    except Exception as e:
        print(f"程序运行错误: {str(e)}")
    finally:
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()

    return processor


if __name__ == '__main__':
    processor = main()