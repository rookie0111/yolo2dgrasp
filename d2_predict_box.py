import math
import yaml
import sys
import os
from scipy.spatial.transform import Rotation as R
import cv2
import numpy as np
import pyrealsense2 as rs
from torch import tensor
import lebai_sdk

from ultralytics import YOLO
from d2_utils import (
    convert_new, eulerAnglesToRotationMatrix,
    load_hand_eye_calibration, convert_to_pose_dict,
    convert_pose_to_robot_units,
)
from d2_cv_process import detect_objects, show_max_conf_box

# 初始化 RealSense 相机
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

global color_img, depth_img, robot, first_run, depth_frame, depth_colormap
color_img = None
depth_img = None
depth_colormap = None
depth_frame = None
robot = None
first_run = True  # 新增首次运行标志
# 手眼标定外参
rotation_matrix = None
translation_vector = None
def get_3d_camera_coordinate(depth_pixel, aligned_depth_frame):
    '''
    acquire a 3D camera coordinate frame
    '''
    x = depth_pixel[0]
    y = depth_pixel[1]
    dis = aligned_depth_frame.get_distance(x, y)
    depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
    camera_coordinate = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel, dis)
    return dis, camera_coordinate


def run_inference()->(tensor, tensor):
    max_conf_box, vis_img = detect_objects(color_img, 0.1) # detect info
    depth_pixel = max_conf_box["center"] # center point
    dis, camera_xyz = get_3d_camera_coordinate(depth_pixel, depth_frame) # distance and camera_xyz
    show_max_conf_box(max_conf_box, vis_img, camera_xyz)  # show max box
    translation, rotation_mat_3x3 = camera_xyz ,eulerAnglesToRotationMatrix([0., 0., max_conf_box["theta"]])
    return translation, rotation_mat_3x3

def test_grasp():
    global color_img, depth_img, robot, first_run

    if rotation_matrix is None or translation_vector is None:
        print("[ERROR] 手眼标定参数未正确加载")
        return
    if color_img is None or depth_img is None:
        print("[WARNING] Waiting for image data...")
        return

    # 图像处理部分
    translation, rotation_mat_3x3 = run_inference(
        color_img,
        depth_img,
    )
    print(f"[DEBUG] 预测结果 - 平移: {translation}, 旋转矩阵:\n{rotation_mat_3x3}")

    # error_code, joints, current_pose, arm_err_ptr, sys_err_ptr = robot.Get_Current_Arm_State()
    current_pose_dict = robot.get_kin_data()['actual_tcp_pose']
    current_pose = [
        current_pose_dict['x'],
        current_pose_dict['y'],
        current_pose_dict['z'],
        current_pose_dict['rx'],
        current_pose_dict['ry'],
        current_pose_dict['rz']
    ]
    print("\n[DEBUG] 当前末端位姿:", current_pose)

    print("\n=== 输入参数验证 ===")
    print(f"1. translation 类型: {type(translation)}, 值: {translation}")
    print(f"2. rotation_mat_3x3 类型: {type(rotation_mat_3x3)}, 值:\n{rotation_mat_3x3}")
    print(f"3. current_pose 类型: {type(current_pose)}, 值: {current_pose}")
    print(f"4. rotation_matrix 类型: {type(rotation_matrix)}, 值:\n{rotation_matrix}")
    print(f"5. translation_vector 类型: {type(translation_vector)}, 值: {translation_vector}")

    if any(x is None for x in [translation, rotation_mat_3x3, current_pose, rotation_matrix, translation_vector]):
        print("[ERROR] 存在未初始化的参数")
        return

    base_pose = convert_new(
        translation,
        rotation_mat_3x3,
        current_pose,
        rotation_matrix,
        translation_vector
    )
    base_pose = convert_pose_to_robot_units(base_pose)
    print("[DEBUG] 基坐标系抓取位姿:", base_pose)
    # 验证数据有效性
    if np.any(np.isnan(translation)) or np.any(np.isnan(rotation_mat_3x3)):
        print("[ERROR] Grasp 预测结果包含 NaN")
        return

    # 首次运行只计算不执行
    if first_run:
        print("[INFO] 首次运行模拟完成，准备正式执行")
        first_run = False
        return  # 直接返回不执行后续动作

    # 正式执行部分
    try:
        # 验证基础位姿
        if not isinstance(base_pose, (list, np.ndarray)) or len(base_pose) != 6:
            raise ValueError("基坐标系抓取位姿格式无效")

        base_pose_np = np.array(base_pose, dtype=float)
        if np.any(np.isnan(base_pose_np)) or np.any(np.isinf(base_pose_np)):
            print("[ERROR] 基坐标系抓取位姿包含无效值，跳过本次执行")
            return

        base_xyz = base_pose_np[:3]
        base_rxyz = base_pose_np[3:]

        # 预抓取计算
        pre_grasp_offset = 0.1
        pre_grasp_pose = np.array(base_pose, dtype=float).copy()

        # 验证欧拉角
        try:
            rotation_mat = R.from_euler('ZYX', pre_grasp_pose[3:][::-1]).as_matrix()
        except Exception as e:
            print(f"[ERROR] 欧拉角转换失败: {e}")
            return

        z_axis = rotation_mat[:, 2]
        pre_grasp_pose[:3] -= z_axis * pre_grasp_offset

        # 运动控制前的验证
        grasp_pose = np.concatenate([base_xyz, base_rxyz]).tolist()

        # intigrasp_pose = {'x': -0.093, 'y': 0.38, 'z': 0.204,
        #                   'rx': 108,
        #                   'ry': 0, 'rz': 178}
        intigrasp_pose = {-96, -88, 90, 268, -88, -103}

        new_joint_pose = robot.kinematics_inverse(intigrasp_pose)
        robot.movej(new_joint_pose, a=1, v=1)
        robot.wait_move()

        # 控制夹爪
        print("[INFO] 打开夹爪")
        robot.set_claw(50, 100)
        robot.wait_move()

        # 执行运动
        pre_grasp_dict = convert_to_pose_dict(pre_grasp_pose.tolist())
        new_joint_pose = robot.kinematics_inverse(pre_grasp_dict)
        robot.movej(new_joint_pose, a=1, v=1)
        robot.wait_move()

        grasp_dict = convert_to_pose_dict(grasp_pose)
        new_joint_pose = robot.kinematics_inverse(grasp_dict)
        robot.movej(new_joint_pose, a=1, v=1)
        robot.wait_move()

        # 夹取物体
        robot.set_claw(50, 0)
        robot.wait_move()

        # 抬起
        pose = {'x': (current_pose_dict['x'] + 120) / 1000, 'y': -(current_pose_dict[1]) / 1000, 'z': 0.3,
                'rx': math.radians(180),
                'ry': math.radians(0), 'rz': math.radians(0)}
        # print("pose:", pose)
        new_joint_pose = robot.kinematics_inverse(pose)
        robot.movej(new_joint_pose, a=1, v=1)
        robot.wait_move()

    except Exception as e:
        print(f"[ERROR] 运动异常: {str(e)}")
        print(f"[DEBUG] 详细错误信息: {type(e).__name__}: {str(e)}")


def callback(color_frame, depth_frame, depth_colormap):
    global color_img, depth_img
    scaling_factor_x = 1
    scaling_factor_y = 1

    color_img = cv2.resize(
        color_frame, None,
        fx=scaling_factor_x,
        fy=scaling_factor_y,
        interpolation=cv2.INTER_AREA
    )
    depth_img = cv2.resize(
        depth_frame, None,
        fx=scaling_factor_x,
        fy=scaling_factor_y,
        interpolation=cv2.INTER_NEAREST
    )

    depth_colormap = cv2.resize(
        depth_colormap, None,
        fx=scaling_factor_x,
        fy=scaling_factor_y,
        interpolation=cv2.INTER_NEAREST
    )

    if color_img is not None and depth_img is not None:
        test_grasp()

def displayD435():
    global first_run

    try:
        profile = pipeline.start(config)

        color_sensor = profile.get_device().query_sensors()[1]
        color_sensor.set_option(rs.option.enable_auto_exposure, 1)

        while True:
            frames = pipeline.wait_for_frames()
            frames = frames.process(frames)

            if not frames: continue

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame: continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            depth_colormap = (cv2.applyColorMap
                              (cv2.convertScaleAbs(depth_img, alpha=0.03), cv2.COLORMAP_JET))

            callback(color_image, depth_image, depth_colormap)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
def main():
    global robot, first_run
    robot_ip = "10.20.17.1"
    # logger.info(f'robot_ip:{robot_ip}')

    try:
        with open("config.yaml", 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
        # ROBOT_TYPE = data.get("ROBOT_TYPE")
        rot_mat, trans_vec, camera_matrix, distortion = load_hand_eye_calibration()
        rotation_matrix = rot_mat  # 保存到全局变量
        translation_vector = trans_vec  # 保存到全局变量

        robot = lebai_sdk.connect(robot_ip, False)
        robot.start_sys()
        tcp = {'x': 0, 'y': 0, 'z': 0.2, 'rz': 0, 'ry': 0, 'rx': 0}
        robot.set_tcp(tcp)

        # 重置首次运行标志
        first_run = True

        # 打印调试信息
        print("\n=== 初始化参数验证 ===")
        print(f"rotation_matrix:\n{rotation_matrix}")
        print(f"translation_vector: {translation_vector}")

        # 启动相机显示
        displayD435()

    except Exception as e:
        print(f"[ERROR] 程序初始化失败: {e}")
        if robot:
            try:
                robot.disconnect()
            except:
                pass
        sys.exit()

if __name__ == '__main__':
    main()