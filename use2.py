from capture2 import main
import time
import threading
from queue import Queue
import copy
import lebai_sdk
import math
# 创建一个全局的队列用于存储最新的检测结果
latest_results_queue = Queue(maxsize=1)
# 创建一个事件用于控制线程的停止
stop_event = threading.Event()
lock = threading.Lock()

detect_stop_event = threading.Event()
capture_stop_event = threading.Event()


def detection_thread():
    """持续获取当前检测到的物体信息的线程"""
    print("检测线程开始")
    processor = main()
    print("processor = main() over")

    while not detect_stop_event.is_set():
        results = processor.latest_results

        # 放入队列（如果已满，先清空）
        with lock:
            if not latest_results_queue.empty():
                latest_results_queue.get()
            latest_results_queue.put(copy.deepcopy(results))

        if results:  # 一旦有检测结果就结束检测
            print("检测到结果，准备退出检测线程")
            detect_stop_event.set()
            break

        time.sleep(1)  # 控制检测频率

    print("检测线程结束")


def capture_thread():
    """处理抓取任务的线程"""
    print("抓取线程开始")
    lebai_sdk.init()
    robot_ip = "10.20.17.1"  # 设定机器人IP地址，根据实际情况修改
    lebai = lebai_sdk.connect(robot_ip, False)  # 创建实例
    lebai.start_sys()  # 启动手臂
    tcp = {'x': 0, 'y': 0, 'z': 0.2, 'rz': 0, 'ry': 0, 'rx': 0}
    lebai.set_tcp(tcp)
    # while not stop_event.is_set():
    while not capture_stop_event.is_set():
        # print("capture_thread while")
        if not latest_results_queue.empty():
            print("capture->not latest_results_queue.empty()")
            lock.acquire()
            objects = latest_results_queue.get()
            lock.release()
            if objects:
                print("\n当前检测到的物体：")
                for obj in objects:
                    if obj['confidence'] < 0.88:
                        continue
                    print("obj:", obj)
                    print(f"- {obj['class_name']}: "
                          f"X={obj['world_pos'][0]:.1f}mm, "
                          f"Y={obj['world_pos'][1]:.1f}mm, "
                          f"旋转角度={obj['angle']:.1f}rad, "
                          f"置信度={obj['confidence']:.2f}")

                # 在这里添加抓取逻辑

                # lebai.stop_sys()  # 停止手臂
                # time.sleep(1)
                # lebai.start_sys()  # 启动手臂

                # 初始位置
                # pose1 = {'x': 0.336, 'y': 0.135, 'z': 0.308, 'rx': math.radians(175),
                #         'ry': math.radians(-16), 'rz': math.radians(-1)}
                # new_joint_pose = lebai.kinematics_inverse(pose1)
                # lebai.movej(new_joint_pose, a=1, v=1)
                # lebai.set_claw(50, 100)
                # lebai.wait_move()

                    # obj = get_current_objects()
                    # print("position_data:", obj)
                    object_position = 0
                    confidence = 0
                    # for data in obj:
                    #     if confidence < data['confidence']:
                    # confidence = obj['confidence']
                    object_position = obj['world_pos']
                    print('object_position:', object_position)
                    print('object_position[0]:', object_position[0])
                    print('object_position[1]:', object_position[1])
                    object_position[1] += 30
                    # 第一个目标点
                    pose = {'x': (object_position[0] + 120) / 1000, 'y': -(object_position[1]) / 1000, 'z': 0.3,
                            'rx': math.radians(180),
                            'ry': math.radians(0), 'rz': math.radians(0)}
                    print("pose:", pose)
                    new_joint_pose = lebai.kinematics_inverse(pose)
                    lebai.movej(new_joint_pose, a=1, v=1)
                    lebai.wait_move()

                    # 张开夹爪
                    lebai.set_claw(50, 100)
                    lebai.wait_move()

                    # 超目标前进
                    pose = {'x': (object_position[0] + 120) / 1000, 'y': -(object_position[1]) / 1000, 'z': 0.10,
                            'rx': math.radians(180),
                            'ry': math.radians(0), 'rz': math.radians(0)}
                    # print("pose:", pose)
                    lebai.movel(pose, a=1, v=0.5)
                    # new_joint_pose = lebai.kinematics_inverse(pose)
                    # lebai.movej(new_joint_pose, a=1, v=1)
                    lebai.wait_move()

                    # 夹取物体
                    lebai.set_claw(50, 0)
                    lebai.wait_move()

                    # 抬起
                    pose = {'x': (object_position[0] + 120) / 1000, 'y': -(object_position[1]) / 1000, 'z': 0.3,
                            'rx': math.radians(180),
                            'ry': math.radians(0), 'rz': math.radians(0)}
                    # print("pose:", pose)
                    new_joint_pose = lebai.kinematics_inverse(pose)
                    lebai.movej(new_joint_pose, a=1, v=1)
                    lebai.wait_move()
                    # start_capture(obj)

                    pose = {'x': 0.105, 'y': 0.492, 'z': 0.3,
                            'rx': math.radians(180),
                            'ry': math.radians(0), 'rz': math.radians(0)}
                    new_joint_pose = lebai.kinematics_inverse(pose)
                    lebai.movej(new_joint_pose, a=1, v=1)
                    lebai.wait_move()

                    pose = {'x': 0.105, 'y': 0.492, 'z': 0,
                            'rx': math.radians(180),
                            'ry': math.radians(0), 'rz': math.radians(0)}
                    new_joint_pose = lebai.kinematics_inverse(pose)
                    lebai.movej(new_joint_pose, a=1, v=1)
                    lebai.wait_move()

                    # 释放
                    lebai.set_claw(0, 100)
                    lebai.wait_move()

                    stop_event.set()

                capture_stop_event.set()
                break

        time.sleep(1)  # 控制处理频率


if __name__ == "__main__":
    try:
        while True:
            # 每次执行先清理 event 状态
            detect_stop_event.clear()
            capture_stop_event.clear()

            # 启动检测线程
            detect_thread = threading.Thread(target=detection_thread)
            detect_thread.start()
            detect_thread.join()  # 等检测线程结束

            # 启动抓取线程
            capture_thread_obj = threading.Thread(target=capture_thread)
            capture_thread_obj.start()
            capture_thread_obj.join()  # 等抓取线程结束

            print("\n--- 完成一轮检测与抓取 ---\n")
            time.sleep(20)
            print("\n--- 下一轮检测与抓取即将开始 ---\n")

    except KeyboardInterrupt:
        print("\n手动停止程序...")
        detect_stop_event.set()
        capture_stop_event.set()
        print("等待线程安全退出...")
        detect_thread.join()
        capture_thread_obj.join()
        print("程序已停止")