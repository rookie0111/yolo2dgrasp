import numpy as np
import open3d as o3d
import pyrealsense2 as rs
import cv2

# 获取深度图和彩色图像（使用上面的代码）
# depth_image和color_image是numpy数组

# 获取相机内参
depth_scale = 0.001  # 深度值的转换因子，具体根据相机配置
intrinsics = rs.intrinsics()
intrinsics.width = 640
intrinsics.height = 480
intrinsics.ppx = 319.58245849609375
intrinsics.ppy = 238.5020751953125
intrinsics.fx = 388.5013122558594  # 示例值
intrinsics.fy = 388.5013122558594  # 示例值
intrinsics.model = rs.distortion.none
intrinsics.coeffs = [0, 0, 0, 0, 0]


# 将深度值转换为点云
def create_point_cloud(depth_image, color_image):
    points = []
    colors = []

    for v in range(depth_image.shape[0]):
        for u in range(depth_image.shape[1]):
            z = depth_image[v, u] * depth_scale  # 将深度值换算成实际的距离
            if z > 0:  # 仅处理有效的点
                x = (u - intrinsics.ppx) * z / intrinsics.fx
                y = (v - intrinsics.ppy) * z / intrinsics.fy
                points.append([x, y, z])
                colors.append(color_image[v, u] / 255.0)  # 归一化颜色值

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(np.array(points))
    point_cloud.colors = o3d.utility.Vector3dVector(np.array(colors))

    return point_cloud


# 创建点云并可视化
color_image_path = r"D:\project\ultralytics-main\ultralytics\outputcolor\color_shot_2.png"
depth_image_path = r"D:\project\ultralytics-main\ultralytics\outputdepth\depth_frame_2.png"

color_image = cv2.imread(color_image_path, cv2.IMREAD_COLOR)
depth_image = cv2.imread(depth_image_path, cv2.IMREAD_GRAYSCALE)
pcd1 = create_point_cloud(depth_image, color_image)
pcd2 = o3d.io.read_point_cloud(r"D:\BaiduNetdiskDownload\bunny.pcd")
o3d.visualization.draw_geometries([pcd1])
# # 法线估计
# radius = 0.01  # 搜索半径
# max_nn = 30  # 邻域内用于估算法线的最大点数
# pcd1.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius, max_nn))  # 执行法线估计
#
# # 可视化
# o3d.visualization.draw_geometries([pcd1],
#                                   window_name="可视化参数设置",
#                                   width=600,
#                                   height=450,
#                                   left=30,
#                                   top=30,
#                                   point_show_normal=True)

# ------------------------- Alpha shapes -----------------------
alpha = 0.3
print(f"alpha={alpha:.3f}")
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd1, alpha)
mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
