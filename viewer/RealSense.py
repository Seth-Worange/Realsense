"""
File: Realsense.py
Author: Pai
Date: 2026-1-8
Description:
    Capture depth frames from Intel RealSense D430, and visualize.

Usage:
    python Realsense.py

Dependencies:
    - pyrealsense2
    - numpy
    - opencv-python
    - os
    - time
    - open3d
    - datatime
    - matplotlib
    - cv2

Notes:
    Tested on Python 3.12, RealSense SDK 2.56.
"""
import os
import time
from datetime import datetime
import open3d as o3d
import matplotlib.pyplot as plt
import pyrealsense2 as rs
import numpy as np
import cv2


# config for chosen
# 1280x720 @6
# 848x480 @10/8/6
# 640x480 @30/15/6
# 640x360 @30
# 480x270 @60/30/15/6
# 256x144 @90


def start_realsense(h=720, w=1280, fps=6):
    """
    Start and Config the RealSense Sensor.

    Parameters
    ----------
    h : int
        Height of the depth image
    w : int
        Width of the depth image
    fps : int
        Frequency of the depth image
    Returns
    -------
    pipeline : pyrealsense2.pipeline
        Active RealSense pipeline object that manages streaming.
    """
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
    active = pipeline.start(config)
    return pipeline


def save_realsense(pipeline, save_path, save_event):
    """
    Save the RealSense Sensor Depth Image Data.

    Parameters
    ----------
    pipeline : pyrealsense2.pipeline
        Active RealSense pipeline object that manages streaming.
    save_path : str
        Depth Image Data saved in.
    Returns
    -------
    None
    """
    os.makedirs(save_path, exist_ok=True)
    depth_sensor = pipeline.get_active_profile().get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        if depth_frame:
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_meters = depth_image * depth_scale
            if save_path:
                if save_event.is_set():
                    formatted_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
                    np.save(save_path + '/' + formatted_time + '.npy', depth_meters)
        time.sleep(0.01)


def finish_realsense(pipeline):
    """
    Finish the RealSense Sensor.

    Parameters
    ----------
    pipeline : pyrealsense2.pipeline
        Active RealSense pipeline object that manages streaming.
    Returns
    -------
    None
    """
    pipeline.stop()


def depth_image_convert_pointclouds_single_frame(path, vis_flag=True):
    """
    Convert the RealSense Depth Image Data to 3D PointCloud and Visualize by Single Frame.

    Parameters
    ----------
    path : str
        path to specific npy file
    Returns
    -------
    xyz : list [num, 3]
    """
    # --- 相机内参 (从 RealSense intrinsics 获取)
    fx = 645.8804931640625
    fy = 645.8804931640625
    ppx = 639.7295532226562
    ppy = 349.48358154296875

    # --- 读取深度图
    depth = np.load(path)  # H×W, 单位：米
    H, W = depth.shape

    # --- 构建像素坐标网格
    u, v = np.meshgrid(np.arange(W), np.arange(H))  # u=X方向, v=Y方向

    # --- 转换为三维坐标
    Z = depth
    X = (u - ppx) * Z / fx
    Y = (v - ppy) * Z / fy

    # --- 拼成 N×3 点云矩阵
    xyz = np.dstack((X, Y, Z)).reshape(-1, 3)

    # --- 可选：去掉无效点（深度=0）
    xyz = xyz[xyz[:, 2] > 0]
    # xyz = xyz[xyz[:, 2] < 2]
    if vis_flag:
        voxel_size = 0.025
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
        pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
        pts = np.asarray(pcd_down.points)

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=pts[:, 2], cmap='turbo', s=2)
        ax.scatter([0], [0], [0], c='r', s=10)
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")
        ax.set_title(f"Voxel Downsampled ({voxel_size:.3f} m)")
        plt.show()
    return xyz


def depth_image_convert_pointclouds_multi_frame(path):
    """
    Convert the RealSense Depth Image Data to 3D PointCloud and Visualize by Multi Frame.

    Parameters
    ----------
    path : str
        path to specific npy file
    Returns
    -------
    None
    """
    # --- 相机内参 (从 RealSense intrinsics 获取)
    fx = 645.8804931640625
    fy = 645.8804931640625
    ppx = 639.7295532226562
    ppy = 349.48358154296875

    xyz_past = None
    plt.ion()
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    # --- 读取深度图
    files = os.listdir(path)
    for idx, f in enumerate(files):
        depth = np.load(path + '/' + f)  # H×W, 单位：米
        H, W = depth.shape

        # --- 构建像素坐标网格
        u, v = np.meshgrid(np.arange(W), np.arange(H))  # u=X方向, v=Y方向

        # --- 转换为三维坐标
        Z = depth
        X = (u - ppx) * Z / fx
        Y = (v - ppy) * Z / fy

        # --- 拼成 N×3 点云矩阵
        xyz = np.dstack((X, Y, Z)).reshape(-1, 3)
        # xyz = (xyz - xyz_past) if xyz_past is not None else xyz
        # xyz_past = xyz
        # --- 去掉无效点（深度=0）
        xyz = xyz[xyz[:, 2] > 0]
        xyz = xyz[xyz[:, 2] < 5]
        voxel_size = 0.025
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
        pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
        pts = np.asarray(pcd_down.points)

        plt.cla()
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=pts[:, 2], cmap='turbo', s=2)
        ax.scatter([0], [0], [0], c='r', s=10)
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")
        ax.set_title("Frame: {}/{}".format(idx, len(files)))
        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        ax.set_zlim([0, 5])
        ax.view_init(elev=-90, azim=-90, roll=0)
        plt.pause(0.1)
    plt.ioff()
    plt.show()
    return


def realtime_dual_display(h=480, w=640, fps=30):
    # 1. 初始化
    pipeline = rs.pipeline()
    config = rs.config()
    
    # D435 开启深度流
    config.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
    # --- 新增：D435 开启彩色流 ---
    config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
    
    profile = pipeline.start(config)

    # --- 新增：创建对齐对象 (将深度帧对齐到颜色帧) ---
    align_to = rs.stream.color
    align = rs.align(align_to)

    # 获取相机内参 (用于点云转换)
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    pinhole_intrinsics = o3d.camera.PinholeCameraIntrinsic(
        intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='D435 3D Color Point Cloud', width=800, height=600)
    pcd = o3d.geometry.PointCloud()
    first_frame = True

    try:
        while True:
            frames = pipeline.wait_for_frames()
            
            # --- 新增：执行对齐 ---
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                continue

            # 处理 2D 显示
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            # 2D 窗口显示彩色图
            cv2.imshow('D435 RGB Monitor', color_image)

            # 处理 3D 点云（带颜色）
            o3d_depth = o3d.geometry.Image(depth_image)
            o3d_color = o3d.geometry.Image(color_image)
            
            # --- 修改：从 RGBD 图像创建彩色点云 ---
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d_color, o3d_depth, depth_scale=1000.0, depth_trunc=3.0, convert_rgb_to_intensity=False)
            
            new_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole_intrinsics)

            pcd.points = new_pcd.points
            pcd.colors = new_pcd.colors 

            if first_frame:
                vis.add_geometry(pcd)
                first_frame = False
            else:
                vis.update_geometry(pcd)

            vis.poll_events()
            vis.update_renderer()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        vis.destroy_window()


if __name__ == '__main__':
    # 推荐 D430 在 640x480 下运行以保持 3D 渲染的流畅度
    realtime_dual_display(w=640, h=480, fps=30)

# if __name__ == '__main__':
#     RGBD_path = r'G:\研二\面向医疗康复的多传感器融合精确人体三维重构技术研究\Data Collection System/RGBD'
#     os.makedirs(RGBD_path, exist_ok=True)
#     # pipeline = start_realsense()
#     # save_realsense(pipeline, save_path=RGBD_path)
#     # depth_image_convert_pointclouds_single_frame(
#     #     r'G:\研二\面向医疗康复的多传感器融合精确人体三维重构技术研究\Data Collection System\RGBD\1\2025_10_20_21_37_47_377628.npy')
#     depth_image_convert_pointclouds_multi_frame(
#         r'G:\研二\面向医疗康复的多传感器融合精确人体三维重构技术研究\Data Collection System\RGBD\1')
