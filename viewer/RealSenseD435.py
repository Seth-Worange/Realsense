"""
File: RealsenseD435.py
Author: Orange
Date: 2026-1-8
Description:
    Capture depth frames from Intel RealSense D435, and visualize.

Usage:
    python Realsense.py

Dependencies:
    - pyrealsense2
    - numpy
    - open3d
    - cv2
    - time
    - threading

Notes:
    Tested on Python 3.12, RealSense SDK 2.56.
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d
from open3d.visualization import gui
from open3d.visualization import rendering
import threading
import time

class RealSenseApp:
    def __init__(self):
        # --- RealSense 硬件配置 ---
        self.w, self.h, self.fps = 640, 480, 30
        try:
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            self.config.enable_stream(rs.stream.depth, self.w, self.h, rs.format.z16, self.fps)
            self.config.enable_stream(rs.stream.color, self.w, self.h, rs.format.bgr8, self.fps)
            self.profile = self.pipeline.start(self.config)
            print("[System] RealSense Camera Started.")
        except Exception as e:
            print(f"[Error] Failed to start RealSense: {e}")
            return

        self.align = rs.align(rs.stream.color) 

        # 获取相机内参 (用于生成点云)
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        intr = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.pinhole_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy)

        self.is_running = True

        # --- 2. 界面参数设置 ---
        self.sidebar_width = 520 

        # --- 3. 初始化 Open3D GUI ---
        self.app = gui.Application.instance
        self.app.initialize()
        
        self.window = self.app.create_window("RealSense Dashboard", 1350, 860)
        w = self.window
        w.set_on_close(self._on_close)
        w.set_on_layout(self._on_layout) # 绑定手动布局

        em = w.theme.font_size

        # --- 创建组件 ---
        
        # A. 左侧面板 (垂直容器)
        self.side_panel = gui.Vert(0.5 * em, gui.Margins(0.5 * em, 0.5 * em, 0.5 * em, 0.5 * em))
        
        # RGB 预览窗
        self.rgb_widget = gui.ImageWidget()
        self.side_panel.add_child(gui.Label("RGB Feed"))
        self.side_panel.add_child(self.rgb_widget)
        
        # Depth 预览窗
        self.depth_widget = gui.ImageWidget()
        self.side_panel.add_child(gui.Label("Depth Heatmap"))
        self.side_panel.add_child(self.depth_widget)
        
        # B. 右侧 3D 场景
        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(w.renderer)
        self.scene_widget.scene.set_background([0.2, 0.2, 0.2, 1.0]) 
        self.scene_widget.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)

        # 直接添加到 Window 
        w.add_child(self.side_panel)
        w.add_child(self.scene_widget)

        # 启动数据线程
        threading.Thread(target=self.update_thread).start()

    # --- 布局回调：强制分配空间 ---
    def _on_layout(self, layout_context):
        r = self.window.content_rect
        
        # 1. 设置侧边栏 (固定宽度)
        self.side_panel.frame = gui.Rect(r.x, r.y, self.sidebar_width, r.height)
        
        # 2. 设置 3D 视图 (占据剩余所有空间)
        scene_x = r.x + self.sidebar_width
        scene_w = r.width - self.sidebar_width
        self.scene_widget.frame = gui.Rect(scene_x, r.y, scene_w, r.height)

    def _on_close(self):
        self.is_running = False
        return True 

    def update_thread(self):
        first_frame = True
        
        while self.is_running:
            try:
                # 1. 获取并对齐帧
                frames = self.pipeline.wait_for_frames(timeout_ms=1000)
                aligned_frames = self.align.process(frames)
                
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                
                if not depth_frame or not color_frame:
                    continue

                # 2. 原始数据处理
                img_color = np.asanyarray(color_frame.get_data())
                img_depth = np.asanyarray(depth_frame.get_data())
                
                # 3. 准备数据：高清原图 (用于生成点云)
                img_color_rgb_full = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
                img_color_rgb_full = np.ascontiguousarray(img_color_rgb_full) 
                
                # 4. 准备数据：UI 缩略图 (用于侧边栏显示)
                preview_w = self.sidebar_width - 30
                # 保持 4:3 的纵横比
                preview_h = int(preview_w * 0.75) 
                
                # 缩放 RGB
                img_color_thumb = cv2.resize(img_color, (preview_w, preview_h))
                img_color_thumb = cv2.cvtColor(img_color_thumb, cv2.COLOR_BGR2RGB)
                img_color_thumb = np.ascontiguousarray(img_color_thumb)
                
                # 缩放 Depth (先伪彩色再缩放)
                img_depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(img_depth, alpha=0.03), cv2.COLORMAP_JET)
                img_depth_thumb = cv2.resize(img_depth_colormap, (preview_w, preview_h))
                img_depth_thumb = cv2.cvtColor(img_depth_thumb, cv2.COLOR_BGR2RGB)
                img_depth_thumb = np.ascontiguousarray(img_depth_thumb)

                # 5. 创建 Open3D Image 对象
                o3d_color_full = o3d.geometry.Image(img_color_rgb_full)
                o3d_depth_full = o3d.geometry.Image(img_depth) 

                o3d_color_thumb = o3d.geometry.Image(img_color_thumb)
                o3d_depth_thumb = o3d.geometry.Image(img_depth_thumb)

                # 6. 生成点云 (始终使用高清图，保证精度)
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    o3d_color_full, o3d_depth_full, 
                    depth_scale=1.0/self.depth_scale, 
                    depth_trunc=6.0, 
                    convert_rgb_to_intensity=False
                )
                
                new_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, self.pinhole_intrinsics)

                # 7. 刷新 UI
                def update_ui():
                    if not self.is_running: return
                    
                    # 更新侧边栏预览图
                    self.rgb_widget.update_image(o3d_color_thumb)
                    self.depth_widget.update_image(o3d_depth_thumb)
                    
                    # 更新 3D 点云
                    if self.scene_widget.scene.has_geometry("pcd"):
                        self.scene_widget.scene.remove_geometry("pcd")
                    
                    mat = rendering.MaterialRecord()
                    mat.shader = "defaultUnlit"
                    mat.point_size = 3.5 
                    self.scene_widget.scene.add_geometry("pcd", new_pcd, mat)
                    
                    nonlocal first_frame
                    if first_frame:
                        bounds = self.scene_widget.scene.bounding_box
                        self.scene_widget.setup_camera(60, bounds, bounds.get_center())
                        
                        # 调整初始视角：稍微拉近，正对前方
                        center = np.array([0, 0, 1.0])
                        eye = np.array([0, 0, -0.1])   
                        up = np.array([0, -1, 0])      
                        self.scene_widget.look_at(center, eye, up)
                        first_frame = False

                self.app.post_to_main_thread(self.window, update_ui)
                
            except Exception as e:
                print(f"Loop Error: {e}")
                time.sleep(0.1)

    def run(self):
        self.app.run()
        self.is_running = False
        self.pipeline.stop()
        print("[System] Application Closed.")

if __name__ == "__main__":
    app = RealSenseApp()
    app.run()