'''
Author: Orange
Date: 2026-02-27 16:01
LastEditors: Orange
LastEditTime: 2026-02-27 16:16
FilePath: seq_loader.py
Description: 
    Realsense Pointcloud Data Loader
'''

import os
import glob
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

class PointCloudSequencePlayer:
    def __init__(self, seq_dir, fps=10.0):
        """
        Point Cloud Sequence Player
        :param seq_dir: frame_XXXX.ply 
        :param fps: frame per second
        """
        self.seq_dir = seq_dir
        self.fps = fps
        self.dt = 1.0 / self.fps if self.fps > 0 else 0.1
        
        # 查找该目录下所有的 ply 文件并排序
        search_pattern = os.path.join(self.seq_dir, "*.ply")
        self.frame_paths = sorted(glob.glob(search_pattern))
        
        self.num_frames = len(self.frame_paths)
        if self.num_frames == 0:
            print(f"[警告] 未在 {self.seq_dir} 找到任何 .ply 文件！")
            
        self.current_idx = 0
        self.prev_pc_coords = None

    def get_frame_count(self):
        return self.num_frames

    def reset(self):
        self.current_idx = 0
        self.prev_pc_coords = None

    def load_frame(self, idx):
        """
        Load specific frame data
        :return: (pc_coords, pc_colors)
                 pc_coords: (N, 3) real 3D coordinates matrix [X, Y, Z]
                 pc_colors: (N, 3) normalized RGB [R, G, B]
        """
        if idx < 0 or idx >= self.num_frames:
            return None, None
            
        path = self.frame_paths[idx]
        pcd = o3d.io.read_point_cloud(path)
        
        if not pcd.has_points():
            return np.zeros((0,3)), np.zeros((0,3))
            
        pc_coords = np.asarray(pcd.points) # (N, 3)
        
        if pcd.has_colors():
            pc_colors = np.asarray(pcd.colors) # (N, 3)
        else:
            pc_colors = np.ones_like(pc_coords) * 0.5 # Default gray
            
        return pc_coords, pc_colors

    def next_frame_with_velocity(self):
        """
        逐帧读取，利用帧间光流法/质心差分法近似计算当前点云的速度 (velocity) 矩阵。
        
        :return: (pc_coords, velocities)
        """
        if self.current_idx >= self.num_frames:
            return None, None
            
        pc_coords, _ = self.load_frame(self.current_idx)
        
        if pc_coords is None or len(pc_coords) == 0:
            self.current_idx += 1
            return np.zeros((0,3)), np.zeros((0,3))
            
        N = pc_coords.shape[0]
        velocities = np.zeros((N, 3))
        
        if self.prev_pc_coords is not None and len(self.prev_pc_coords) > 0:
            # Use SciPy's cKDTree for vectorized batch query
            tree = cKDTree(self.prev_pc_coords)
            
            # find nearest neighbors
            distances, indices = tree.query(pc_coords, k=1)
            
            # Construct valid mask: cKDTree returns true distance, not squared distance
            valid_mask = distances < 0.5  
            
            # Matrix assignment of velocity
            velocities[valid_mask] = (pc_coords[valid_mask] - self.prev_pc_coords[indices[valid_mask]]) / self.dt

        # Update state machine
        self.prev_pc_coords = pc_coords
        self.current_idx += 1
        
        return pc_coords, velocities

    def get_all_as_radar_targets(self, frame_idx, constant_rcs=0.1):
        """
        转化为[pc, vel, rcs]
        """
        # 保存原始状态以便恢复
        saved_idx = self.current_idx
        saved_prev = self.prev_pc_coords
        
        if frame_idx == 0:
            # 如果是第 0 帧，它没有前一帧来计算速度。为了信息完整，我们借用第 1 帧算出的速度（假设初速度=第1帧速度）
            if self.num_frames > 1:
                self.current_idx = 0
                self.prev_pc_coords = self.load_frame(0)[0]
                _, next_vel = self.next_frame_with_velocity() # 这算的是 0->1 的速度
                
                # 重新读一回第 0 帧的空间数据
                pc, _ = self.load_frame(0)
                vel = next_vel[:len(pc)] if next_vel is not None else np.zeros_like(pc)
            else:
                pc, _ = self.load_frame(0)
                vel = np.zeros_like(pc)
        else:
            # 正常情况：计算 frame_idx-1 -> frame_idx 的速度
            self.current_idx = frame_idx                          # 直接指向目标帧
            self.prev_pc_coords = self.load_frame(frame_idx - 1)[0]  # 前一帧始终存在(frame_idx >= 1)
            pc, vel = self.next_frame_with_velocity()
        
        # 恢复状态机
        self.current_idx = saved_idx
        self.prev_pc_coords = saved_prev
        
        if pc is None:
            return None, None, None
            
        rcs = np.ones(pc.shape[0]) * constant_rcs
        return pc, vel, rcs

    def iter_all_as_radar_targets(self, constant_rcs=0.1):
        """
        生成器：逐帧遍历整个序列，yield (frame_idx, pc, vel, rcs)。
        
        利用内部状态机顺序读取，每帧自动通过 KDTree 最近邻匹配估算速度。
        比逐帧调用 get_all_as_radar_targets() 高效，因为避免了重复加载前一帧。
        """
        self.reset()
        
        for frame_idx in range(self.num_frames):
            pc, vel = self.next_frame_with_velocity()
            
            if pc is None or len(pc) == 0:
                continue
            
            rcs = np.ones(pc.shape[0]) * constant_rcs
            yield frame_idx, pc, vel, rcs
        
        # 遍历结束后重置状态机，避免污染后续调用
        self.reset()
