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

class PointCloudSequencePlayer:
    def __init__(self, seq_dir, fps=10.0):
        """
        点云序列加载器，专门用于读取一段采集好的连续单帧 PointCloud 并计算物理信息
        :param seq_dir: 保存有 frame_XXXX.ply 的文件夹绝对或相对路径
        :param fps: 采集该序列时的实际帧率，用于推算 d_time
        """
        self.seq_dir = seq_dir
        self.fps = fps
        self.dt = 1.0 / self.fps if self.fps > 0 else 0.1
        
        # 查找该目录下所有的 ply 文件并按数字序号排序
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
        读取特定帧的纯净数据
        :return: (pc_coords, pc_colors)
                 pc_coords: (N, 3) 真实三维坐标距阵 [X, Y, Z]
                 pc_colors: (N, 3) 对应的归一化 RGB [R, G, B]
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
        [雷达仿真核心组件]
        逐帧向后读取，并利用帧间光流法/质心差分法近似计算当前点云的速度 (velocity) 矩阵。
        
        :return: (pc_coords, velocities, pc_colors)
                 velocities 即为雷达探测到的每个反射点的 [vx, vy, vz]
                 如果是最后一段或空序列返回 None
        """
        if self.current_idx >= self.num_frames:
            return None, None, None
            
        pc_coords, pc_colors = self.load_frame(self.current_idx)
        
        if pc_coords is None or len(pc_coords) == 0:
            self.current_idx += 1
            return np.zeros((0,3)), np.zeros((0,3)), np.zeros((0,3))
            
        N = pc_coords.shape[0]
        velocities = np.zeros((N, 3))
        
        if self.prev_pc_coords is not None and len(self.prev_pc_coords) > 0:
            # 真实情况下，由于点云 N 的数量每一帧是不定长的，且对应顺序随机，
            # 直接做减法 pc_coords - prev_pc 最多引起维数灾难。
            # 因此这里用一种巧妙的“最近邻插值假定 (Nearest Neighbor Assumption)”来估算：
            # 找到当前帧中每个点在上一帧中最靠近的那个“躯干部位点”，认为它是从那里移动过来的。
            
            # 使用 Open3D 的 KDTree 进行极速的欧氏距离查找
            pcd_prev = o3d.geometry.PointCloud()
            pcd_prev.points = o3d.utility.Vector3dVector(self.prev_pc_coords)
            kdtree = o3d.geometry.KDTreeFlann(pcd_prev)
            
            for i in range(N):
                query_point = pc_coords[i]
                # k=1 表示找最近的1个邻居点
                [_, idx, dist] = kdtree.search_knn_vector_3d(query_point, 1)
                
                # 若最近点距离差异过大 ( > 0.3 米 ) 认为是噪声或新出现的肢体，速度赋 0
                if dist[0] < 0.3**2: # dist 返回的是距离的平方
                    prev_point = self.prev_pc_coords[idx[0]]
                    velocities[i] = (query_point - prev_point) / self.dt

        # 更新状态机
        self.prev_pc_coords = pc_coords
        self.current_idx += 1
        
        return pc_coords, velocities, pc_colors

    def get_all_as_radar_targets(self, frame_idx, constant_rcs=0.1):
        """
        快速将指定的帧转化为雷达仿真需要的数据结构 [pc, vel, rcs]
        便于在 radar_generate.py 中直接替换掉虚拟随机生成假人。
        """
        # 保存原始状态以便恢复
        saved_idx = self.current_idx
        saved_prev = self.prev_pc_coords
        
        # 强制将状态机挪到它的前一帧计算
        self.current_idx = max(0, frame_idx - 1)
        self.prev_pc_coords = self.load_frame(self.current_idx)[0] if self.current_idx > 0 else None
        
        # 向前步进 1 步并计算 velocity
        pc, vel, _ = self.next_frame_with_velocity()
        
        # 恢复状态机
        self.current_idx = saved_idx
        self.prev_pc_coords = saved_prev
        
        if pc is None:
            return None, None, None
            
        rcs = np.ones(pc.shape[0]) * constant_rcs
        return pc, vel, rcs
