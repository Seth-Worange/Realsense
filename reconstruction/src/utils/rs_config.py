'''
Author: Orange
Date: 2026-02-27 11:10
LastEditors: Orange
LastEditTime: 2026-02-27 11:14
FilePath: rs_config.py
Description: 
    RealSense configuration parameters
'''


class RSConfig:
    def __init__(self):
        # Camera Resolution and FPS
        self.width = 640
        self.height = 480
        self.fps = 30
        
        # Point Cloud Processing Parameters
        self.clip_distance_m = 3.0          # Maximum depth distance (meters)
        self.voxel_size = 0.02              # Downsample voxel size
        self.human_depth_tolerance = 0.7    # Depth tolerance for human masking
        self.depth_scale_default = 0.001
        
        # MediaPipe Flags
        self.enable_face = True
        self.enable_hand = True 
        
        # Post-Processing Hardware Filters
        self.decimation_mag = 1             # 1 = No decimation
        self.spatial_mag = 2
        self.spatial_smooth_alpha = 0.5
        self.spatial_smooth_delta = 20
        self.spatial_holes_fill = 0
        self.temporal_smooth_alpha = 0.4
        self.temporal_smooth_delta = 20
