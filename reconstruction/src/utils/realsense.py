'''
Author: Orange
Date: 2026-02-27 10:42
LastEditors: Orange
LastEditTime: 2026-02-27 16:03
FilePath: realsense.py
Description: 
    RealSense Processor: Capture frames and generate point clouds
    MediaPipe version: 0.10.31
'''


import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import open3d as o3d
from enum import Enum

class PCDMode(Enum):
    FULL = 0       
    HUMAN_ONLY = 1  
    BACKGROUND = 2  

from .rs_config import RSConfig

# =========================================
# RealSenseProcessor
# Func： Capture frames and generate point clouds
# =========================================
class RealSenseProcessor:
    def __init__(self, rs_config=None):
        if rs_config is None:
            rs_config = RSConfig()
            
        self.width = rs_config.width
        self.height = rs_config.height
        self.fps = rs_config.fps
        
        # RealSense Pipeline Setup
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        
        self.align = rs.align(rs.stream.color)
        self.profile = None
        self.started = False
        
        self.pinhole_intrinsics = None
        self.depth_scale = rs_config.depth_scale_default 
        self.clip_distance_m = rs_config.clip_distance_m 
        self.voxel_size = rs_config.voxel_size
        self.human_depth_tolerance = rs_config.human_depth_tolerance 
        self.enable_face = rs_config.enable_face
        self.enable_hand = rs_config.enable_hand 

        # Hardware Filters
        self.decimation = rs.decimation_filter()
        self.decimation.set_option(rs.option.filter_magnitude, rs_config.decimation_mag) 
        
        self.spatial = rs.spatial_filter()
        self.spatial.set_option(rs.option.filter_magnitude, rs_config.spatial_mag)
        self.spatial.set_option(rs.option.filter_smooth_alpha, rs_config.spatial_smooth_alpha)
        self.spatial.set_option(rs.option.filter_smooth_delta, rs_config.spatial_smooth_delta)
        self.spatial.set_option(rs.option.holes_fill, rs_config.spatial_holes_fill)
        
        self.temporal = rs.temporal_filter()
        self.temporal.set_option(rs.option.filter_smooth_alpha, rs_config.temporal_smooth_alpha)
        self.temporal.set_option(rs.option.filter_smooth_delta, rs_config.temporal_smooth_delta)
        
        self.hole_filling = rs.hole_filling_filter()

        # Initialize MediaPipe
        self.mp_pose = None
        self._init_mediapipe()

    # Initialize MediaPipe
    def _init_mediapipe(self):
        base_path = "C:\\OrangeFiles\\code\\mediapipe\\"
        pose_path = base_path + "pose_landmarker_heavy.task"
        hand_path = base_path + "hand_landmarker.task"
        face_path = base_path + "face_landmarker.task"
        
        try:
            # 1. Pose
            base_options = python.BaseOptions(model_asset_path=pose_path)
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                output_segmentation_masks=True
            )
            self.mp_pose = vision.PoseLandmarker.create_from_options(options)
            
            # 2. Hand
            base_options_hand = python.BaseOptions(model_asset_path=hand_path)
            options_hand = vision.HandLandmarkerOptions(
                base_options=base_options_hand,
                running_mode=vision.RunningMode.IMAGE,
                num_hands=2
            )
            self.mp_hand = vision.HandLandmarker.create_from_options(options_hand)

            # 3. Face
            base_options_face = python.BaseOptions(model_asset_path=face_path)
            options_face = vision.FaceLandmarkerOptions(
                base_options=base_options_face,
                running_mode=vision.RunningMode.IMAGE,
                num_faces=1
            )
            self.mp_face = vision.FaceLandmarker.create_from_options(options_face)
            
            print("[System] MediaPipe Pose/Hand/Face Landmarkers Loaded.")
            
        except Exception as e:
            print(f"[Error] Failed to load MediaPipe Models: {e}")
            self.mp_pose = None
            self.mp_hand = None
            self.mp_face = None

    def start(self):
        try:
            self.profile = self.pipeline.start(self.config)
            depth_sensor = self.profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()
            
            intr = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            self.pinhole_intrinsics = o3d.camera.PinholeCameraIntrinsic(
                intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy)
            
            self.started = True
            print("[System] RealSense Processor Started.")
        except Exception as e:
            print(f"[Error] Pipeline start failed: {e}")
            self.started = False

    def stop(self):
        if self.started:
            self.pipeline.stop()
            self.started = False

    # 2D pixel -> 3D coordinate
    def _deproject_pixel(self, x, y, depth_frame, intr):
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return None
            
        # 1. Try Center Pixel
        dist = depth_frame.get_distance(x, y)
        
        # 2. Robust Search: If invalid or noisy, search neighborhood
        # Fingertips often have noisy depth edges.
        if dist <= 0 or dist > 3.0:
            valid_depths = []
            radius = 2 # 5x5 window
            for dy in range(-radius, radius+1):
                for dx in range(-radius, radius+1):
                    if dx == 0 and dy == 0: continue
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.width and 0 <= ny < self.height:
                        d = depth_frame.get_distance(nx, ny)
                        if 0 < d <= 3.0:
                            valid_depths.append(d)
            
            if valid_depths:
                # Use median to avoid outliers
                dist = np.median(valid_depths)
            else:
                return None

        return rs.rs2_deproject_pixel_to_point(intr, [x, y], dist)

    # Execute inference, return: (segmentation mask, skeleton_data dict, keypoints_info list)
    def _analyze_pose_and_keypoints(self, color_img, depth_frame):
        default_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        
        # Structure for UI visualization
        skeleton_data = {
            'pose': [],
            'hands': [], # List of list of points
            'face': []   # List of points
        }
        
        # Structure for SMPL fitting: [idx, x3d, y3d, z3d, x2d, y2d, visibility]
        keypoints_info = [] 

        if not self.mp_pose:
            return default_mask, skeleton_data, keypoints_info

        # 1. Image preprocessing
        img_rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        
        intr = depth_frame.profile.as_video_stream_profile().get_intrinsics()

        # --- A. Pose Detection ---
        pose_result = self.mp_pose.detect(mp_image)
        segmentation_mask = default_mask
        
        if pose_result.segmentation_masks and len(pose_result.segmentation_masks) > 0:
            seg_mask_float = pose_result.segmentation_masks[0].numpy_view()
            segmentation_mask = (seg_mask_float > 0.5).astype(np.uint8)

        if pose_result.pose_landmarks and len(pose_result.pose_landmarks) > 0:
            landmarks = pose_result.pose_landmarks[0]
            # Convert for UI
            curr_pose_3d = []
            for idx, lm in enumerate(landmarks):
                px, py = int(lm.x * self.width), int(lm.y * self.height)
                p3d = self._deproject_pixel(px, py, depth_frame, intr)
                if p3d:
                    curr_pose_3d.append(p3d)
                    # Add to SMPL list with 2D info and visibility
                    # lm.x, lm.y are normalized [0,1]
                    vis = lm.visibility if hasattr(lm, 'visibility') else 1.0
                    keypoints_info.append([idx, p3d[0], p3d[1], p3d[2], lm.x, lm.y, vis])
                else:
                    curr_pose_3d.append(None) # Keep index structure for drawing lines
            skeleton_data['pose'] = curr_pose_3d

        # --- B. Hand Detection (Adaptive) ---
        # Strategy: Only run heavy Hand Landmarker if hand is close enough (large pixel size)
        # We use Pose landmarks as a proxy to estimate hand size.
        if self.enable_hand and self.mp_hand and pose_result.pose_landmarks:
            landmarks = pose_result.pose_landmarks[0]
            
            # Check Left Hand Size (Wrist 15 -> Index 19)
            left_hand_size = 0
            if 15 < len(landmarks) and 19 < len(landmarks):
                dx = (landmarks[15].x - landmarks[19].x) * self.width
                dy = (landmarks[15].y - landmarks[19].y) * self.height
                left_hand_size = (dx**2 + dy**2)**0.5
            
            # Check Right Hand Size (Wrist 16 -> Index 20)
            right_hand_size = 0
            if 16 < len(landmarks) and 20 < len(landmarks):
                dx = (landmarks[16].x - landmarks[20].x) * self.width
                dy = (landmarks[16].y - landmarks[20].y) * self.height
                right_hand_size = (dx**2 + dy**2)**0.5
            
            # Threshold: e.g. 40 pixels
            if left_hand_size > 40 or right_hand_size > 40:
                hand_result = self.mp_hand.detect(mp_image)
                for hand_lms in hand_result.hand_landmarks:
                    curr_hand_3d = []
                    for lm in hand_lms:
                        px, py = int(lm.x * self.width), int(lm.y * self.height)
                        p3d = self._deproject_pixel(px, py, depth_frame, intr)
                        curr_hand_3d.append(p3d if p3d else None)
                    skeleton_data['hands'].append(curr_hand_3d)

        # --- C. Face Detection ---
        if self.enable_face and self.mp_face:
            face_result = self.mp_face.detect(mp_image)
            if face_result.face_landmarks and len(face_result.face_landmarks) > 0:
                face_lms = face_result.face_landmarks[0]
                # Keep all points
                curr_face_3d = []
                for lm in face_lms:
                    px, py = int(lm.x * self.width), int(lm.y * self.height)
                    p3d = self._deproject_pixel(px, py, depth_frame, intr)
                    curr_face_3d.append(p3d if p3d else None)
                skeleton_data['face'] = curr_face_3d
        
        # Fallback: Use Pose landmarks (0-10) if high-res face not detected or disabled
        if not skeleton_data['face'] and len(skeleton_data['pose']) >= 11:
            skeleton_data['face'] = skeleton_data['pose'][0:11]
                            
        return segmentation_mask, skeleton_data, keypoints_info

    # Generate point cloud
    def _generate_point_cloud(self, color_bgr, depth_img):
        color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        o3d_color = o3d.geometry.Image(color_rgb)
        o3d_depth = o3d.geometry.Image(depth_img)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_color, o3d_depth, depth_scale=1.0/self.depth_scale,
            depth_trunc=6.0, convert_rgb_to_intensity=False)     
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, self.pinhole_intrinsics)
        if self.voxel_size > 0.001: 
            pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
        return pcd

    # Process hybrid mask, now supports external computed_mask
    def _process_hybrid_mask(self, depth_img, mode: PCDMode, computed_mask=None):
        if mode == PCDMode.FULL:
            return depth_img

        semantic_mask = computed_mask if computed_mask is not None else np.zeros_like(depth_img, dtype=np.uint8)

        # Distance clipping logic
        max_dist_units = self.clip_distance_m / self.depth_scale
        depth_mask_global = (depth_img > 0) & (depth_img < max_dist_units)

        final_mask = None
        if mode == PCDMode.HUMAN_ONLY:
            human_mask = cv2.bitwise_and(semantic_mask, depth_mask_global.astype(np.uint8))
            
            # Adaptive depth filtering (noise reduction)
            human_pixels = depth_img[human_mask == 1]
            if len(human_pixels) > 0:
                median_depth = np.median(human_pixels)
                tolerance = self.human_depth_tolerance / self.depth_scale
                adaptive_mask = (depth_img < (median_depth + tolerance)).astype(np.uint8)
                final_mask = cv2.bitwise_and(human_mask, adaptive_mask)
            else:
                final_mask = human_mask

        elif mode == PCDMode.BACKGROUND:
            bg_mask = 1 - semantic_mask
            final_mask = cv2.bitwise_and(bg_mask, depth_mask_global.astype(np.uint8))
        else:
            final_mask = depth_mask_global.astype(np.uint8)

        output_depth = depth_img.copy()
        output_depth[final_mask == 0] = 0
        return output_depth

    # Main entry point
    def get_data_bundle(self, mode: PCDMode = PCDMode.HUMAN_ONLY):
        if not self.started: return None
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            aligned_frames = self.align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame: return None
            
            # Apply Filters
            filtered = self.spatial.process(depth_frame)
            filtered = self.temporal.process(filtered)
            filtered = self.hole_filling.process(filtered)
            
            # Cast back to depth_frame to access .get_distance()
            depth_frame = filtered.as_depth_frame()
            if not depth_frame: return None
            
            img_color = np.asanyarray(color_frame.get_data())
            img_depth = np.asanyarray(depth_frame.get_data())
            
            # 1. Call new function to get Mask and keypoints
            semantic_mask, skeleton_data, keypoints_info = self._analyze_pose_and_keypoints(img_color, depth_frame)
            
            # 2. Process depth image (pass the Mask just obtained)
            masked_depth = self._process_hybrid_mask(img_depth, mode, computed_mask=semantic_mask)
            
            # 3. Generate point cloud
            pcd = self._generate_point_cloud(img_color, masked_depth)
            
            return {
                'color': img_color,
                'depth': img_depth,      
                'masked_depth': masked_depth, 
                'pcd': pcd,
                'mode': mode,
                'skeleton_data': skeleton_data,
                'keypoints_info': keypoints_info 
            }
        except Exception as e:
            print(f"[Error] Process failed: {e}")
            return None


