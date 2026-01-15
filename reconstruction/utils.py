import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import open3d as o3d
from enum import Enum
import torch
import smplx


class PCDMode(Enum):
    FULL = 0       
    HUMAN_ONLY = 1  
    BACKGROUND = 2  

# =========================================
# RealSenseProcessor
# Func： Capture frames and generate point clouds
# =========================================
class RealSenseProcessor:
    def __init__(self, width=640, height=480, fps=30):
        """
        Initialize pipeline, parameters, and mediapipe model.
        """
        self.width = width
        self.height = height
        self.fps = fps
        
        # Initialize RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        
        self.align = rs.align(rs.stream.color)
        self.profile = None
        self.started = False
        
        # Camera intrinsics and params
        self.pinhole_intrinsics = None
        self.depth_scale = 0.001 
        self.clip_distance_m = 3.0 # Global max distance

        # down sample factor for point cloud
        self.voxel_size = 0.01
        
        # Adaptive filter tolerance (meters)
        self.human_depth_tolerance = 0.7 

        # =========================================
        # Initialize MediaPipe Image Segmenter
        # =========================================
        model_path = "C:\\OrangeFiles\\code\\mediapipe\\selfie_segmenter.tflite"
        try:
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.ImageSegmenterOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                output_category_mask=True
            )
            self.mp_seg = vision.ImageSegmenter.create_from_options(options)
            print("[System] MediaPipe Image Segmenter (0.10.31) Loaded.")
        except Exception as e:
            print(f"[Error] Failed to load MediaPipe model: {e}")
            self.mp_seg = None

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
            print(f"[Error] Failed to start pipeline: {e}")
            self.started = False

    def _generate_point_cloud(self, color_bgr, depth_img):
        color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        
        o3d_color = o3d.geometry.Image(color_rgb)
        o3d_depth = o3d.geometry.Image(depth_img)
        
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_color, o3d_depth,
            depth_scale=1.0/self.depth_scale,
            depth_trunc=6.0, 
            convert_rgb_to_intensity=False
        )     
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, self.pinhole_intrinsics)

        if self.voxel_size > 0.001: 
            pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
            
        return pcd

    def _process_hybrid_mask(self, color_img, depth_img, mode: PCDMode):
        if mode == PCDMode.FULL:
            return depth_img

        if self.mp_seg is None:
            return depth_img

        img_rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result = self.mp_seg.segment(mp_image)

        category_mask = result.category_mask.numpy_view()
        semantic_mask = (category_mask <= 0.5).astype(np.uint8)

        max_dist_units = self.clip_distance_m / self.depth_scale
        depth_mask_global = (depth_img > 0) & (depth_img < max_dist_units)
        depth_mask_global = depth_mask_global.astype(np.uint8)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        semantic_mask = cv2.erode(semantic_mask, kernel)

        final_mask = None

        if mode == PCDMode.HUMAN_ONLY:
            human_mask = cv2.bitwise_and(semantic_mask, depth_mask_global)
            human_pixels = depth_img[human_mask == 1]
            
            if len(human_pixels) > 0:
                median_depth = np.median(human_pixels)
                tolerance_units = self.human_depth_tolerance / self.depth_scale
                adaptive_cutoff = median_depth + tolerance_units
                adaptive_mask = (depth_img < adaptive_cutoff).astype(np.uint8)
                final_mask = cv2.bitwise_and(human_mask, adaptive_mask)
            else:
                final_mask = human_mask

        elif mode == PCDMode.BACKGROUND:
            final_mask = 1 - semantic_mask
            final_mask = cv2.bitwise_and(final_mask, depth_mask_global)
        else:
            final_mask = depth_mask_global

        output_depth = depth_img.copy()
        output_depth[final_mask == 0] = 0
            
        return output_depth

    def get_data_bundle(self, mode: PCDMode = PCDMode.HUMAN_ONLY):
        if not self.started:
            return None
            
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            aligned_frames = self.align.process(frames)
            
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                return None
            
            img_color = np.asanyarray(color_frame.get_data())
            img_depth = np.asanyarray(depth_frame.get_data())
            
            masked_depth = self._process_hybrid_mask(img_color, img_depth, mode)
            pcd = self._generate_point_cloud(img_color, masked_depth)
            
            return {
                'color': img_color,
                'depth': img_depth,      
                'masked_depth': masked_depth, 
                'pcd': pcd,
                'mode': mode
            }
        except Exception as e:
            print(f"[Error] Frame capture failed: {e}")
            return None

    def stop(self):
        if self.started:
            self.pipeline.stop()
            self.started = False
            print("[System] RealSense Processor Stopped.")


class RealTimeSMPLFitter:
    def __init__(self, model_path, gender='neutral', device='cuda'):
        self.device = torch.device(device)
        
        # Load SMPLX model
        self.model = smplx.create(
            model_path=model_path,
            model_type='smplx',
            gender=gender,
            use_pca=False,
            batch_size=1
        ).to(self.device)
        
        self.transl = torch.zeros((1, 3), dtype=torch.float32, device=self.device, requires_grad=True)
        init_rot = torch.tensor([np.pi, 0, 0], dtype=torch.float32, device=self.device) 
        self.global_orient = init_rot.reshape(1, 3).detach().requires_grad_(True)
        
        self.betas = torch.zeros((1, 10), dtype=torch.float32, device=self.device, requires_grad=True)
        self.body_pose = torch.zeros((1, 63), dtype=torch.float32, device=self.device, requires_grad=True)
        
        self.is_initialized = False

    def fit(self, pcd_points, iterations=5):
        if len(pcd_points) < 10:
            return None

        # Downsample point cloud
        if len(pcd_points) > 800:
            indices = np.random.choice(len(pcd_points), 800, replace=False)
            pcd_subset = pcd_points[indices]
        else:
            pcd_subset = pcd_points
            
        target_tensor = torch.from_numpy(pcd_subset).float().to(self.device)

        # Cold start logic (correct position)
        if not self.is_initialized:
            centroid = torch.mean(target_tensor, dim=0)
            with torch.no_grad():
                self.transl[:] = centroid
                self.transl[0, 1] += 0.2 
            self.is_initialized = True
            current_iters = 30 
        else:
            current_iters = iterations

        optimizer = torch.optim.Adam([self.transl, self.global_orient, self.betas, self.body_pose], lr=0.01)

        for i in range(current_iters):
            optimizer.zero_grad()
            
            output = self.model(
                betas=self.betas,
                global_orient=self.global_orient,
                body_pose=self.body_pose,
                transl=self.transl,
                return_verts=True
            )
            vertices = output.vertices[0]
            
            # Distance calculation
            diff = target_tensor.unsqueeze(1) - vertices.unsqueeze(0) # [N, V, 3]
            if vertices.shape[0] > 2000:
                 v_indices = torch.randperm(vertices.shape[0])[:2000]
                 diff = diff[:, v_indices, :]
            
            dist_sq = (diff ** 2).sum(-1) # [N, V_subset]
            min_dist, _ = torch.min(dist_sq, dim=1)
            loss_fitting = torch.mean(min_dist)
            
            loss_pose = torch.mean(self.body_pose ** 2)
            loss_shape = torch.mean(self.betas ** 2)
            
            loss = loss_fitting * 20.0 + loss_pose * 5.0 + loss_shape * 1.0
            
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            output = self.model(
                betas=self.betas,
                global_orient=self.global_orient,
                body_pose=self.body_pose,
                transl=self.transl
            )
            verts = output.vertices[0].cpu().numpy()
            faces = self.model.faces
            
        return verts, faces
    
    def reset(self):
        self.is_initialized = False
        with torch.no_grad():
            self.betas.fill_(0)
            self.body_pose.fill_(0)
            init_rot = torch.tensor([np.pi, 0, 0], dtype=torch.float32, device=self.device)
            self.global_orient[:] = init_rot