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
        self.width = width
        self.height = height
        self.fps = fps
        
        # RealSense Pipeline Setup
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        
        self.align = rs.align(rs.stream.color)
        self.profile = None
        self.started = False
        
        self.pinhole_intrinsics = None
        self.depth_scale = 0.001 
        self.clip_distance_m = 3.0 
        self.voxel_size = 0.02
        self.human_depth_tolerance = 0.7 

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
        dist = depth_frame.get_distance(x, y)
        if dist <= 0 or dist > 3.0: return None
        return rs.rs2_deproject_pixel_to_point(intr, [x, y], dist)

    # Execute inference, return: (segmentation mask, skeleton_data dict)
    def _analyze_pose_and_keypoints(self, color_img, depth_frame):
        default_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        
        # Structure for UI visualization
        skeleton_data = {
            'pose': [],
            'hands': [], # List of list of points
            'face': []   # List of points
        }
        
        # Structure for SMPL fitting
        keypoints_3d_smpl = [] 

        if not self.mp_pose:
            return default_mask, skeleton_data, keypoints_3d_smpl

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
                    # Add to SMPL list
                    keypoints_3d_smpl.append([idx, p3d[0], p3d[1], p3d[2]])
                else:
                    curr_pose_3d.append(None) # Keep index structure for drawing lines
            skeleton_data['pose'] = curr_pose_3d

        # --- B. Hand Detection ---
        if self.mp_hand:
            hand_result = self.mp_hand.detect(mp_image)
            for hand_lms in hand_result.hand_landmarks:
                curr_hand_3d = []
                for lm in hand_lms:
                    px, py = int(lm.x * self.width), int(lm.y * self.height)
                    p3d = self._deproject_pixel(px, py, depth_frame, intr)
                    curr_hand_3d.append(p3d if p3d else None)
                skeleton_data['hands'].append(curr_hand_3d)

        # --- C. Face Detection ---
        if self.mp_face:
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
        
        # Use Pose landmarks (0-10) if high-res face not detected
        if not skeleton_data['face'] and len(skeleton_data['pose']) >= 11:
            skeleton_data['face'] = skeleton_data['pose'][0:11]
                            
        return segmentation_mask, skeleton_data, keypoints_3d_smpl

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
            
            img_color = np.asanyarray(color_frame.get_data())
            img_depth = np.asanyarray(depth_frame.get_data())
            
            # 1. Call new function to get Mask and keypoints
            semantic_mask, skeleton_data, keypoints_3d_smpl = self._analyze_pose_and_keypoints(img_color, depth_frame)
            
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
                'keypoints_3d': keypoints_3d_smpl 
            }
        except Exception as e:
            print(f"[Error] Process failed: {e}")
            return None


class RealTimeSMPLFitter:
    def __init__(self, model_path, gender='neutral', device='cuda'):
        self.device = torch.device(device) if torch.cuda.is_available() else torch.device('cpu')
        self.model = smplx.create(
            model_path=model_path, model_type='smplx', gender=gender,
            use_pca=False, batch_size=1
        ).to(self.device)
        
        self.transl = torch.zeros((1, 3), dtype=torch.float32, device=self.device, requires_grad=True)
        # Initial rotation: 180 degrees around X axis
        init_rot = torch.tensor([np.pi, 0, 0], dtype=torch.float32, device=self.device) 
        self.global_orient = init_rot.reshape(1, 3).detach().requires_grad_(True)
        self.betas = torch.zeros((1, 10), dtype=torch.float32, device=self.device, requires_grad=True)
        self.body_pose = torch.zeros((1, 63), dtype=torch.float32, device=self.device, requires_grad=True)
        self.is_initialized = False

        # MediaPipe ID -> SMPL-X Joint ID (Expanded)
        self.joint_map = {
            # Arms
            15: 20, 16: 21, 13: 18, 14: 19,
            11: 16, 12: 17, # Shoulders
            # Legs & Hips
            23: 1, 24: 2,   # Hips
            25: 4, 26: 5,   # Knees
            27: 7, 28: 8    # Ankles
        }

    def fit(self, pcd_points, keypoints_3d=None, iterations=5):
        if len(pcd_points) < 10: return None

        # 1. Downsample point cloud
        if len(pcd_points) > 400:
            indices = np.random.choice(len(pcd_points), 400, replace=False)
            pcd_subset = pcd_points[indices]
        else:
            pcd_subset = pcd_points
        target_tensor = torch.from_numpy(pcd_subset).float().to(self.device)

        # 2. Process 3D joints
        has_joints = False
        smpl_joint_indices = None
        target_joint_coords = None
        match_count = 0
        
        # Helper to find specific body parts
        hip_targets = []
        
        if keypoints_3d and len(keypoints_3d) > 0:
            smpl_indices_list = []
            target_coords_list = []
            for kp in keypoints_3d:
                mp_idx, x, y, z = kp
                if mp_idx in self.joint_map:
                    s_idx = self.joint_map[mp_idx]
                    smpl_indices_list.append(s_idx)
                    target_coords_list.append([x, y, z])
                    match_count += 1
                    
                    # Collect hips (SMPL 1 and 2) for smart init
                    if s_idx == 1 or s_idx == 2:
                        hip_targets.append([x, y, z])
            
            if len(smpl_indices_list) > 0:
                smpl_joint_indices = torch.tensor(smpl_indices_list, dtype=torch.long, device=self.device)
                target_joint_coords = torch.tensor(target_coords_list, dtype=torch.float32, device=self.device)
                has_joints = True
        
        if match_count < 3 and keypoints_3d:
             pass

        # 3. Smart Initialization
        if len(hip_targets) > 0:
            hip_center = np.mean(hip_targets, axis=0)
            hip_center_tensor = torch.tensor(hip_center, dtype=torch.float32, device=self.device)
            
            # If not initialized, or if optimization drifted too far (optional check), reset
            if not self.is_initialized:
                with torch.no_grad():
                    self.transl[:] = hip_center_tensor
                    # self.transl[0, 1] += 0.0 # Hips are roughly at pelvis center
                self.is_initialized = True
                current_iters = 30
            else:
                # Soft Anchor: Gently nudge translation to target (30% blend)
                # This fixes long-term drift without causing high-frequency jitter (Hard Alignment)
                with torch.no_grad():
                     self.transl[:] = self.transl * 0.7 + hip_center_tensor * 0.3
                current_iters = iterations
        else:
            # Fallback to centroid if no hips detected
            if not self.is_initialized:
                centroid = torch.mean(target_tensor, dim=0)
                with torch.no_grad():
                    self.transl[:] = centroid
                    self.transl[0, 1] += 0.2 
                self.is_initialized = True
                current_iters = 30
            else:
                current_iters = iterations

        # 4. Optimization loop
        optimizer = torch.optim.Adam([self.transl, self.global_orient, self.betas, self.body_pose], lr=0.02)

        for i in range(current_iters):
            optimizer.zero_grad()
            
            # return_joints=True
            output = self.model(
                betas=self.betas, global_orient=self.global_orient,
                body_pose=self.body_pose, transl=self.transl,
                return_verts=True
            )
            vertices = output.vertices[0]
            joints = output.joints[0] # SMPL joints
            
            # Loss 1: Point cloud fitting (Scan-to-Mesh)
            diff = target_tensor.unsqueeze(1) - vertices.unsqueeze(0)
            # Random sampling for acceleration
            if vertices.shape[0] > 1000:
                 v_indices = torch.randperm(vertices.shape[0])[:1000]
                 diff = diff[:, v_indices, :]
            
            dist_sq = (diff ** 2).sum(-1)
            loss_fitting = torch.mean(torch.min(dist_sq, dim=1)[0])
            
            # Loss 2: Joint anchor loss
            loss_joints = torch.tensor(0.0, device=self.device)
            if has_joints:
                current_joints_subset = joints[smpl_joint_indices]
                loss_joints = torch.nn.functional.mse_loss(current_joints_subset, target_joint_coords)
            
            # Loss 3: Regularization
            loss_pose = torch.mean(self.body_pose ** 2)
            loss_shape = torch.mean(self.betas ** 2)
            
            # Weight allocation:
            loss = loss_fitting * 20.0 + loss_joints * 500.0 + loss_pose * 0.1 + loss_shape * 1.0
            
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            output = self.model(
                betas=self.betas, global_orient=self.global_orient,
                body_pose=self.body_pose, transl=self.transl
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