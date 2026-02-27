import numpy as np
import torch
import smplx

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
        
        # Motion Smoothness buffers
        self.prev_pose = None
        self.prev_betas = None
        self.prev_transl = None
        self.prev_orient = None

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

    def project_points(self, points_3d, intrinsics):
        # intrinsics: [fx, fy, cx, cy]
        fx, fy, cx, cy = intrinsics
        # Z should not be zero
        z = points_3d[:, 2].clamp(min=1e-3)
        x = points_3d[:, 0] * fx / z + cx
        y = points_3d[:, 1] * fy / z + cy
        return torch.stack([x, y], dim=-1)

    def fit(self, pcd_points, keypoints_info=None, intrinsics=None, iterations=5):
        if len(pcd_points) < 10: return None

        # 1. Downsample point cloud
        if len(pcd_points) > 400:
            indices = np.random.choice(len(pcd_points), 400, replace=False)
            pcd_subset = pcd_points[indices]
        else:
            pcd_subset = pcd_points
        target_tensor = torch.from_numpy(pcd_subset).float().to(self.device)

        # 2. Process Keypoints (Hybrid 2D/3D)
        has_joints = False
        smpl_joint_indices = None
        target_joint_3d = None
        target_joint_2d = None
        joint_confidence = None
        
        hip_targets = []
        torso_smpl_indices = [] # For Rigid Phase
        torso_target_indices = []
        
        if keypoints_info and len(keypoints_info) > 0:
            s_indices = []
            t_3d = []
            t_2d = []
            vis = []
            
            # intrinsics tuple: (fx, fy, cx, cy, w, h)
            width = intrinsics[4] if intrinsics else 640
            height = intrinsics[5] if intrinsics else 480
            
            for i, item in enumerate(keypoints_info):
                # item: [mp_idx, x, y, z, u_norm, v_norm, visibility]
                mp_idx = item[0]
                if mp_idx in self.joint_map:
                    s_idx = self.joint_map[mp_idx]
                    s_indices.append(s_idx)
                    t_3d.append([item[1], item[2], item[3]])
                    # Convert Norm 2D -> Pixel 2D
                    t_2d.append([item[4] * width, item[5] * height])
                    vis.append(item[6]) # Visibility score
                    
                    # Store Hips for init
                    if s_idx == 1 or s_idx == 2:
                        hip_targets.append([item[1], item[2], item[3]])
                    
                    # Store Torso (Hips + Shoulders) for Rigid Phase
                    # SMPL: 1,2 (Hips), 16,17 (Shoulders)
                    if s_idx in [1, 2, 16, 17]:
                        torso_smpl_indices.append(s_idx)
                        # We need the index in the *subset* tensor, which is simply 'len(t_3d)-1'
                        torso_target_indices.append(len(t_3d)-1)

            if len(s_indices) > 0:
                smpl_joint_indices = torch.tensor(s_indices, dtype=torch.long, device=self.device)
                target_joint_3d = torch.tensor(t_3d, dtype=torch.float32, device=self.device)
                target_joint_2d = torch.tensor(t_2d, dtype=torch.float32, device=self.device)
                joint_confidence = torch.tensor(vis, dtype=torch.float32, device=self.device).unsqueeze(1)
                has_joints = True

        # 3. Smart Initialization (Soft Anchor)
        if len(hip_targets) > 0:
            hip_center = np.mean(hip_targets, axis=0)
            hip_center_tensor = torch.tensor(hip_center, dtype=torch.float32, device=self.device)
            if not self.is_initialized:
                with torch.no_grad():
                    self.transl[:] = hip_center_tensor
                self.is_initialized = True
            else:
                with torch.no_grad():
                     self.transl[:] = self.transl * 0.7 + hip_center_tensor * 0.3

        # 4. Optimization Strategy
        if has_joints and len(torso_smpl_indices) >= 2:
             rigid_optimizer = torch.optim.Adam([self.transl, self.global_orient], lr=0.05)
             
             t_torso_idxs = torch.tensor(torso_target_indices, dtype=torch.long, device=self.device)
             
             for _ in range(5):
                 rigid_optimizer.zero_grad()
                 output = self.model(
                     betas=self.betas.detach(),
                     body_pose=self.body_pose.detach(), # Frozen
                     global_orient=self.global_orient,
                     transl=self.transl
                 )
                 joints = output.joints[0]
                 
                 # Only match torso joints in 3D
                 curr_torso = joints[torch.tensor(torso_smpl_indices, device=self.device)]
                 targ_torso = target_joint_3d[t_torso_idxs]
                 
                 loss_rigid = torch.nn.functional.mse_loss(curr_torso, targ_torso)
                 loss_rigid.backward()
                 rigid_optimizer.step()

        # Phase 2: Full Articulation (Hybrid Loss)
        optimizer = torch.optim.Adam([self.transl, self.global_orient, self.betas, self.body_pose], lr=0.02)
        
        intr_tensor = None
        if intrinsics:
            # Extract [fx, fy, cx, cy]
            intr_tensor = torch.tensor(intrinsics[0:4], dtype=torch.float32, device=self.device)

        for i in range(iterations):
            optimizer.zero_grad()
            
            output = self.model(
                betas=self.betas, global_orient=self.global_orient,
                body_pose=self.body_pose, transl=self.transl,
                return_verts=True
            )
            vertices = output.vertices[0]
            joints = output.joints[0]
            
            # Loss 1: Scan-to-Mesh (PCD)
            diff = target_tensor.unsqueeze(1) - vertices.unsqueeze(0)
            if vertices.shape[0] > 1000:
                 v_indices = torch.randperm(vertices.shape[0])[:1000]
                 diff = diff[:, v_indices, :]
            dist_sq = (diff ** 2).sum(-1)
            loss_fitting = torch.mean(torch.min(dist_sq, dim=1)[0])
            
            loss_joints_3d = torch.tensor(0.0, device=self.device)
            loss_joints_2d = torch.tensor(0.0, device=self.device)
            
            if has_joints:
                # 3D Joint Loss (Weighted by confidence)
                curr_joints = joints[smpl_joint_indices]
                dist_3d = (curr_joints - target_joint_3d) ** 2
                loss_joints_3d = torch.mean(dist_3d * joint_confidence)
                
                # 2D Reprojection Loss (The FrankMocap Special)
                if intr_tensor is not None:
                    proj_2d = self.project_points(curr_joints, intr_tensor)
                    dist_2d = (proj_2d - target_joint_2d) ** 2
                    # Pixel coordinates are large (e.g. 500px), MSE will be huge (250000).
                    # Normalize or scale down. Let's scale down by 1e-4.
                    loss_joints_2d = torch.mean(dist_2d * joint_confidence) * 1e-4

            # Loss 3: Regularization
            loss_pose = torch.mean(self.body_pose ** 2)
            loss_shape = torch.mean(self.betas ** 2)
            
            # Loss 4: Motion Smoothness
            loss_smooth = torch.tensor(0.0, device=self.device)
            if self.prev_pose is not None:
                pose_diff = torch.mean((self.body_pose - self.prev_pose) ** 2)
                transl_diff = torch.mean((self.transl - self.prev_transl) ** 2)
                orient_diff = torch.mean((self.global_orient - self.prev_orient) ** 2)
                loss_smooth = (pose_diff + transl_diff + orient_diff)

            # Hybrid Weights
            # 3D Joints: 500.0 (Strong depth constraint)
            # 2D Joints: 50.0 (Strong visual lock, but 3D is primary for depth)
            # Fitting: 20.0 (Surface detail)
            # Pose: 0.1 (Regularization, reduced to allow movement)
            # Smooth: 5.0 (Penalize jitter)
            loss = (loss_fitting * 20.0 + 
                    loss_joints_3d * 500.0 + 
                    loss_joints_2d * 50.0 + 
                    loss_pose * 0.1 + 
                    loss_shape * 1.0 +
                    loss_smooth * 5.0)
            
            loss.backward()
            optimizer.step()

        # Save current state for next frame smoothness
        with torch.no_grad():
            self.prev_pose = self.body_pose.detach().clone()
            self.prev_betas = self.betas.detach().clone()
            self.prev_transl = self.transl.detach().clone()
            self.prev_orient = self.global_orient.detach().clone()

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