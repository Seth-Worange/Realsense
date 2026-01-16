'''
Author: Seth-Worange
Date: 2026-01-10 19:53:50
LastEditors: Seth-Worange
LastEditTime: 2026-01-16 18:01:37
FilePath: \Realsense\reconstruction\pcdRebuild.py
Description: 
    Reconstruct point clouds from RealSense D435, extract and process foreground objects.
    MediaPipe version: 0.10.31
'''

from utils import RealSenseProcessor, PCDMode, RealTimeSMPLFitter
import numpy as np
import cv2
import open3d as o3d
from open3d.visualization import gui
from open3d.visualization import rendering
import time
import threading


# ==========================================
# UI Layer: HumanSegUI
# ==========================================
class HumanSegUI:
    def __init__(self):
        self.processor = RealSenseProcessor()
        self.is_running = False
        self.current_mode = PCDMode.HUMAN_ONLY 
        
        self.show_rgb = False
        
        self.vis_min_dist = 0.0
        self.vis_max_dist = 4.0

        self.app = gui.Application.instance
        self.app.initialize()
        self.window = self.app.create_window("RealSense Human Reconstruction", 1400, 950)
        self.window.set_on_close(self._on_close)
        self.window.set_on_layout(self._on_layout)
        
        self.sidebar_width = 450
        self.side_panel = gui.ScrollableVert(0, gui.Margins(10, 10, 10, 10))
        
        # --- Group 1: View Options ---
        self.view_group = gui.CollapsableVert("View Options", 0.3 * 16, gui.Margins(5,5,5,5))
        self.view_group.set_is_open(True)
        
        self.mode_combo = gui.Combobox()
        self.mode_combo.add_item("Full Scene")
        self.mode_combo.add_item("Human Only (AI)")
        self.mode_combo.add_item("Background (No Human)")
        self.mode_combo.selected_index = 1 
        self.mode_combo.set_on_selection_changed(self._on_mode_change)
        
        # Display toggles in a horizontal row for compactness
        display_row = gui.Horiz(5)
        
        self.pcd_check = gui.Checkbox("Point Cloud")
        self.pcd_check.checked = True
        self.show_pcd = True
        self.pcd_check.set_on_checked(self._on_pcd_toggle)
        
        self.rgb_check = gui.Checkbox("Texture")
        self.rgb_check.checked = False
        self.rgb_check.set_on_checked(self._on_rgb_toggle)

        self.skel_check = gui.Checkbox("Skeleton")
        self.skel_check.checked = False
        self.show_skeleton = False
        self.skel_check.set_on_checked(self._on_skeleton_toggle)
        
        # self.view_group.add_child(gui.Label("View Mode"))
        self.view_group.add_child(self.mode_combo)
        
        display_row.add_child(self.pcd_check)
        display_row.add_child(self.rgb_check)
        display_row.add_child(self.skel_check)
        self.view_group.add_child(display_row)

        # --- Group 2: Reconstruction ---
        self.recon_group = gui.CollapsableVert("Reconstruction", 0.3 * 16, gui.Margins(5,5,5,5))
        self.recon_group.set_is_open(True)
        
        smpl_path = 'C:/OrangeFiles/科研/Realsense/reconstruction/models/smplx/SMPLX_NEUTRAL.npz'
        try:
            self.smpl_fitter = RealTimeSMPLFitter(smpl_path) 
            self.show_smpl = False
        except Exception as e:
            print(f"SMPL Init Warning: {e}")
            self.smpl_fitter = None
        
        recon_row = gui.Horiz(5)
        
        # Use spaces for approximate vertical alignment with top row
        self.smpl_check = gui.Checkbox("SMPL        ") 
        self.smpl_check.set_on_checked(lambda c: setattr(self, 'show_smpl', c))
        
        self.face_check = gui.Checkbox("Face     ")
        self.face_check.checked = False 
        self.face_check.set_on_checked(self._on_face_toggle)
        self.processor.enable_face = False # Sync initial state

        self.hand_check = gui.Checkbox("Hand   ")
        self.hand_check.checked = False
        self.hand_check.set_on_checked(self._on_hand_toggle)
        self.processor.enable_hand = False # Sync initial state

        recon_row.add_child(self.smpl_check)
        recon_row.add_child(self.face_check)
        recon_row.add_child(self.hand_check)
        self.recon_group.add_child(recon_row)

        # Add Collapsable Groups FIRST
        self.side_panel.add_child(self.view_group)
        self.side_panel.add_child(self.recon_group)

        # --- Parameters (Flat) ---
        
        param_row = gui.Horiz(10)
        param_row.add_child(gui.Label("Clip(m)"))
        self.thresh_slider = gui.Slider(gui.Slider.DOUBLE) 
        self.thresh_slider.set_limits(0.5, 5.0) 
        self.thresh_slider.double_value = self.processor.clip_distance_m
        self.thresh_slider.set_on_value_changed(self._on_thresh_change)
        
        param_row.add_child(self.thresh_slider)
        self.side_panel.add_child(param_row)

        # --- Stream Monitor (Flat) ---
        self.side_panel.add_child(gui.Label("")) # Filler for spacing

        self.rgb_widget = gui.ImageWidget()
        self.depth_widget = gui.ImageWidget()
        self.status_label = gui.Label("Status: Ready")
        
        self.side_panel.add_child(gui.Label("Color Stream"))
        self.side_panel.add_child(self.rgb_widget)
        self.side_panel.add_child(gui.Label("Segmentation Mask"))
        self.side_panel.add_child(self.depth_widget)
        self.side_panel.add_child(self.status_label)
        
        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)
        self.scene_widget.scene.set_background([0.02, 0.02, 0.05, 1.0])
        
        self.window.add_child(self.side_panel)
        self.window.add_child(self.scene_widget)
        
        self.processor.start()
        self.is_running = True
        threading.Thread(target=self.run_loop, daemon=True).start()

    def _on_layout(self, ctx):
        r = self.window.content_rect
        # Responsive: Sidebar is 33% of window width
        self.sidebar_width = int(r.width * 0.3)
        # Minimum width check
        if self.sidebar_width < 300: self.sidebar_width = 300
        
        self.side_panel.frame = gui.Rect(r.x, r.y, self.sidebar_width, r.height)
        self.scene_widget.frame = gui.Rect(r.x + self.sidebar_width, r.y, r.width - self.sidebar_width, r.height)

    def _on_close(self):
        self.is_running = False
        self.processor.stop()
        return True
        
    def _on_mode_change(self, new_val, new_idx):
        modes = [PCDMode.FULL, PCDMode.HUMAN_ONLY, PCDMode.BACKGROUND]
        self.current_mode = modes[new_idx]
        
    def _on_thresh_change(self, new_val):
        self.processor.clip_distance_m = new_val

    def _on_rgb_toggle(self, is_checked):
        self.show_rgb = is_checked

    def _on_pcd_toggle(self, is_checked):
        self.show_pcd = is_checked

    def _on_skeleton_toggle(self, is_checked):
        self.show_skeleton = is_checked

    def _on_face_toggle(self, is_checked):
        self.processor.enable_face = is_checked

    def _on_hand_toggle(self, is_checked):
        self.processor.enable_hand = is_checked

    def run_loop(self):
        first_run = True
        # Aesthetic Colors
        pcd_color = [0.0, 0.6, 0.8] # Metallic Blue
        smpl_mesh = None
        
        # Connections
        POSE_CONN = [[11,12],[11,13],[13,15],[12,14],[14,16],[11,23],[12,24],[23,24],[23,25],[24,26],[25,27],[26,28],[27,29],[28,30]]
        HAND_CONN = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]]

        while self.is_running:
            t0 = time.time()
            data = self.processor.get_data_bundle(mode=self.current_mode)
            process_time = (time.time() - t0) * 1000
            
            if data is None:
                time.sleep(0.01)
                continue
            
            pcd = data['pcd']
            
            if len(pcd.points) > 0 and not self.show_rgb:
                pcd.paint_uniform_color(pcd_color)

            # SMPL fitting logic
            fitted_geom = None
            # Only run when: 1. User enables SMPL display 2. Fitter is initialized 3. Point cloud has enough points
            if self.show_smpl and hasattr(self, 'smpl_fitter') and self.smpl_fitter is not None and len(pcd.points) > 50:
                points_np = np.asarray(pcd.points)
                kp_info = data.get('keypoints_info', None)
                
                # Extract intrinsics for 2D reprojection
                intr = self.processor.pinhole_intrinsics.intrinsic_matrix
                w, h = self.processor.width, self.processor.height
                intr_params = (intr[0,0], intr[1,1], intr[0,2], intr[1,2], w, h)
                
                result = self.smpl_fitter.fit(points_np, keypoints_info=kp_info, intrinsics=intr_params, iterations=5) 
                
                if result:
                    verts, faces = result
                    if smpl_mesh is None:
                        smpl_mesh = o3d.geometry.TriangleMesh()
                    smpl_mesh.vertices = o3d.utility.Vector3dVector(verts)
                    smpl_mesh.triangles = o3d.utility.Vector3iVector(faces)
                    smpl_mesh.compute_vertex_normals()
                    smpl_mesh.paint_uniform_color([1, 0.8, 0.6]) # Set to skin color
                    fitted_geom = smpl_mesh
            
            # Skeleton Rendering
            skel_geoms = {} # name -> geometry
            if self.show_skeleton:
                skel_data = data.get('skeleton_data', {})
                
                # 1. Pose
                if 'pose' in skel_data and skel_data['pose']:
                    points = skel_data['pose']
                    valid_inds = [i for i, p in enumerate(points) if p is not None]
                    if len(valid_inds) > 0:
                        dense_points = [points[i] for i in valid_inds]
                        map_idx = {orig: new for new, orig in enumerate(valid_inds)}
                        
                        lines = []
                        for s, e in POSE_CONN:
                            if s in map_idx and e in map_idx:
                                lines.append([map_idx[s], map_idx[e]])
                        
                        if lines:
                            ls = o3d.geometry.LineSet()
                            ls.points = o3d.utility.Vector3dVector(dense_points)
                            ls.lines = o3d.utility.Vector2iVector(lines)
                            ls.paint_uniform_color([1.0, 0.84, 0.0]) # Gold body
                            skel_geoms['skel_pose'] = ls

                # 2. Hands
                if 'hands' in skel_data:
                    for i, hand_pts in enumerate(skel_data['hands']):
                        valid_inds = [k for k, p in enumerate(hand_pts) if p is not None]
                        if len(valid_inds) > 0:
                            dense_points = [hand_pts[k] for k in valid_inds]
                            map_idx = {orig: new for new, orig in enumerate(valid_inds)}
                            lines = []
                            for s, e in HAND_CONN:
                                if s in map_idx and e in map_idx:
                                    lines.append([map_idx[s], map_idx[e]])
                            if lines:
                                ls = o3d.geometry.LineSet()
                                ls.points = o3d.utility.Vector3dVector(dense_points)
                                ls.lines = o3d.utility.Vector2iVector(lines)
                                ls.paint_uniform_color([1.0, 0.0, 0.5]) # Hot Pink hands
                                skel_geoms[f'skel_hand_{i}'] = ls
                
                # 3. Face
                if 'face' in skel_data and skel_data['face']:
                    face_pts = [p for p in skel_data['face'] if p is not None]
                    if len(face_pts) > 0:
                        pc = o3d.geometry.PointCloud()
                        pc.points = o3d.utility.Vector3dVector(face_pts)
                        pc.paint_uniform_color([0.8, 1.0, 1.0]) # Cyan face dots
                        skel_geoms['skel_face'] = pc

            # UI Preview logic
            # Dynamic resizing based on current sidebar width
            preview_w = self.sidebar_width - 40 # Margin adjustment
            if preview_w < 100: preview_w = 100
            preview_h = int(preview_w * 0.75)
            
            rgb_small = cv2.resize(data['color'], (preview_w, preview_h))
            rgb_small = cv2.cvtColor(rgb_small, cv2.COLOR_BGR2RGB)
            
            masked_depth_vis = data['masked_depth']
            depth_color = cv2.applyColorMap(
                cv2.convertScaleAbs(masked_depth_vis, alpha=0.03),
                cv2.COLORMAP_JET
            )
            
            if self.current_mode == PCDMode.HUMAN_ONLY:
                cv2.putText(depth_color, "Human Segmented", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
            depth_small = cv2.resize(depth_color, (preview_w, preview_h))
            depth_small = cv2.cvtColor(depth_small, cv2.COLOR_BGR2RGB)
            
            o3d_rgb = o3d.geometry.Image(np.ascontiguousarray(rgb_small))
            o3d_depth = o3d.geometry.Image(np.ascontiguousarray(depth_small))
            
            def update():
                if not self.is_running: return
                self.rgb_widget.update_image(o3d_rgb)
                self.depth_widget.update_image(o3d_depth)
                self.status_label.text = f"Proc Time: {process_time:.1f}ms | Points: {len(pcd.points)}"
                
                scene = self.scene_widget.scene
                if scene.has_geometry("pcd"): 
                    scene.remove_geometry("pcd")
                
                if self.show_pcd:
                    mat = rendering.MaterialRecord()
                    mat.shader = "defaultUnlit"
                    mat.point_size = 3.0 
                    scene.add_geometry("pcd", pcd, mat)

                # SMPL Rendering
                if scene.has_geometry("smpl"):
                    scene.remove_geometry("smpl")
                
                if fitted_geom is not None:
                    mat_smpl = rendering.MaterialRecord()
                    mat_smpl.shader = "defaultLit" # Use lit shader for better muscle visibility
                    scene.add_geometry("smpl", fitted_geom, mat_smpl)
                
                # Skeleton Rendering Update
                for name in ["skel_pose", "skel_face", "skel_hand_0", "skel_hand_1"]:
                     if scene.has_geometry(name): scene.remove_geometry(name)
 
                if self.show_skeleton:
                    mat_line = rendering.MaterialRecord()
                    mat_line.shader = "unlitLine"
                    mat_line.line_width = 3.0
                    
                    mat_pt = rendering.MaterialRecord()
                    mat_pt.shader = "defaultUnlit"
                    mat_pt.point_size = 5.0

                    for name, geom in skel_geoms.items():
                        if isinstance(geom, o3d.geometry.LineSet):
                             scene.add_geometry(name, geom, mat_line)
                        else:
                             scene.add_geometry(name, geom, mat_pt)

                nonlocal first_run
                if first_run and len(pcd.points) > 0:
                    bounds = scene.bounding_box
                    self.scene_widget.setup_camera(60, bounds, bounds.get_center())
                    self.scene_widget.look_at(
                        np.array([0, 0, 1.0]),
                        np.array([0, -0.5, -0.5]),
                        np.array([0, -1, 0])
                    )
                    first_run = False
            
            self.app.post_to_main_thread(self.window, update)
            

    def run(self):
        self.app.run()

if __name__ == "__main__":
    app = HumanSegUI()
    app.run()