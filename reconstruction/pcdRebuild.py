'''
Author: Seth-Worange
Date: 2026-01-10 19:53:50
LastEditors: Seth-Worange
LastEditTime: 2026-01-15 14:42:33
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
        self.window = self.app.create_window("RealSense Human Reconstruction", 1400, 920)
        self.window.set_on_close(self._on_close)
        self.window.set_on_layout(self._on_layout)
        
        self.sidebar_width = 400
        self.side_panel = gui.Vert(10, gui.Margins(10, 10, 10, 10))
        
        # Existing Controls
        self.mode_combo = gui.Combobox()
        self.mode_combo.add_item("Full Scene")
        self.mode_combo.add_item("Human Only (AI)")
        self.mode_combo.add_item("Background (No Human)")
        self.mode_combo.selected_index = 1 
        self.mode_combo.set_on_selection_changed(self._on_mode_change)
        
        self.rgb_check = gui.Checkbox("Show RGB Texture")
        self.rgb_check.checked = False
        self.rgb_check.set_on_checked(self._on_rgb_toggle)
        
        self.thresh_slider = gui.Slider(gui.Slider.DOUBLE) 
        self.thresh_slider.set_limits(0.5, 5.0) 
        self.thresh_slider.double_value = self.processor.clip_distance_m
        self.thresh_slider.set_on_value_changed(self._on_thresh_change)

        self.rgb_widget = gui.ImageWidget()
        self.depth_widget = gui.ImageWidget()
        self.status_label = gui.Label("Status: Ready")
        
        # Layout Adding
        self.side_panel.add_child(gui.Label("View Mode"))
        self.side_panel.add_child(self.mode_combo)
        self.side_panel.add_child(self.rgb_check)
        self.side_panel.add_child(gui.Label("Geometry Clip Dist (m)"))
        self.side_panel.add_child(self.thresh_slider)

        self.side_panel.add_child(gui.Label("Color Stream"))
        self.side_panel.add_child(self.rgb_widget)
        self.side_panel.add_child(gui.Label("Segmentation Mask"))
        self.side_panel.add_child(self.depth_widget)
        self.side_panel.add_child(self.status_label)
        
        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)
        self.scene_widget.scene.set_background([0.1, 0.1, 0.1, 1.0])
        
        self.window.add_child(self.side_panel)
        self.window.add_child(self.scene_widget)
        
        self.processor.start()
        self.is_running = True
        threading.Thread(target=self.run_loop, daemon=True).start()

    def _on_layout(self, ctx):
        r = self.window.content_rect
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

    def run_loop(self):
        first_run = True
        gray_color = [0.5, 0.5, 0.5]

        while self.is_running:
            t0 = time.time()
            data = self.processor.get_data_bundle(mode=self.current_mode)
            process_time = (time.time() - t0) * 1000
            
            if data is None:
                time.sleep(0.01)
                continue
            
            pcd = data['pcd']
            
            if len(pcd.points) > 0 and not self.show_rgb:
                pcd.paint_uniform_color(gray_color)
                
            # UI Preview logic
            preview_w = self.sidebar_width - 20
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
                
                mat = rendering.MaterialRecord()
                mat.shader = "defaultUnlit"
                mat.point_size = 3.0 
                scene.add_geometry("pcd", pcd, mat)
                
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