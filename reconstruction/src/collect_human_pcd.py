import os
import time
import threading
import open3d as o3d
from open3d.visualization import gui
from open3d.visualization import rendering
import numpy as np

from utils.rs_config import RSConfig
from utils.realsense import RealSenseProcessor, PCDMode

class PointCloudCollectorUI:
    def __init__(self):
        print("====================================")
        print(" Human Point Cloud Sequential Collector ")
        print("====================================")
        
        config = RSConfig()
        config.voxel_size = 0.02
        config.human_depth_tolerance = 0.5
        
        print("[1] 初始化 RealSense ...")
        self.processor = RealSenseProcessor(rs_config=config)
        self.is_running = False
        
        # Recording State
        self.is_recording = False
        self.target_fps = 10.0
        self.duration_limit = 0.0  # 0 means infinite until manual stop
        self.save_interval = 1.0 / self.target_fps
        
        self.recording_start_time = 0.0
        self.last_save_time = 0.0
        self.saved_count = 0
        self.seq_dir = ""
        
        self.base_save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datas", "rs_pointcloud")
        if not os.path.exists(self.base_save_dir):
            os.makedirs(self.base_save_dir)

        # UI Setup
        self.app = gui.Application.instance
        self.app.initialize()
        self.window = self.app.create_window("Point Cloud Collector", 1000, 700)
        self.window.set_on_close(self._on_close)
        self.window.set_on_layout(self._on_layout)
        
        self.sidebar_width = 300
        self.side_panel = gui.ScrollableVert(0, gui.Margins(10, 10, 10, 10))
        
        # --- Settings Panel ---
        self.settings_group = gui.CollapsableVert("Recording Settings", 0.5 * 16, gui.Margins(5,5,5,5))
        self.settings_group.set_is_open(True)
        
        # FPS Number Edit
        fps_layout = gui.Horiz(10)
        fps_layout.add_child(gui.Label("Target FPS: "))
        self.fps_edit = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.fps_edit.double_value = self.target_fps
        self.fps_edit.set_limits(1, 40)
        self.fps_edit.set_on_value_changed(self._on_fps_changed)
        fps_layout.add_child(self.fps_edit)
        self.settings_group.add_child(fps_layout)
        
        # Duration Number Edit
        dur_layout = gui.Horiz(10)
        dur_layout.add_child(gui.Label("Duration (s, 0=inf): "))
        self.dur_edit = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.dur_edit.double_value = self.duration_limit
        self.dur_edit.set_limits(0, 3600)
        self.dur_edit.set_on_value_changed(self._on_dur_changed)
        dur_layout.add_child(self.dur_edit)
        self.settings_group.add_child(dur_layout)
        
        self.side_panel.add_child(self.settings_group)
        
        # --- Status & Control Panel ---
        self.control_group = gui.CollapsableVert("Control", 0.5 * 16, gui.Margins(5,5,5,5))
        self.control_group.set_is_open(True)
        
        self.start_btn = gui.Button("     Start Recording      ")
        self.start_btn.set_on_clicked(self._on_start_stop)
        self.control_group.add_child(self.start_btn)
        
        self.status_label = gui.Label("Status: STANDBY         ")
        self.frames_label = gui.Label("Frames Saved: 0")
        self.time_label = gui.Label("Elapsed: 0.0s")
        self.points_label = gui.Label("Points: 0")
        
        self.control_group.add_child(gui.Label("")) # spacing
        self.control_group.add_child(self.status_label)
        self.control_group.add_child(self.frames_label)
        self.control_group.add_child(self.time_label)
        self.control_group.add_child(self.points_label)
        
        self.side_panel.add_child(self.control_group)

        self.side_panel.add_child(gui.Label(""))
        self.side_panel.add_child(gui.Label("Press the start button"))
        self.side_panel.add_child(gui.Label("to record into datas/"))
        
        # --- 3D Scene Widget ---
        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)
        self.scene_widget.scene.set_background([0.1, 0.1, 0.15, 1.0])
        self.scene_widget.scene.show_axes(True)
        
        self.window.add_child(self.side_panel)
        self.window.add_child(self.scene_widget)
        
        # Hardware start
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
        print("硬件句柄释放完毕。")
        return True

    def _on_fps_changed(self, val):
        self.target_fps = val
        self.save_interval = 1.0 / self.target_fps
        print(f"Target FPS updated to: {val}")
        
    def _on_dur_changed(self, val):
        self.duration_limit = val
        print(f"Target duration limit updated to: {val}s")

    def _on_start_stop(self):
        if not self.is_recording:
            # Start
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.seq_dir = os.path.join(self.base_save_dir, f"seq_{timestamp}")
            os.makedirs(self.seq_dir, exist_ok=True)
            
            self.is_recording = True
            self.recording_start_time = time.time()
            self.last_save_time = self.recording_start_time
            self.saved_count = 0
            
            self.start_btn.text = "Stop Recording"
            self.status_label.text = "Status: RECORDING..."
            self.status_label.text_color = gui.Color(1.0, 0.0, 0.0) # Red
            print(f"\n[录制开始] 保存至 {self.seq_dir} | target_fps={self.target_fps}, duration_limit={self.duration_limit}s")
        else:
            # Stop
            self.is_recording = False
            self.start_btn.text = "Start Recording"
            self.status_label.text = "Status: STANDBY"
            self.status_label.text_color = gui.Color(1.0, 1.0, 1.0) # White
            print(f"[录制结束] 用户手动结束。总采集帧数: {self.saved_count}")

    def run_loop(self):
        first_run = True
        mat = rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        mat.point_size = 4.0 
        
        while self.is_running:
            # For accurate FPS control, check time BEFORE doing expensive processing
            current_time = time.time()
            if self.is_recording and self.save_interval > 0:
                elapsed_since_last = current_time - self.last_save_time
                if elapsed_since_last < self.save_interval:
                    # Too early to acquire the next targeted frame, sleep tightly
                    time.sleep(max(0.001, self.save_interval - elapsed_since_last - 0.002))
                    continue
                    
            data = self.processor.get_data_bundle(mode=PCDMode.HUMAN_ONLY)
            if data is None:
                time.sleep(0.005)
                continue
            
            pcd = data['pcd']
            pts_count = len(pcd.points)
            
            pcd.paint_uniform_color([0.0, 0.8, 0.4]) # Aesthetic green
            
            current_time = time.time()
            elapsed_time = 0.0
            
            if self.is_recording:
                elapsed_time = current_time - self.recording_start_time
                
                # Auto stop logic
                if self.duration_limit > 0 and elapsed_time >= self.duration_limit:
                    print(f"\n[录制自动结束] 达到设定时长 {self.duration_limit}s。总采集帧数: {self.saved_count}")
                    self.is_recording = False
                    
                    def ui_stop_update():
                        self.start_btn.text = "Start Recording"
                        self.status_label.text = "Status: STANDBY"
                        self.status_label.text_color = gui.Color(1.0, 1.0, 1.0)
                    self.app.post_to_main_thread(self.window, ui_stop_update)
                
                # Save frame
                if self.is_recording:
                    if pts_count > 0:
                        filename = os.path.join(self.seq_dir, f"frame_{self.saved_count:04d}.ply")
                        o3d.io.write_point_cloud(filename, pcd)
                        self.saved_count += 1
                        # Update exactly to the expected tick rather than current_time to avoid drifting
                        self.last_save_time += self.save_interval
                        if current_time - self.last_save_time > self.save_interval:
                             # If we lagged significantly, reset the anchor
                             self.last_save_time = current_time

            def update():
                if not self.is_running: return
                
                # Update UI texts
                self.points_label.text = f"Points: {pts_count}"
                if self.is_recording:
                    self.frames_label.text = f"Frames Saved: {self.saved_count}"
                    self.time_label.text = f"Elapsed: {elapsed_time:.1f}s"
                
                # Update 3D Scene
                scene = self.scene_widget.scene
                if scene.has_geometry("pcd"): 
                    scene.remove_geometry("pcd")
                
                if pts_count > 0:
                    scene.add_geometry("pcd", pcd, mat)
                
                nonlocal first_run
                if first_run and pts_count > 0:
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
    app = PointCloudCollectorUI()
    app.run()
