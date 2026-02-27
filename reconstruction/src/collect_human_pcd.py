import os
import cv2
import time
import argparse
import open3d as o3d
import numpy as np

from utils.rs_config import RSConfig
from utils.realsense import RealSenseProcessor, PCDMode

def main():
    parser = argparse.ArgumentParser(description="Human Point Cloud Sequential Collector")
    parser.add_argument("--fps", type=int, default=10, help="Target saving FPS (default: 10)")
    parser.add_argument("--duration", type=float, default=0, help="Auto-stop after X seconds (0 means manual stop via UI)")
    args = parser.parse_args()

    print("====================================")
    print("    Human Point Cloud Collector     ")
    print("====================================")
    
    # 1. 实例化配置参数
    config = RSConfig()
    config.voxel_size = 0.02 # 降采样体素大小
    config.human_depth_tolerance = 0.5 # 景深容忍度
    
    # 根据用户期望设定的保存帧率计算保存周期
    target_fps = args.fps
    save_interval = 1.0 / target_fps
    
    print(f"[1] 初始化 RealSense ...")
    processor = RealSenseProcessor(rs_config=config)
    processor.start()
    
    # 2. 建立保存目录
    base_save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datas", "rs_pointcloud")
    if not os.path.exists(base_save_dir):
        os.makedirs(base_save_dir)
        
    print(f"[2] 保存基路径: {base_save_dir}")
    print(f"[3] 采集设置: 目标帧率 = {target_fps} FPS, 时长 = {'手动控制' if args.duration <= 0 else str(args.duration) + ' 秒'}")
    print("[4] 快捷键说明:")
    print("    - 持续运行中...")
    print("    - 空格键/S键 : 开始/手动停止 【序列采集】")
    print("    - Q键        : 退出程序")
    
    cv2.namedWindow('RealSense Stream', cv2.WINDOW_AUTOSIZE)
    
    is_recording = False
    recording_start_time = 0.0
    last_save_time = 0.0
    saved_count = 0
    seq_dir = ""
    
    try:
        while True:
            # 持续获取只包含人体的点云数据
            data = processor.get_data_bundle(mode=PCDMode.HUMAN_ONLY)
            
            if data is None:
                time.sleep(0.01)
                continue
                
            color_img = data['color']
            pcd = data['pcd']
            pts_count = len(pcd.points)
            
            current_time = time.time()
            
            # --- 自动停止逻辑检测 ---
            if is_recording and args.duration > 0:
                if (current_time - recording_start_time) >= args.duration:
                    print(f" -> 达到预设时长 {args.duration}s，自动停止录制。共采集 {saved_count} 帧。")
                    is_recording = False
                    
            # --- 录制保存逻辑 ---
            if is_recording:
                # 按照指定的帧率进行时间控制
                if (current_time - last_save_time) >= save_interval:
                    if pts_count > 0:
                        filename = os.path.join(seq_dir, f"frame_{saved_count:04d}.ply")
                        o3d.io.write_point_cloud(filename, pcd)
                        saved_count += 1
                        last_save_time = current_time
                        
                        # 在保存的瞬间，画面边框显示红色代表正在录制
                        cv2.rectangle(color_img, (0, 0), (color_img.shape[1]-1, color_img.shape[0]-1), (0, 0, 255), 5)
            
            # --- 界面绘制显示 ---
            if is_recording:
                duration_str = f"Rec: {current_time - recording_start_time:.1f}s"
                cv2.putText(color_img, "• RECORDING", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(color_img, f"Frames: {saved_count} | {duration_str}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            else:
                cv2.putText(color_img, "STANDBY", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(color_img, "Press S or Space to Start Rec", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
            status_text = f"Points: {pts_count}"
            cv2.putText(color_img, status_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow('RealSense Stream', color_img)
            
            # --- 键盘事件响应 ---
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("用户主动断开.")
                break
            elif key == ord('s') or key == ord(' '):
                if not is_recording:
                    # 开始新的录制序列
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    seq_dir = os.path.join(base_save_dir, f"seq_{timestamp}")
                    os.makedirs(seq_dir, exist_ok=True)
                    
                    is_recording = True
                    recording_start_time = time.time()
                    last_save_time = recording_start_time
                    saved_count = 0
                    
                    print(f"\n[开始录制] 数据将保存在: {seq_dir}")
                else:
                    # 手动停止录制
                    is_recording = False
                    print(f" -> 手动停止录制。共采集 {saved_count} 帧。")
                    
    finally:
        processor.stop()
        cv2.destroyAllWindows()
        print("硬件句柄释放完毕。")

if __name__ == "__main__":
    main()
