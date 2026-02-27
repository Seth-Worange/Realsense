'''
Author: Orange
Date: 2026-02-26 16:53
LastEditors: Orange
LastEditTime: 2026-02-27 18:41
FilePath: radar_generate.py
Description: Vectorized simulator for mmWave FMCW Radar Human Echoes
'''

import numpy as np
import matplotlib.pyplot as plt
import time
import os

from utils import PointCloudSequencePlayer
from utils.radar_config import RadarConfig
from utils.radar_dsp import simulate_adc, process_radar_data, ca_cfar_2d, extract_point_cloud

def main():
    print("初始化雷达配置参数...")
    config = RadarConfig()

    # 读取您的连拍数据集绝对路径，避免因为执行目录不同导致找不到文件
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    seq_path = os.path.join(base_dir, "datas", "rs_pointcloud", "seq_20260227_150659")
    player = PointCloudSequencePlayer(seq_path, fps=6.0)
    
    # 例如我们想抽取第 50 帧时那个正在挥手的瞬间，直接送入雷达核心进行干涉探测
    target_frame = 50
    pc, vel, rcs = player.get_all_as_radar_targets(frame_idx=target_frame)
    
    if pc is None:
        print(f"载入失败: 请检查文件路径 {seq_path} 下是否有起码 {target_frame} 帧以上的 .ply 文件")
        return
    
    # ----------------------------------------------------
    # 空间系矫正变换 (Coordinate Transformation)
    # ----------------------------------------------------
    # 1. 坐标轴重新映射 (Axis Swapping)
    # RealSense 坐标系: X向右, Y向下, Z向前(深度)
    # 雷达约定坐标系: X向右(方位角), Y向前(深度), Z向上(俯仰角)
    pc_radar_frame = np.zeros_like(pc)
    pc_radar_frame[:, 0] = pc[:, 0]    # Radar X = RS X
    pc_radar_frame[:, 1] = pc[:, 2]    # Radar Y = RS Z
    pc_radar_frame[:, 2] = -pc[:, 1]   # Radar Z = -RS Y (RS Y向下，Radar Z向上)
    
    vel_radar_frame = np.zeros_like(vel)
    vel_radar_frame[:, 0] = vel[:, 0]
    vel_radar_frame[:, 1] = vel[:, 2]
    vel_radar_frame[:, 2] = -vel[:, 1]
    
    # 2. 旋转补偿: 假设摄像头和雷达安装在同一物理位置，但雷达具有 15 度的下倾角
    # 雷达低头视物（俯视），意味着为了把相机坐标系对准雷达，我们需要把世界往上抬起
    # 相当于绕 X 轴旋转 +15 度
    radar_pitch_deg = -15 # 雷达下倾对应正向旋转补偿
    theta_pitch = np.deg2rad(radar_pitch_deg)
    
    # 绕 X 轴旋转矩阵
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(theta_pitch), -np.sin(theta_pitch)],
        [0, np.sin(theta_pitch),  np.cos(theta_pitch)]
    ])
    
    # 将现实世界(已轴对齐)的点云投影到以雷达倾角为基准的坐标系下
    pc_radar_frame = pc_radar_frame @ R_x.T
    vel_radar_frame = vel_radar_frame @ R_x.T

    rcs = rcs * np.random.uniform(0.5, 1.5, size=rcs.shape)
    
    # 剩下的事情完全不变：送入您的 ADC 仿真算法并解析热力图！
    adc_cube = simulate_adc(pc_radar_frame, vel_radar_frame, rcs, config)
    
    print("运行雷达基带数字信号处理 (DSP)...")
    start_time = time.time()
    rdm, angle_fft_2d, range_axis, doppler_axis, theta_axis, phi_axis = process_radar_data(adc_cube, config)
    print(f" -> 频域 FFT 解析完成, 耗时 {time.time() - start_time:.3f} 秒.")
    
    print("在 Range-Doppler 域截取 CFAR 并解析角度产生雷达点云...")
    radar_pc = extract_point_cloud(angle_fft_2d, range_axis, doppler_axis, theta_axis, phi_axis, rdm)
    print(f" -> 检出 {len(radar_pc)} 个稀疏雷达反射点.")
    
    print("可视化渲染输出...")
    plt.figure(figsize=(12, 5))
    
    # 绘制距离-速度雷达能量谱图
    plt.subplot(1, 2, 1)
    rdm_db = 10 * np.log10(rdm + 1e-10)
    plt.imshow(rdm_db, aspect='auto', 
               extent=[range_axis[0], range_axis[-1], doppler_axis[0], doppler_axis[-1]], 
               cmap='jet', origin='lower')
    plt.title('Range-Doppler Map (RDM)')
    plt.xlabel('Range (m)')
    plt.ylabel('Velocity (m/s)')
    plt.colorbar(label='Power (dB)')
    
    # 绘制生成的仿真雷达点云，基于真实的点云底座对比
    ax = plt.subplot(1, 2, 2, projection='3d')
    
    # 底座 (真实情况，已经过雷达坐标系矫正)
    ax.scatter(pc_radar_frame[:, 0], pc_radar_frame[:, 1], pc_radar_frame[:, 2], 
               c='gray', s=2, alpha=0.2, label='Ground Truth PC')
               
    # 雷达 (仿真观测)
    if len(radar_pc) > 0:
        scatter = ax.scatter(radar_pc[:, 0], radar_pc[:, 1], radar_pc[:, 2], 
                             c=radar_pc[:, 3], cmap='coolwarm', s=30, edgecolors='black', alpha=0.9)
        plt.colorbar(scatter, label='Velocity (m/s)', ax=ax, shrink=0.5, pad=0.1)
        
    ax.set_title('3D Radar Point Cloud')
    ax.set_xlabel('Azimuth/X (m)')
    ax.set_ylabel('Depth/Y (m)')
    ax.set_zlabel('Height/Z (m)')
    ax.legend()
    ax.set_xlim(-2, 2)
    ax.set_ylim(0, 8)
    ax.set_zlim(-1, 2)
    
    # 设置合理的初始观察视角
    ax.view_init(elev=20, azim=-60)
    
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), 'radar_simulation_result.png')
    plt.savefig(output_path, dpi=300)
    print(f" -> 成功！结果图已存至 {output_path}")
    plt.show()

if __name__ == "__main__":
    main()
