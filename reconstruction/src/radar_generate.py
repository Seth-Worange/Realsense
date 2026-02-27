'''
Author: Orange
Date: 2026-02-26 16:53
LastEditors: Orange
LastEditTime: 2026-02-27 16:02
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

    # 读取您的连拍数据集路径
    player = PointCloudSequencePlayer("../datas/rs_pointcloud/seq_20260227_150030/", fps=10.0)
    # 例如我们想抽取第 50 帧时那个正在挥手的瞬间，直接送入雷达核心进行干涉探测
    pc, vel, rcs = player.get_all_as_radar_targets(frame_idx=50)
    # 剩下的事情完全不变：送入您的 ADC 仿真算法并解析热力图！
    adc_cube = simulate_adc(pc, vel, rcs, config)
    
    print("运行雷达基带数字信号处理 (DSP)...")
    start_time = time.time()
    rdm, angle_fft, range_axis, doppler_axis, theta_axis = process_radar_data(adc_cube, config)
    print(f" -> 频域 FFT 解析完成, 耗时 {time.time() - start_time:.3f} 秒.")
    
    print("在 Range-Doppler 域截取 CFAR 并解析角度产生雷达点云...")
    radar_pc = extract_point_cloud(angle_fft, range_axis, doppler_axis, theta_axis, rdm)
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
    plt.subplot(1, 2, 2)
    # 底座 (真实情况)
    plt.scatter(pc[:, 0], pc[:, 1], c='gray', s=5, alpha=0.3, label='Ground Truth PC')
    # 雷达 (仿真观测)
    if len(radar_pc) > 0:
        scatter = plt.scatter(radar_pc[:, 0], radar_pc[:, 1], c=radar_pc[:, 3], cmap='coolwarm', s=50, edgecolors='black')
        plt.colorbar(scatter, label='Velocity (m/s)')
        
    plt.title('Radar Point Cloud Map')
    plt.xlabel('Azimuth/X (m)')
    plt.ylabel('Depth/Y (m)')
    plt.legend()
    plt.xlim(-2, 2)
    plt.ylim(0, 8)
    plt.grid(True)
    
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), 'radar_simulation_result.png')
    plt.savefig(output_path, dpi=300)
    print(f" -> 成功！结果图已存至 {output_path}")
    plt.show()

if __name__ == "__main__":
    main()
