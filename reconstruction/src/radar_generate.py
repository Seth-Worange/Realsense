'''
Author: Orange
Date: 2026-02-26 16:53
LastEditors: Orange
LastEditTime: 2026-02-27 10:52
FilePath: radar_generate.py
Description: Vectorized simulator for mmWave FMCW Radar Human Echoes
'''

import numpy as np
import matplotlib.pyplot as plt
import time
import os

from utils.radar_config import RadarConfig
from utils.radar_dsp import simulate_adc, process_radar_data, ca_cfar_2d, extract_point_cloud

def main():
    print("[1] 初始化雷达配置参数...")
    config = RadarConfig()
    
    print("[2] 模拟生成人体点云及运动速度...")
    num_points = 1000
    y_center = 4.0 # 距离雷达 4 米处
    v_body = -1.2  # 走向雷达主体速度
    
    pc = np.zeros((num_points, 3))
    vel = np.zeros((num_points, 3))
    rcs = np.ones((num_points,)) * 0.1 # 简化的微小 RCS
    
    # 形似人体的散布
    pc[:, 0] = np.random.normal(0, 0.2, num_points)             # 肩宽方向
    pc[:, 1] = y_center + np.random.normal(0, 0.1, num_points)  # 厚度方向
    pc[:, 2] = np.random.normal(1.0, 0.5, num_points)           # 身高方向
    
    vel[:, 1] = v_body # 总体径向向着原点运动
    
    # 添加一个微多普勒干扰源: 例如手臂前后摆动导致的差动速度
    hands_idx = np.random.choice(num_points, int(num_points*0.1), replace=False)
    vel[hands_idx, 1] += np.sin(np.pi) * 2.0 + 1.5 # 模拟挥手造成的一个正向速度回波
    
    print(f"[3] 模拟毫米波雷达电磁波传输，生成 ADC Raw Data ({num_points} 个人体反射点)...")
    start_time = time.time()
    adc_cube = simulate_adc(pc, vel, rcs, config)
    print(f" -> ADC 频域数据生成完毕, 耗时 {time.time() - start_time:.3f} 秒.")
    
    print("[4] 运行雷达基带数字信号处理 (DSP)...")
    start_time = time.time()
    rdm, angle_fft, range_axis, doppler_axis, theta_axis = process_radar_data(adc_cube, config)
    print(f" -> 频域 FFT 解析完成, 耗时 {time.time() - start_time:.3f} 秒.")
    
    print("[5] 在 Range-Doppler 域截取 CFAR 并解析角度产生雷达点云...")
    radar_pc = extract_point_cloud(angle_fft, range_axis, doppler_axis, theta_axis, rdm)
    print(f" -> 检出 {len(radar_pc)} 个稀疏雷达反射点.")
    
    print("[6] 可视化渲染输出...")
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
