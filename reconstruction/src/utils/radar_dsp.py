'''
Author: Orange
Date: 2026-02-27 10:51
LastEditors: Orange
LastEditTime: 2026-03-03 14:14
FilePath: radar_dsp.py
Description: 
    Radar DSP: Process radar data and extract point clouds
'''


import numpy as np
from scipy.signal import convolve2d
from .radar_config import RadarConfig

def simulate_adc(pc, vel, rcs, config: RadarConfig, batch_size=50):
    """
    点云到雷达 ADC 原始数据的正向仿真器
    采用分批 (Batched) 矩阵运算，极大提高执行效率。
    """
    P = pc.shape[0]
    # 在 TDM-MIMO 中，如果发射端不同时发射（时分复用），实际上是在不同的 Chirp 或时隙里发射
    # 为了简化且等效，这里我们直接生成一个大小为 NumChirps x (NumTx * NumRx) 的虚拟矩阵
    # 物理意义上相当于雷达硬件完成了 TDM 合并，直接提取出虚拟阵列
    N_virt = config.NumTx * config.NumRx
    adc_cube = np.zeros((config.NumChirps, N_virt, config.NumSamples), dtype=np.complex128)
    
    t_fast = np.arange(config.NumSamples) / config.Fs
    t_slow = np.arange(config.NumChirps) * config.PRT
    
    # 构建虚拟阵列天线的位置
    # 在进入 batch 循环前，预先构建 Tx 和 Rx 的扩展坐标矩阵
    tx_arr = np.zeros((N_virt, 3))
    rx_arr = np.zeros((N_virt, 3))
    v_idx = 0
    for t in range(config.NumTx):
        for r in range(config.NumRx):
            tx_arr[v_idx] = config.TxPos[t]
            rx_arr[v_idx] = config.RxPos[r]
            v_idx += 1
            
    # 进入 Batch 循环
    for i in range(0, P, batch_size):
        idx_end = min(i + batch_size, P)
        pb_pc = pc[i:idx_end]    
        pb_vel = vel[i:idx_end]  
        pb_rcs = rcs[i:idx_end]  
        
        pos_n = pb_pc[:, None, :] + pb_vel[:, None, :] * t_slow[None, :, None]
        
        # --- 优化点：利用广播一次性计算所有虚拟天线的距离 ---
        # pos_n 扩展为 (PB, Nc, 1, 3)，与 tx_arr (N_virt, 3) 广播计算
        pos_n_exp = pos_n[:, :, None, :]
        R_tx = np.linalg.norm(pos_n_exp - tx_arr, axis=-1)
        R_rx = np.linalg.norm(pos_n_exp - rx_arr, axis=-1)
                
        # 计算双程飞行时间 Tau
        tau = (R_tx + R_rx) / config.c # (PB, Nc, N_virt)
        
        # IF 信号的完整相位模型: 2 * pi * [f_c * tau + K * tau * t_fast]
        phase = 2 * np.pi * (config.fc * tau[:, :, :, None] + 
                             config.K * tau[:, :, :, None] * t_fast[None, None, None, :])
        
        # 根据雷达方程计算接收振幅 A ~ RCS / R^4，由于幅值是电压所以开方 -> sqrt(RCS)/R^2
        R_tx_clip = np.clip(R_tx, 0.1, None)
        R_rx_clip = np.clip(R_rx, 0.1, None)
        amp = np.sqrt(pb_rcs)[:, None, None] / (R_tx_clip ** 2 * R_rx_clip ** 2)
        
        # 生成复基带 IF 信号：
        signal = amp[:, :, :, None] * np.exp(1j * phase)
        
        # 将所有点的电磁波线性叠加至 ADC 立方体中
        adc_cube += np.sum(signal, axis=0)
        
    # 添加高斯白噪声模拟接收机真实热噪声与杂波
    # 我们基于纯理论回波的平均信号功率，设置一个动态的 SNR，比如 20dB
    sig_power = np.mean(np.abs(adc_cube) ** 2)
    target_snr_db = 20.0
    noise_power = sig_power / (10 ** (target_snr_db / 10.0))
    if noise_power == 0:
        noise_power = 1e-12 # 防除0
        
    noise = (np.random.normal(scale=np.sqrt(noise_power/2), size=adc_cube.shape) + 
             1j * np.random.normal(scale=np.sqrt(noise_power/2), size=adc_cube.shape))
    adc_cube += noise
    
    # 加入仪器不可避免的本振（VCO）相位噪声
    phase_noise = np.random.normal(0, np.deg2rad(1.0), size=adc_cube.shape) # 1度
    adc_cube *= np.exp(1j * phase_noise)
    
    return adc_cube

def process_radar_data(adc_cube, config: RadarConfig):
    """
    雷达数字信号处理 (DSP) 链路：Range FFT -> Doppler FFT -> Angle FFT
    """
    # 1. 距离域 FFT (快时间)
    win_range = np.hanning(config.NumSamples)
    adc_cube_w = adc_cube * win_range[None, None, :]
    range_fft = np.fft.fft(adc_cube_w, axis=2)
    range_fft = range_fft[:, :, :config.NumSamples // 2] # 截取有效频段
    
    # 计算距离物理坐标轴
    fast_freqs = np.fft.fftfreq(config.NumSamples, d=1/config.Fs)[:config.NumSamples // 2]
    range_axis = fast_freqs * config.c / (2 * config.K)
    
    # 2. 多普勒域 FFT (慢时间)
    win_doppler = np.hanning(config.NumChirps)
    range_fft_w = range_fft * win_doppler[:, None, None]
    doppler_fft = np.fft.fftshift(np.fft.fft(range_fft_w, axis=0), axes=0)
    
    # 计算速度物理坐标轴
    slow_freqs = np.fft.fftshift(np.fft.fftfreq(config.NumChirps, d=config.PRT))
    doppler_axis = slow_freqs * config.wavelength / 2
    
    # 3. 角度域 2D FFT (包含 Tx 和 Rx 合成的复杂面阵)
    d = config.wavelength / 2
    
    # 动态获取网格坐标范围以兼容带有整体偏移量的高程配置
    vx_list = []
    vz_list = []
    for t in range(config.NumTx):
        for r in range(config.NumRx):
            vx_list.append(int(round((config.TxPos[t, 0] + config.RxPos[r, 0]) / d)))
            vz_list.append(int(round((config.TxPos[t, 2] + config.RxPos[r, 2]) / d)))
            
    min_vx, max_vx = min(vx_list), max(vx_list)
    min_vz, max_vz = min(vz_list), max(vz_list)
    num_vx = max_vx - min_vx + 1
    num_vz = max_vz - min_vz + 1
    
    # 建立动态伸缩的虚拟面阵网格
    grid = np.zeros((config.NumChirps, num_vz, num_vx, config.NumSamples // 2), dtype=np.complex128)
    
    idx = 0
    for t in range(config.NumTx):
        for r in range(config.NumRx):
            vx = int(round((config.TxPos[t, 0] + config.RxPos[r, 0]) / d)) - min_vx
            vz = int(round((config.TxPos[t, 2] + config.RxPos[r, 2]) / d)) - min_vz
            grid[:, vz, vx, :] = doppler_fft[:, idx, :]
            idx += 1
            
    # --- 修改部分：解耦的 1D 角度 FFT ---
    N_az = 64
    N_el = 64
    
    # 1. 提取水平方向阵列 (水平截面) 进行方位角估计
    az_data = grid[:, 0, :, :] # 取底层水平阵元列 (NumChirps, num_vx, NumSamples//2)
    win_az = np.hanning(num_vx)
    az_data_w = az_data * win_az[None, :, None]
    az_fft = np.fft.fftshift(np.fft.fft(az_data_w, n=N_az, axis=1), axes=1)
    
    # 2. 提取垂直方向阵列 进行俯仰角估计 (自动搜寻有最多垂直跨度的阵元列)
    col_has_signal = np.sum(np.abs(grid), axis=(0, 3)) > 0
    best_vx = np.argmax(np.sum(col_has_signal, axis=0))
    
    el_data = grid[:, :, best_vx, :] # (NumChirps, num_vz, NumSamples//2)
    win_el = np.hanning(num_vz)
    el_data_w = el_data * win_el[None, :, None]
    el_fft = np.fft.fftshift(np.fft.fft(el_data_w, n=N_el, axis=1), axes=1)
    
    # 计算物理坐标轴
    az_freqs = np.fft.fftshift(np.fft.fftfreq(N_az, d=1.0))
    el_freqs = np.fft.fftshift(np.fft.fftfreq(N_el, d=1.0))
    
    sin_theta = -2 * az_freqs
    theta_axis = np.where(np.abs(sin_theta) <= 1.0, np.arcsin(sin_theta) * 180 / np.pi, np.nan)
    
    sin_phi = -2 * el_freqs
    phi_axis = np.where(np.abs(sin_phi) <= 1.0, np.arcsin(sin_phi) * 180 / np.pi, np.nan)
    
    # 聚合获得 距离-多普勒 雷达图 (Range-Doppler Map)
    rdm = np.mean(np.abs(doppler_fft), axis=1)
    
    # 返回分离的 az_fft 和 el_fft
    return rdm, az_fft, el_fft, range_axis, doppler_axis, theta_axis, phi_axis

def ca_cfar_2d(rdm, guard_cells=(2, 2), train_cells=(4, 4), pfa=1e-3):
    """
    二维细胞平均恒虚警率检测 (2D CA-CFAR) 
    使用 2D 卷积极速提升计算效率
    """
    gr, gc = guard_cells
    tr, tc = train_cells
    
    # 构造卷积核
    kernel_size = (2*(gr+tr)+1, 2*(gc+tc)+1)
    kernel = np.ones(kernel_size)
    
    # 保护单元与待检测单元(CUT)中心置 0
    gr_start, gr_end = tr, tr + 2*gr + 1
    gc_start, gc_end = tc, tc + 2*gc + 1
    kernel[gr_start:gr_end, gc_start:gc_end] = 0
    
    N_train = np.sum(kernel)
    alpha = N_train * (pfa ** (-1.0 / N_train) - 1)
    
    # 使用卷积获取局部能量积分
    rdm_sq = rdm ** 2
    noise_sum = convolve2d(rdm_sq, kernel, mode='same', boundary='symm')
    noise_level = noise_sum / N_train
    threshold = alpha * noise_level
    
    # 加入基础能量下界，只检测超过本底噪声的峰值
    min_power = np.max(rdm_sq) * 1e-5 
    mask = (rdm_sq > threshold) & (rdm_sq > min_power)
    
    return mask

def extract_point_cloud(az_fft, el_fft, range_axis, doppler_axis, theta_axis, phi_axis, rdm):
    """
    通过二维 CFAR 和 1D 角度解耦 FFT 的极值寻优抽取 3D 雷达点云 (包含高度)
    """
    cfar_mask = ca_cfar_2d(rdm)
    doppler_idx, range_idx = np.where(cfar_mask)
    
    pc_radar = []
    
    for d_idx, r_idx in zip(doppler_idx, range_idx):
        # 独立寻优方位角和俯仰角
        az_profile = np.abs(az_fft[d_idx, :, r_idx]) 
        el_profile = np.abs(el_fft[d_idx, :, r_idx]) 
        
        az_idx = np.argmax(az_profile)
        el_idx = np.argmax(el_profile)
        
        # 峰值有效性检测: 当频谱近似平坦时，该维度无角度分辨力，默认 boresight (0°)
        PEAK_RATIO_THRESHOLD = 1.2
        
        az_mean = np.mean(az_profile)
        if az_mean > 0 and az_profile[az_idx] / az_mean > PEAK_RATIO_THRESHOLD:
            theta = theta_axis[az_idx]
        else:
            theta = 0.0  # 无有效方位角信息，默认 boresight
            
        el_mean = np.mean(el_profile)
        if el_mean > 0 and el_profile[el_idx] / el_mean > PEAK_RATIO_THRESHOLD:
            phi = phi_axis[el_idx]
        else:
            phi = 0.0  # 无有效俯仰角信息，默认 boresight
        
        if np.isnan(theta) or np.isnan(phi):
            continue
            
        v = doppler_axis[d_idx]
        r = range_axis[r_idx]
        intensity = rdm[d_idx, r_idx]
        
        # --- 显式球坐标 -> 笛卡尔映射 ---
        theta_rad = np.deg2rad(theta)
        phi_rad = np.deg2rad(phi)
        
        x = r * np.cos(phi_rad) * np.sin(theta_rad)   # 方位方向
        y = r * np.cos(phi_rad) * np.cos(theta_rad)   # 纵深方向
        z = r * np.sin(phi_rad)                        # 高度方向
        
        pc_radar.append([x, y, z, v, intensity])
        
    return np.array(pc_radar) if len(pc_radar) > 0 else np.zeros((0, 5))
