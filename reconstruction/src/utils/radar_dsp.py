'''
Author: Orange
Date: 2026-02-27 10:51
LastEditors: Orange
LastEditTime: 2026-02-27 16:59
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
    adc_cube = np.zeros((config.NumChirps, N_virt, config.NumSamples), dtype=np.complex64)
    
    t_fast = np.arange(config.NumSamples) / config.Fs
    t_slow = np.arange(config.NumChirps) * config.PRT
    
    # 构建虚拟阵列天线的位置
    # virtual_pos 维度: (NumTx * NumRx, 3) 
    # 即对应图示的 0, ω, 2ω, 3ω, 4ω ... (ω=d1=lambda/2)
    virtual_pos = np.zeros((N_virt, 3))
    idx = 0
    for t in range(config.NumTx):
        for r in range(config.NumRx):
            # TDM-MIMO 中虚拟阵列位置 = Tx位置 + Rx位置
            virtual_pos[idx] = config.TxPos[t] + config.RxPos[r]
            idx += 1
            
    # 将海量的点云切成 Batch 计算，避免内存爆炸 (OOM)
    for i in range(0, P, batch_size):
        idx_end = min(i + batch_size, P)
        pb_pc = pc[i:idx_end]    # (PB, 3)
        pb_vel = vel[i:idx_end]  # (PB, 3)
        pb_rcs = rcs[i:idx_end]  # (PB,)
        
        # 计算慢时间对应的靶标绝对位置
        # pos_n 维度: (PB, Nc, 3)
        pos_n = pb_pc[:, None, :] + pb_vel[:, None, :] * t_slow[None, :, None]
        
        # 在远场近似下，双程距离等于 2 * 中心距离 - (发射天线投影 + 接收天线投影)
        # 也就是单程距离等效为从 virtual_pos 进行单向收发
        # R_center: (PB, Nc)
        
        # 扩展出 Virtual Array 维度
        pos_n_virt = pos_n[:, :, None, :] # (PB, Nc, 1, 3)
        # R_virt: (PB, Nc, N_virt)
        # 这里计算每个虚拟天线的等效单程 R (假设双程被平分为两个单程以匹配远场近场混杂)
        # 精确写法： R_total = || Pos - Tx || + || Pos - Rx || 
        # 为了高效，我们将 virtual_array 的相控效果退化为中心等效：
        
        # 严格计算距离
        R_tx = np.zeros((idx_end - i, config.NumChirps, N_virt))
        R_rx = np.zeros((idx_end - i, config.NumChirps, N_virt))
        
        v_idx = 0
        for t in range(config.NumTx):
            for r in range(config.NumRx):
                # np.linalg.norm: (PB, Nc, 3) -> (PB, Nc)
                R_tx[:, :, v_idx] = np.linalg.norm(pos_n - config.TxPos[t], axis=-1)
                R_rx[:, :, v_idx] = np.linalg.norm(pos_n - config.RxPos[r], axis=-1)
                v_idx += 1
                
        # 计算双程飞行时间 Tau
        tau = (R_tx + R_rx) / config.c # (PB, Nc, N_virt)
        
        # IF 信号的完整相位模型: 2 * pi * [f_c * tau + K * tau * t_fast]
        phase = 2 * np.pi * (config.fc * tau[:, :, :, None] + 
                             config.K * tau[:, :, :, None] * t_fast[None, None, None, :])
        
        # 根据雷达方程计算接收振幅 A ~ RCS / R^4，由于幅值是电压所以开方 -> sqrt(RCS)/R^2
        R_tot = R_tx + R_rx
        R_tot = np.clip(R_tot, 0.1, None) # 避免由于距离为 0 引起的除 0 错误
        amp = np.sqrt(pb_rcs)[:, None, None] / (R_tot ** 2)
        
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
    
    # 建立 8(Azimuth) x 3(Elevation) 的虚拟面阵网格
    grid = np.zeros((config.NumChirps, 3, 8, config.NumSamples // 2), dtype=np.complex64)
    
    idx = 0
    for t in range(config.NumTx):
        for r in range(config.NumRx):
            vx = int(round((config.TxPos[t, 0] + config.RxPos[r, 0]) / d))
            vz = int(round((config.TxPos[t, 2] + config.RxPos[r, 2]) / d))
            grid[:, vz, vx, :] = doppler_fft[:, idx, :]
            idx += 1
            
    # 空间域加窗
    win_az = np.hanning(8)
    win_el = np.hanning(3)
    grid_w = grid * win_el[None, :, None, None] * win_az[None, None, :, None]
    
    N_az = 64
    N_el = 64
    angle_fft_2d = np.fft.fftshift(np.fft.fft2(grid_w, s=(N_el, N_az), axes=(1, 2)), axes=(1, 2))
    
    # 计算方位角与俯仰角物理坐标轴
    az_freqs = np.fft.fftshift(np.fft.fftfreq(N_az, d=1.0))
    el_freqs = np.fft.fftshift(np.fft.fftfreq(N_el, d=1.0))
    
    # 根据傅里叶变换的相移方向，我们需要加上负系数对应到实际物理空间
    # exp(-j * 2 * pi * f * m)
    sin_theta = -2 * az_freqs
    valid_theta = np.abs(sin_theta) <= 1.0
    theta_axis = np.full(N_az, np.nan)
    theta_axis[valid_theta] = np.arcsin(sin_theta[valid_theta]) * 180 / np.pi
    
    sin_phi = -2 * el_freqs
    valid_phi = np.abs(sin_phi) <= 1.0
    phi_axis = np.full(N_el, np.nan)
    phi_axis[valid_phi] = np.arcsin(sin_phi[valid_phi]) * 180 / np.pi
    
    # 聚合获得 距离-多普勒 雷达图 (Range-Doppler Map)
    rdm = np.mean(np.abs(doppler_fft), axis=1)
    
    return rdm, angle_fft_2d, range_axis, doppler_axis, theta_axis, phi_axis

def ca_cfar_2d(rdm, guard_cells=(2, 2), train_cells=(4, 4), pfa=1e-5):
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
    min_power = np.max(rdm_sq) * 1e-4 
    mask = (rdm_sq > threshold) & (rdm_sq > min_power)
    
    return mask

def extract_point_cloud(angle_fft_2d, range_axis, doppler_axis, theta_axis, phi_axis, rdm):
    """
    通过二维 CFAR 和 2D 角度 FFT 的极值寻优抽取 3D 雷达点云 (包含高度)
    """
    cfar_mask = ca_cfar_2d(rdm)
    doppler_idx, range_idx = np.where(cfar_mask)
    
    pc_radar = []
    
    for d_idx, r_idx in zip(doppler_idx, range_idx):
        angle_profile = np.abs(angle_fft_2d[d_idx, :, :, r_idx]) # (N_el, N_az)
        el_idx, az_idx = np.unravel_index(np.argmax(angle_profile), angle_profile.shape)
        
        theta = theta_axis[az_idx]
        phi = phi_axis[el_idx]
        
        if np.isnan(theta) or np.isnan(phi):
            continue
            
        v = doppler_axis[d_idx]
        r = range_axis[r_idx]
        intensity = rdm[d_idx, r_idx]
        
        # 从极坐标映射回笛卡尔空间 (方位角 Azimuth = theta，俯仰角 Elevation = phi)
        # 约定 Y 为前方深度纵深
        y = r * np.cos(phi * np.pi / 180) * np.cos(theta * np.pi / 180)
        x = r * np.cos(phi * np.pi / 180) * np.sin(theta * np.pi / 180)
        z = r * np.sin(phi * np.pi / 180) # Z 为高度
        
        pc_radar.append([x, y, z, v, intensity])
        
    return np.array(pc_radar) if len(pc_radar) > 0 else np.zeros((0, 5))
