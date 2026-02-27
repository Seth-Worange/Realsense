import numpy as np
from scipy.signal import convolve2d
from .radar_config import RadarConfig

def simulate_adc(pc, vel, rcs, config: RadarConfig, batch_size=50):
    """
    点云到雷达 ADC 原始数据的正向仿真器
    采用分批 (Batched) 矩阵运算，极大提高执行效率。
    """
    P = pc.shape[0]
    adc_cube = np.zeros((config.NumChirps, config.NumRx, config.NumSamples), dtype=np.complex64)
    
    t_fast = np.arange(config.NumSamples) / config.Fs
    t_slow = np.arange(config.NumChirps) * config.PRT
    
    tx_pos = config.TxPos[0] # (3,)
    rx_pos = config.RxPos    # (NumRx, 3)
    
    # 将海量的点云切成 Batch 计算，避免内存爆炸 (OOM)
    for i in range(0, P, batch_size):
        idx_end = min(i + batch_size, P)
        pb_pc = pc[i:idx_end]    # (PB, 3)
        pb_vel = vel[i:idx_end]  # (PB, 3)
        pb_rcs = rcs[i:idx_end]  # (PB,)
        
        # 计算慢时间对应的靶标绝对位置
        # pos_n 维度: (PB, Nc, 3)
        pos_n = pb_pc[:, None, :] + pb_vel[:, None, :] * t_slow[None, :, None]
        
        # 每个点到 Tx 和 Rx 的欧式距离
        R_tx = np.linalg.norm(pos_n - tx_pos, axis=-1) # (PB, Nc)
        
        # 扩展出 Receive Antenna 维度
        pos_n_rx = pos_n[:, :, None, :] # (PB, Nc, 1, 3)
        R_rx = np.linalg.norm(pos_n_rx - rx_pos[None, None, :, :], axis=-1) # (PB, Nc, N_rx)
        
        # 计算双程飞行时间 Tau
        tau = (R_tx[:, :, None] + R_rx) / config.c # (PB, Nc, N_rx)
        
        # IF 信号的完整相位模型: 2 * pi * [f_c * tau + K * tau * t_fast]
        # 注意此模型天然包含微多普勒与空间域相位差异
        phase = 2 * np.pi * (config.fc * tau[:, :, :, None] + 
                             config.K * tau[:, :, :, None] * t_fast[None, None, None, :])
        
        # 根据雷达方程计算接收振幅 A ~ RCS / R^4，由于幅值是电压所以开方 -> sqrt(RCS)/R^2
        R_tot = R_tx[:, :, None] + R_rx
        R_tot = np.clip(R_tot, 0.1, None) # 避免由于距离为 0 引起的除 0 错误
        amp = np.sqrt(pb_rcs)[:, None, None] / (R_tot ** 2)
        
        # 生成复基带 IF 信号：在雷达混频器 (LO-RF) 逻辑下，相位的导数取正代表正中频
        signal = amp[:, :, :, None] * np.exp(1j * phase)
        
        # 将所有点的电磁波线性叠加至 ADC 立方体中
        adc_cube += np.sum(signal, axis=0)
        
    # 添加高斯白噪声模拟接收机底噪
    noise_power = 1e-12
    noise = (np.random.normal(scale=np.sqrt(noise_power/2), size=adc_cube.shape) + 
             1j * np.random.normal(scale=np.sqrt(noise_power/2), size=adc_cube.shape))
    adc_cube += noise
    
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
    
    # 3. 角度域 FFT (天线阵列空间维)
    win_angle = np.hanning(config.NumRx)
    doppler_fft_w = doppler_fft * win_angle[None, :, None]
    
    N_angle = 64 # Zero Padding 提升角度分别率
    angle_fft = np.fft.fftshift(np.fft.fft(doppler_fft_w, n=N_angle, axis=1), axes=1)
    
    # 计算方位角坐标轴 (支持从 -90 到 90)
    angle_freqs = np.fft.fftshift(np.fft.fftfreq(N_angle, d=1.0))
    sin_theta = 2 * angle_freqs # 因为 d = lambda / 2
    valid = np.abs(sin_theta) <= 1.0
    theta_axis = np.full(N_angle, np.nan)
    theta_axis[valid] = np.arcsin(sin_theta[valid]) * 180 / np.pi
    
    # 聚合获得 距离-多普勒 雷达图 (Range-Doppler Map)
    rdm = np.mean(np.abs(doppler_fft), axis=1)
    
    return rdm, angle_fft, range_axis, doppler_axis, theta_axis

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

def extract_point_cloud(angle_fft, range_axis, doppler_axis, theta_axis, rdm):
    """
    通过二维 CFAR 和 角度 FFT 的极值寻优抽取 3D 雷达点云
    """
    cfar_mask = ca_cfar_2d(rdm)
    doppler_idx, range_idx = np.where(cfar_mask)
    
    pc_radar = []
    
    for d_idx, r_idx in zip(doppler_idx, range_idx):
        angle_profile = np.abs(angle_fft[d_idx, :, r_idx])
        a_idx = np.argmax(angle_profile)
        
        theta = theta_axis[a_idx]
        if np.isnan(theta):
            continue
            
        v = doppler_axis[d_idx]
        r = range_axis[r_idx]
        intensity = rdm[d_idx, r_idx]
        
        # 从极坐标映射回笛卡尔空间 (约定 Y 轴为深度纵深前向)
        x = r * np.sin(theta * np.pi / 180)
        y = r * np.cos(theta * np.pi / 180)
        z = 0 # 1D 阵列不可测算高度
        
        pc_radar.append([x, y, z, v, intensity])
        
    return np.array(pc_radar) if len(pc_radar) > 0 else np.zeros((0, 5))
