'''
Author: Orange
Date: 2026-02-27 10:51
LastEditors: Orange
LastEditTime: 2026-03-04 14:18
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
        
        # 幅值衰减模型：室内近距 (1-3m) 人体场景采用 1/R² 软化模型
        # 理论 1/R⁴ 在近距离动态范围过大，导致弱散射点被躯干掩盖
        R_tx_clip = np.clip(R_tx, 0.1, None)
        R_rx_clip = np.clip(R_rx, 0.1, None)
        amp = np.sqrt(pb_rcs)[:, None, None] / (R_tx_clip * R_rx_clip)
        
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
    雷达数字信号处理 (DSP) 链路：Range FFT -> Doppler FFT -> 构建虚拟阵列网格
    角度估计由 Capon 波束形成在 extract_point_cloud 中完成
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
    
    # 3. 构建虚拟天线面阵网格 (用于后续 Capon 波束形成)
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
    
    # 记录哪些网格位置有真实天线数据（用于 Capon 稀疏阵列处理）
    grid_mask = np.zeros((num_vz, num_vx), dtype=bool)
    
    idx = 0
    for t in range(config.NumTx):
        for r in range(config.NumRx):
            vx = int(round((config.TxPos[t, 0] + config.RxPos[r, 0]) / d)) - min_vx
            vz = int(round((config.TxPos[t, 2] + config.RxPos[r, 2]) / d)) - min_vz
            grid[:, vz, vx, :] = doppler_fft[:, idx, :]
            grid_mask[vz, vx] = True
            idx += 1
    
    # 提取方位和俯仰方向的有效阵元索引
    # 方位: 取底层水平行 (vz=0) 中有数据的列索引
    az_element_indices = np.where(grid_mask[0, :])[0]
    
    # 俯仰: 取纵向阵元最多的列, 提取该列中有数据的行索引
    col_counts = np.sum(grid_mask, axis=0)
    best_vx_col = np.argmax(col_counts)
    el_element_indices = np.where(grid_mask[:, best_vx_col])[0]
    
    # 聚合获得 距离-多普勒 雷达图 (Range-Doppler Map)
    rdm = np.mean(np.abs(doppler_fft), axis=1)
    
    # 构建返回的阵列元数据
    array_info = {
        'grid': grid,
        'grid_mask': grid_mask,
        'az_indices': az_element_indices,
        'el_indices': el_element_indices,
        'best_vx_col': best_vx_col,
        'num_vx': num_vx,
        'num_vz': num_vz,
    }
    
    return rdm, array_info, range_axis, doppler_axis


def capon_beamforming(data_vector, element_positions, scan_angles_deg, diag_load=0.1):
    """
    Capon (MVDR) 自适应波束形成角度估计
    
    Parameters
    ----------
    data_vector : ndarray, shape (N_elements,), complex
        某个 (doppler, range) bin 上各阵元的复数采样值
    element_positions : ndarray, shape (N_elements,), float
        阵元在扫描方向上的归一化位置 (以 d=λ/2 为单位)
    scan_angles_deg : ndarray, shape (N_scan,), float
        扫描角度网格 (度)
    diag_load : float
        对角加载系数 (相对于 trace(R)/N)，增强小阵列鲁棒性
        
    Returns
    -------
    theta_est : float
        估计角度 (度)
    spectrum : ndarray, shape (N_scan,)
        Capon 空间谱
    """
    N = len(data_vector)
    x = data_vector.reshape(N, 1)
    
    # 协方差矩阵 (单快拍 + 对角加载)
    R = x @ x.conj().T
    load = diag_load * np.trace(R).real / N
    R += load * np.eye(N)
    
    # 求逆 (对于 3~8 阵元的小矩阵，直接 inv 即可)
    try:
        R_inv = np.linalg.inv(R)
    except np.linalg.LinAlgError:
        # 奇异矩阵 fallback: 返回 0°
        return 0.0, np.zeros(len(scan_angles_deg))
    
    # 构建导向矢量矩阵并扫描 Capon 谱
    scan_rad = np.deg2rad(scan_angles_deg)
    spectrum = np.zeros(len(scan_angles_deg))
    
    for i, theta_r in enumerate(scan_rad):
        # 导向矢量: a(θ) = exp(-j * 2π * d_pos * sin(θ))
        # element_positions 已归一化为 d=λ/2 单位，空间频率 = sin(θ) / 2
        a = np.exp(-1j * np.pi * element_positions * np.sin(theta_r)).reshape(N, 1)
        denom = (a.conj().T @ R_inv @ a).real.item()
        spectrum[i] = 1.0 / max(denom, 1e-20)
    
    # 峰值搜索 + 抛物线插值精化
    peak_idx = np.argmax(spectrum)
    
    if 0 < peak_idx < len(spectrum) - 1:
        # 对数域抛物线插值
        y_l = np.log(spectrum[peak_idx - 1] + 1e-30)
        y_c = np.log(spectrum[peak_idx] + 1e-30)
        y_r = np.log(spectrum[peak_idx + 1] + 1e-30)
        denom_interp = 2 * y_c - y_l - y_r
        if abs(denom_interp) > 1e-12:
            delta = 0.5 * (y_l - y_r) / denom_interp
            delta = np.clip(delta, -0.5, 0.5)
            theta_est = scan_angles_deg[peak_idx] + delta * (scan_angles_deg[1] - scan_angles_deg[0])
        else:
            theta_est = scan_angles_deg[peak_idx]
    else:
        theta_est = scan_angles_deg[peak_idx]
    
    return theta_est, spectrum

def ca_cfar_2d(rdm, threshold_db=10.0, **kwargs):
    """
    噪声自适应阈值检测，替代传统 CA-CFAR
    
    CA-CFAR 对人体等扩展目标存在严重的"目标遮蔽"效应：
    多个目标 bin 互相靠近时，训练单元中的目标能量抬高了噪声估计，
    导致阈值虚高，大量真实目标被漏检。
    
    本方法使用全局中位数估计噪声底，再加上固定 dB 门限进行检测。
    对于人体室内场景更为可靠。
    
    Parameters
    ----------
    rdm : ndarray
        Range-Doppler Map
    threshold_db : float
        检测门限，高于噪声底多少 dB 判为目标（默认 10dB）
    """
    rdm_sq = rdm ** 2
    
    # 使用中位数作为噪声底估计（对目标占比 < 50% 的场景鲁棒）
    noise_floor = np.median(rdm_sq)
    
    # 转换 dB 门限为线性乘子
    threshold = noise_floor * (10 ** (threshold_db / 10.0))
    
    mask = rdm_sq > threshold
    
    return mask


def _find_spectrum_peaks(spectrum, scan_angles, peak_ratio=0.3, min_separation_deg=5.0):
    """
    在 Capon 空间谱中搜索所有显著峰值
    
    Parameters
    ----------
    spectrum : ndarray
        Capon 空间功率谱
    scan_angles : ndarray
        对应角度网格 (度)
    peak_ratio : float
        峰值阈值 = 全局最大值 × peak_ratio
    min_separation_deg : float
        两个峰值之间的最小角度间隔 (度)
        
    Returns
    -------
    peaks : list of float
        所有检出峰值对应的角度 (度)
    """
    if len(spectrum) < 3:
        return [scan_angles[np.argmax(spectrum)]]
    
    threshold = np.max(spectrum) * peak_ratio
    step = scan_angles[1] - scan_angles[0] if len(scan_angles) > 1 else 1.0
    min_sep_bins = max(1, int(min_separation_deg / abs(step)))
    
    # 搜索所有局部极大值
    candidates = []
    for i in range(1, len(spectrum) - 1):
        if spectrum[i] > spectrum[i-1] and spectrum[i] > spectrum[i+1] and spectrum[i] > threshold:
            # 抛物线插值精化
            y_l = np.log(spectrum[i-1] + 1e-30)
            y_c = np.log(spectrum[i] + 1e-30)
            y_r = np.log(spectrum[i+1] + 1e-30)
            denom = 2 * y_c - y_l - y_r
            if abs(denom) > 1e-12:
                delta = 0.5 * (y_l - y_r) / denom
                delta = np.clip(delta, -0.5, 0.5)
                angle = scan_angles[i] + delta * step
            else:
                angle = scan_angles[i]
            candidates.append((spectrum[i], angle, i))
    
    # 如果没有找到局部极大值，使用全局最大值
    if not candidates:
        return [scan_angles[np.argmax(spectrum)]]
    
    # 按能量降序排列，贪心选取（NMS 风格）
    candidates.sort(key=lambda x: -x[0])
    selected = []
    used_bins = set()
    
    for power, angle, idx in candidates:
        # 检查是否与已选峰值太近
        too_close = False
        for used_idx in used_bins:
            if abs(idx - used_idx) < min_sep_bins:
                too_close = True
                break
        if not too_close:
            selected.append(angle)
            used_bins.add(idx)
    
    return selected if selected else [scan_angles[np.argmax(spectrum)]]


def extract_point_cloud(array_info, range_axis, doppler_axis, rdm):
    """
    通过二维 CFAR + Capon 多峰波束形成 + 邻域扩展 提取 3D 雷达点云
    支持同一 Range-Doppler bin 的多角度目标提取
    """
    grid = array_info['grid']
    az_indices = array_info['az_indices']
    el_indices = array_info['el_indices']
    best_vx_col = array_info['best_vx_col']
    
    # Capon 角度扫描网格
    az_scan = np.linspace(-60, 60, 121)   # 方位角: -60° ~ +60°, 1°步长
    el_scan = np.linspace(-45, 45, 91)    # 俯仰角: -45° ~ +45°, 1°步长
    
    # 阵元位置 (归一化到 d=λ/2 单位)
    az_positions = az_indices.astype(float)
    el_positions = el_indices.astype(float)
    
    # --- CFAR 检测 ---
    cfar_mask = ca_cfar_2d(rdm)
    doppler_idx, range_idx = np.where(cfar_mask)
    
    # --- 邻域扩展: 收集 CFAR 检出点周围的高能量 bin ---
    N_dop, N_rng = rdm.shape
    NEIGHBOR_RADIUS = 2
    NEIGHBOR_RATIO = 0.4          # 放宽至 40%
    
    expanded_bins = set()
    for d_idx, r_idx in zip(doppler_idx, range_idx):
        expanded_bins.add((d_idx, r_idx))
        center_power = rdm[d_idx, r_idx] ** 2
        
        for dd in range(-NEIGHBOR_RADIUS, NEIGHBOR_RADIUS + 1):
            for dr in range(-NEIGHBOR_RADIUS, NEIGHBOR_RADIUS + 1):
                if dd == 0 and dr == 0:
                    continue
                nd, nr = d_idx + dd, r_idx + dr
                if 0 <= nd < N_dop and 0 <= nr < N_rng:
                    if rdm[nd, nr] ** 2 > center_power * NEIGHBOR_RATIO:
                        expanded_bins.add((nd, nr))
    
    # --- 对每个目标 bin 进行 Capon 多峰角度估计 ---
    pc_radar = []
    
    for d_idx, r_idx in expanded_bins:
        # 提取方位阵元数据
        az_data = grid[d_idx, 0, az_indices, r_idx]
        
        # 提取俯仰阵元数据
        el_data = grid[d_idx, el_indices, best_vx_col, r_idx]
        
        # Capon 方位角谱 → 多峰搜索
        _, az_spec = capon_beamforming(az_data, az_positions, az_scan)
        az_peaks = _find_spectrum_peaks(az_spec, az_scan, peak_ratio=0.3, min_separation_deg=5.0)
        
        # Capon 俯仰角谱 → 多峰搜索
        _, el_spec = capon_beamforming(el_data, el_positions, el_scan)
        el_peaks = _find_spectrum_peaks(el_spec, el_scan, peak_ratio=0.3, min_separation_deg=8.0)
        
        v = doppler_axis[d_idx]
        r = range_axis[r_idx]
        intensity = rdm[d_idx, r_idx]
        
        # 对每个 (方位峰, 俯仰峰) 组合生成一个点
        for theta in az_peaks:
            for phi in el_peaks:
                if np.isnan(theta) or np.isnan(phi):
                    continue
                
                theta_rad = np.deg2rad(theta)
                phi_rad = np.deg2rad(phi)
                
                x = r * np.cos(phi_rad) * np.sin(theta_rad)
                y = r * np.cos(phi_rad) * np.cos(theta_rad)
                z = r * np.sin(phi_rad)
                
                pc_radar.append([x, y, z, v, intensity])
    
    return np.array(pc_radar) if len(pc_radar) > 0 else np.zeros((0, 5))
