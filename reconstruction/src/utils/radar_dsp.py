'''
Author: Orange
Date: 2026-02-27 10:51
LastEditors: Orange
LastEditTime: 2026-03-05 17:50
FilePath: radar_dsp.py
Description: 
    Radar DSP: Process radar data and extract point clouds
    支持 CuPy GPU 加速 (自动检测, 可 fallback 到 NumPy CPU)
'''

import numpy as np
from scipy.ndimage import percentile_filter
from .radar_config import RadarConfig

# ------------------------------------------------------------------
# Auto choose GPU / CPU 
# First call simulate_adc to initialize
# ------------------------------------------------------------------
_xp = None 
CUDA_DEVICE_ID = 0  # Default to GPU 0

def _get_xp():
    """
    Auto choose GPU / CPU
    """
    global _xp
    if _xp is not None:
        return _xp
    try:
        import cupy as cp
        # 指定使用的显卡编号并检查可用性
        cp.cuda.Device(CUDA_DEVICE_ID).use()
        cp.cuda.Device(CUDA_DEVICE_ID).compute_capability
        _xp = (cp, True)
        print(f"[radar_dsp] ✓ CuPy GPU 加速已启用 (Device {CUDA_DEVICE_ID})")
    except Exception:
        _xp = (np, False)
        print("[radar_dsp] CuPy 不可用或显卡序号错误, 使用 NumPy CPU 模式")
    return _xp

def simulate_adc(pc, vel, rcs, config: RadarConfig, batch_size=None):
    """
    RS Point Cloud -> Radar ADC Raw Data
    采用分批 (Batched) 矩阵运算, 支持 CuPy GPU 加速。
    """
    xp, is_gpu = _get_xp()
    
    # BatchSize varies between gpu and cpu
    if batch_size is None:
        batch_size = 256 if is_gpu else 50
    
    P = pc.shape[0]

    # Virtual Array
    N_virt = config.NumTx * config.NumRx
    
    # 构建3D Radar Cube
    # complex64 精度
    adc_cube = xp.zeros((config.NumChirps, N_virt, config.NumSamples), dtype=xp.complex64)
    
    # 快慢时间维度
    t_fast = xp.asarray(np.arange(config.NumSamples, dtype=np.float32) / config.Fs)
    t_slow = xp.asarray(np.arange(config.NumChirps, dtype=np.float32) * config.PRT)
    
    # 构建虚拟阵列天线的位置（坐标矩阵）
    tx_arr_np = np.zeros((N_virt, 3), dtype=np.float32)
    rx_arr_np = np.zeros((N_virt, 3), dtype=np.float32)
    v_idx = 0

    # 记录虚拟天线中每个通道对应的收发天线
    for t in range(config.NumTx):
        for r in range(config.NumRx):
            tx_arr_np[v_idx] = config.TxPos[t]
            rx_arr_np[v_idx] = config.RxPos[r]
            v_idx += 1
    tx_arr = xp.asarray(tx_arr_np)
    rx_arr = xp.asarray(rx_arr_np)
    
    fc_f32 = np.float32(config.fc)
    K_f32 = np.float32(config.K)
    c_f64 = config.c  # 光速保持 float64 避免距离精度损失
            
    # 进入 Batch 循环
    for i in range(0, P, batch_size):
        idx_end = min(i + batch_size, P)
        pb_pc = xp.asarray(pc[i:idx_end].astype(np.float32))
        pb_vel = xp.asarray(vel[i:idx_end].astype(np.float32))
        pb_rcs = xp.asarray(rcs[i:idx_end].astype(np.float32))
        
        # Constant Velocity Model
        # pos_n = pb_pc + pb_vel * t_slow
        # (BatchSize, NumChirps, 3)
        pos_n = pb_pc[:, None, :] + pb_vel[:, None, :] * t_slow[None, :, None]
        
        # --- 利用广播一次性计算所有虚拟天线的距离 ---
        # pos_n：[Nums, Chirps, Pos] --> (BatchSize, NumChirps, 3)
        # tx_arr/rx_arr：[NumVirt, Pos] --> (NumVirt, 3)
        pos_n_exp = pos_n[:, :, None, :]

        # 散射点到收发天线距离（广播从-1元素开始比对）
        diff_tx = pos_n_exp - tx_arr
        diff_rx = pos_n_exp - rx_arr

        # R_tx/R_rx: (BatchSize, NumChirps, NumVirt)
        R_tx = xp.sqrt(xp.sum(diff_tx * diff_tx, axis=-1)) 
        R_rx = xp.sqrt(xp.sum(diff_rx * diff_rx, axis=-1))
                
        # 计算双程飞行时间 Tau
        # tau: (BatchSize, NumChirps, NumVirt)
        tau = (R_tx + R_rx) / c_f64  
        
        # IF 信号相位模型: 2 * pi * [f_c * tau + K * tau * t_fast]
        tau_exp = tau[:, :, :, None]
        # phase: (BatchSize, NumChirps, NumVirt, NumSamples)
        phase = np.float32(2 * np.pi) * (fc_f32 * tau_exp + K_f32 * tau_exp * t_fast[None, None, None, :])
        
        # 幅值衰减模型：室内近距 (1-3m) 人体场景采用 1/R² 软化模型
        # 理论 1/R⁴ 在近距离动态范围过大，导致弱散射点被躯干掩盖
        R_tx_clip = xp.clip(R_tx, 0.1, None)
        R_rx_clip = xp.clip(R_rx, 0.1, None)
        # 目前简单认为人体各个散射点RCS相同
        amp = xp.sqrt(pb_rcs)[:, None, None] / (R_tx_clip * R_rx_clip)
        
        # 生成复基带 IF 信号并叠加至 ADC 立方体
        # signal: (BatchSize, NumChirps, NumVirt, NumSamples)
        signal = amp[:, :, :, None] * xp.exp(1j * phase)

        # 人体目标信号合成
        adc_cube += xp.sum(signal, axis=0)
        
        # 释放 batch 中间变量
        del pos_n, pos_n_exp, diff_tx, diff_rx, R_tx, R_rx, tau, tau_exp, phase, amp, signal
        
    # 添加高斯白噪声模拟接收机真实热噪声与杂波
    # 基于纯理论回波的平均信号功率，设置一个动态的 SNR = 20dB
    sig_power = float(xp.mean(xp.abs(adc_cube) ** 2))
    target_snr_db = 20.0

    # 噪声功率
    noise_power = sig_power / (10 ** (target_snr_db / 10.0))
    if noise_power == 0:
        noise_power = 1e-12  # 防除0 
    # 噪声方差
    noise_std = np.float32(np.sqrt(noise_power / 2))
    # 复噪声信号
    noise = (xp.random.normal(0, noise_std, adc_cube.shape).astype(xp.float32) +
             1j * xp.random.normal(0, noise_std, adc_cube.shape).astype(xp.float32))
    adc_cube += noise
    
    # 加入仪器本振（VCO）相位噪声
    phase_noise = xp.random.normal(0, np.float32(np.deg2rad(1.0)), size=adc_cube.shape).astype(xp.float32)
    adc_cube *= xp.exp(1j * phase_noise)
    
    # GPU 模式: 将结果传回 CPU
    if is_gpu:
        adc_cube = adc_cube.get()
        
    # (NumChirps, NumVirt, NumSamples)
    return adc_cube

def process_radar_data(adc_cube, config: RadarConfig):
    """
    DSP链路：Range FFT -> Doppler FFT -> 构建虚拟阵列网格
    角度估计：Capon 波束形成
    """
    # 距离维 FFT 
    n_fft_range = config.NumSamples * 2
    win_range = np.hanning(config.NumSamples)
    adc_cube_w = adc_cube * win_range[None, None, :]
    range_fft = np.fft.fft(adc_cube_w, n=n_fft_range, axis=2)
    range_fft = range_fft[:, :, :n_fft_range // 2] # 截取正频率段 (对应 0 -> R_max)
    
    # 距离物理坐标轴
    fast_freqs = np.fft.fftfreq(n_fft_range, d=1/config.Fs)[:n_fft_range // 2]
    range_axis = fast_freqs * config.c / (2 * config.K)
    
    # 多普勒维 FFT
    win_doppler = np.hanning(config.NumChirps)
    range_fft_w = range_fft * win_doppler[:, None, None]
    doppler_fft = np.fft.fftshift(np.fft.fft(range_fft_w, axis=0), axes=0)
    
    # 多普勒物理坐标轴
    slow_freqs = np.fft.fftshift(np.fft.fftfreq(config.NumChirps, d=config.PRT))
    doppler_axis = slow_freqs * config.wavelength / 2

    # 聚合获得 距离-多普勒 雷达图 (Range-Doppler Map)
    rdm = np.mean(np.abs(doppler_fft), axis=1)
    
    # 构建虚拟天线面阵网格
    d = config.wavelength / 2
    n_bins_range = n_fft_range // 2
    
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
    grid = np.zeros((config.NumChirps, num_vz, num_vx, n_bins_range), dtype=np.complex64)
    
    # 记录哪些网格位置有真实天线数据
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
    
    # 全向量化: 一次构建所有导向矢量矩阵并批量求 Capon 谱
    # 导向矢量: a_n(θ) = exp(-jπ * pos_n * sin(θ))
    # element_positions 已归一化为 d=λ/2 单位，空间频率 = sin(θ) / 2
    scan_rad = np.deg2rad(scan_angles_deg)  # (M,)
    A = np.exp(-1j * np.pi * element_positions[:, None] * np.sin(scan_rad[None, :]))  # (N, M)
    R_inv_A = R_inv @ A  # (N, M)
    # Capon 谱: P(θ) = 1 / Re(a^H R^{-1} a), 对角线元素 = 逐列点积
    denom = np.real(np.sum(A.conj() * R_inv_A, axis=0))  # (M,)
    spectrum = 1.0 / np.maximum(denom, 1e-20)
    
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

def ca_cfar_2d(rdm, guard_cells=(2, 2), train_cells=(8, 8), pfa=1e-3):
    """
    OS-CFAR (Ordered Statistics CFAR)
    
    使用训练窗口内的第 75 百分位值作为噪声估计。
    即使窗口中 25% 的单元被目标污染，噪声估计仍然准确，
    从而解决了传统 CA-CFAR 对扩展目标的遮蔽问题。
    """
    gr, gc = guard_cells
    tr, tc = train_cells
    
    rdm_sq = rdm ** 2
    
    # OS-CFAR: 用大窗口 75th 百分位作为鲁棒噪声估计
    win_r = 2 * (gr + tr) + 1
    win_c = 2 * (gc + tc) + 1
    noise_est = percentile_filter(rdm_sq, percentile=75, size=(win_r, win_c))
    
    # 阈值乘子 (从 pfa 推导)
    N_train = win_r * win_c - (2*gr+1) * (2*gc+1)
    alpha = N_train * (pfa ** (-1.0 / max(N_train, 1)) - 1)
    alpha = np.clip(alpha, 3.0, 30.0)
    
    threshold = alpha * noise_est
    
    # 附加下界: 防止纯噪声区域随机检出
    min_power = np.max(rdm_sq) * 1e-5
    mask = (rdm_sq > threshold) & (rdm_sq > min_power)
    
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
    OS-CFAR 检测 + Capon 波束形成 + 邻域扩展提取 3D 雷达点云
    
    - CFAR 检出 bin: 允许多角度提取 (Capon 多峰)
    - 邻域扩展 bin: 仅取最强单峰 (避免噪声虚假角度)
    - 所有 bin 的 Capon 峰值须通过信号质量验证
    """
    grid = array_info['grid']
    az_indices = array_info['az_indices']
    el_indices = array_info['el_indices']
    best_vx_col = array_info['best_vx_col']
    
    az_scan = np.linspace(-60, 60, 121)
    el_scan = np.linspace(-45, 45, 91)
    az_positions = az_indices.astype(float)
    el_positions = el_indices.astype(float)
    
    # 信号质量阈值：Capon 谱峰值/均值 > 此值才接受该角度
    PEAK_SNR_THRESHOLD = 3.5
    
    # --- OS-CFAR 检测 ---
    cfar_mask = ca_cfar_2d(rdm)
    doppler_idx, range_idx = np.where(cfar_mask)
    
    cfar_bins = set()
    for d_idx, r_idx in zip(doppler_idx, range_idx):
        cfar_bins.add((d_idx, r_idx))
    
    # --- 邻域扩展（仅围绕 CFAR 检出点） ---
    N_dop, N_rng = rdm.shape
    NEIGHBOR_RADIUS = 2
    NEIGHBOR_RATIO = 0.55
    
    neighbor_bins = set()
    for d_idx, r_idx in cfar_bins:
        center_power = rdm[d_idx, r_idx] ** 2
        for dd in range(-NEIGHBOR_RADIUS, NEIGHBOR_RADIUS + 1):
            for dr in range(-NEIGHBOR_RADIUS, NEIGHBOR_RADIUS + 1):
                if dd == 0 and dr == 0:
                    continue
                nd, nr = d_idx + dd, r_idx + dr
                if 0 <= nd < N_dop and 0 <= nr < N_rng:
                    key = (nd, nr)
                    if key not in cfar_bins and rdm[nd, nr] ** 2 > center_power * NEIGHBOR_RATIO:
                        neighbor_bins.add(key)
    
    # --- 角度估计辅助函数 ---
    def _estimate_angles(d_idx, r_idx, allow_multi_peak):
        az_data = grid[d_idx, 0, az_indices, r_idx]
        el_data = grid[d_idx, el_indices, best_vx_col, r_idx]
        
        _, az_spec = capon_beamforming(az_data, az_positions, az_scan)
        _, el_spec = capon_beamforming(el_data, el_positions, el_scan)
        
        # 信号质量验证
        az_snr = np.max(az_spec) / (np.mean(az_spec) + 1e-30)
        el_snr = np.max(el_spec) / (np.mean(el_spec) + 1e-30)
        
        if az_snr < PEAK_SNR_THRESHOLD:
            az_list = [0.0]
        elif allow_multi_peak:
            az_list = _find_spectrum_peaks(az_spec, az_scan, peak_ratio=0.5, min_separation_deg=10.0)
        else:
            az_list = [az_scan[np.argmax(az_spec)]]
        
        if el_snr < PEAK_SNR_THRESHOLD:
            el_list = [0.0]
        elif allow_multi_peak:
            el_list = _find_spectrum_peaks(el_spec, el_scan, peak_ratio=0.5, min_separation_deg=10.0)
        else:
            el_list = [el_scan[np.argmax(el_spec)]]
        
        return az_list, el_list
    
    pc_radar = []
    
    # CFAR 检出 bin → 多角度提取
    for d_idx, r_idx in cfar_bins:
        az_peaks, el_peaks = _estimate_angles(d_idx, r_idx, allow_multi_peak=True)
        v = doppler_axis[d_idx]
        r = range_axis[r_idx]
        intensity = rdm[d_idx, r_idx]
        
        for theta in az_peaks:
            for phi in el_peaks:
                if np.isnan(theta) or np.isnan(phi):
                    continue
                theta_rad, phi_rad = np.deg2rad(theta), np.deg2rad(phi)
                x = r * np.cos(phi_rad) * np.sin(theta_rad)
                y = r * np.cos(phi_rad) * np.cos(theta_rad)
                z = r * np.sin(phi_rad)
                pc_radar.append([x, y, z, v, intensity])
    
    # 邻域扩展 bin → 仅最强单峰
    for d_idx, r_idx in neighbor_bins:
        az_peaks, el_peaks = _estimate_angles(d_idx, r_idx, allow_multi_peak=False)
        v = doppler_axis[d_idx]
        r = range_axis[r_idx]
        intensity = rdm[d_idx, r_idx]
        theta, phi = az_peaks[0], el_peaks[0]
        if np.isnan(theta) or np.isnan(phi):
            continue
        theta_rad, phi_rad = np.deg2rad(theta), np.deg2rad(phi)
        x = r * np.cos(phi_rad) * np.sin(theta_rad)
        y = r * np.cos(phi_rad) * np.cos(theta_rad)
        z = r * np.sin(phi_rad)
        pc_radar.append([x, y, z, v, intensity])
    
    return np.array(pc_radar) if len(pc_radar) > 0 else np.zeros((0, 5))
