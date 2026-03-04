"""
诊断脚本：分析人体点云经雷达仿真管线后各阶段的信号状态
目的：找出手臂等部位点云丢失的根因
"""
import numpy as np
import sys, os, types, importlib.util

def _load_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils")
_cfg_mod = _load_module("radar_config", os.path.join(_base, "radar_config.py"))

_pkg = types.ModuleType("utils")
_pkg.__path__ = [_base]
sys.modules["utils"] = _pkg
sys.modules["utils.radar_config"] = _cfg_mod
_dsp_mod = _load_module("utils.radar_dsp", os.path.join(_base, "radar_dsp.py"))

RadarConfig = _cfg_mod.RadarConfig
simulate_adc = _dsp_mod.simulate_adc
process_radar_data = _dsp_mod.process_radar_data
ca_cfar_2d = _dsp_mod.ca_cfar_2d
extract_point_cloud = _dsp_mod.extract_point_cloud

_seq_mod = _load_module("utils.seq_loader", os.path.join(_base, "seq_loader.py"))
sys.modules["utils.seq_loader"] = _seq_mod
PointCloudSequencePlayer = _seq_mod.PointCloudSequencePlayer

def main():
    config = RadarConfig()
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    seq_path = os.path.join(base_dir, "datas", "rs_pointcloud", "seq_20260227_150659")
    player = PointCloudSequencePlayer(seq_path, fps=6.0)
    
    target_frame = 50
    pc, vel, rcs = player.get_all_as_radar_targets(frame_idx=target_frame)
    
    if pc is None:
        print("载入失败")
        return
    
    # 坐标变换 (同 radar_generate.py)
    pc_radar = np.zeros_like(pc)
    pc_radar[:, 0] = pc[:, 0]
    pc_radar[:, 1] = pc[:, 2]
    pc_radar[:, 2] = -pc[:, 1]
    
    vel_radar = np.zeros_like(vel)
    vel_radar[:, 0] = vel[:, 0]
    vel_radar[:, 1] = vel[:, 2]
    vel_radar[:, 2] = -vel[:, 1]
    
    rcs = rcs * np.random.uniform(0.8, 1.2, size=rcs.shape)
    
    print("=" * 70)
    print("阶段 1：输入点云统计")
    print("=" * 70)
    print(f"  总点数: {len(pc_radar)}")
    print(f"  X 范围: [{pc_radar[:,0].min():.2f}, {pc_radar[:,0].max():.2f}] m")
    print(f"  Y 范围: [{pc_radar[:,1].min():.2f}, {pc_radar[:,1].max():.2f}] m")
    print(f"  Z 范围: [{pc_radar[:,2].min():.2f}, {pc_radar[:,2].max():.2f}] m")
    
    R = np.sqrt(np.sum(pc_radar**2, axis=1))
    print(f"  距离范围: [{R.min():.2f}, {R.max():.2f}] m")
    print(f"  RCS: 均值={rcs.mean():.3f}, 全部相同={np.all(rcs == rcs[0])}")
    
    # 计算每个点的理论接收功率
    amp_per_point = np.sqrt(rcs) / (R ** 4)
    amp_per_point = np.clip(amp_per_point, 0, None)
    
    print(f"\n  理论接收幅值 (amp ~ sqrt(RCS)/R^4):")
    print(f"    最强: {amp_per_point.max():.6e}")
    print(f"    最弱: {amp_per_point[amp_per_point > 0].min():.6e}")
    print(f"    比值: {amp_per_point.max() / amp_per_point[amp_per_point > 0].min():.1f}x")
    
    # 分析 Range-Doppler bin 分布
    print("\n" + "=" * 70)
    print("阶段 2：Range-Doppler bin 占用分析")
    print("=" * 70)
    
    range_res = config.c / (2 * config.B)
    vel_res = config.wavelength / (2 * config.NumChirps * config.PRT)
    print(f"  距离分辨率: {range_res:.4f} m")
    print(f"  速度分辨率: {vel_res:.4f} m/s")
    
    # 为每个点计算它落入哪个 (range, doppler) bin
    v_radial = np.sum(vel_radar * pc_radar, axis=1) / R  # 径向速度
    range_bin = (R / range_res).astype(int)
    doppler_bin = ((v_radial + config.wavelength/(4*config.PRT)) / vel_res).astype(int)
    
    # 统计每个 bin 被多少个点占据
    bin_keys = list(zip(range_bin.tolist(), doppler_bin.tolist()))
    from collections import Counter
    bin_counts = Counter(bin_keys)
    
    total_unique_bins = len(bin_counts)
    multi_point_bins = sum(1 for c in bin_counts.values() if c > 1)
    max_sharing = max(bin_counts.values())
    avg_sharing = np.mean(list(bin_counts.values()))
    
    print(f"  总输入点数: {len(pc_radar)}")
    print(f"  唯一 (R, v) bin 数: {total_unique_bins}")
    print(f"  >1 个点共享同 bin: {multi_point_bins} bins")
    print(f"  最大共享数: {max_sharing} 个点/bin")
    print(f"  平均 bin 内点数: {avg_sharing:.1f}")
    print(f"  → 按 bin 上限估算，最多可提取 {total_unique_bins} 个点")
    
    # 运行完整管线
    print("\n" + "=" * 70)
    print("阶段 3：CFAR 检出分析")
    print("=" * 70)
    
    adc_cube = simulate_adc(pc_radar, vel_radar, rcs, config)
    rdm, array_info, range_axis, doppler_axis = process_radar_data(adc_cube, config)
    
    cfar_mask = ca_cfar_2d(rdm)
    cfar_count = np.sum(cfar_mask)
    print(f"  RDM shape: {rdm.shape}")
    print(f"  RDM 能量: max={rdm.max():.6e}, mean={rdm.mean():.6e}, ratio={rdm.max()/rdm.mean():.1f}x")
    print(f"  CFAR 检出 bin 数: {cfar_count}")
    
    # 分析 RDM 中目标能量分布
    rdm_db = 10 * np.log10(rdm + 1e-30)
    rdm_sq = rdm ** 2
    threshold_ratio = rdm_sq.max() * 1e-6
    significant_bins = np.sum(rdm_sq > threshold_ratio)
    print(f"  RDM 中显著能量 bin 数 (>max*1e-6): {significant_bins}")
    print(f"  RDM 中 CFAR 检出比例: {cfar_count}/{significant_bins} = {cfar_count/max(significant_bins,1)*100:.1f}%")
    
    # 提取点云
    radar_pc = extract_point_cloud(array_info, range_axis, doppler_axis, rdm)
    print(f"\n  最终输出雷达点云数: {len(radar_pc)}")
    if len(radar_pc) > 0:
        print(f"  X 范围: [{radar_pc[:,0].min():.2f}, {radar_pc[:,0].max():.2f}] m")
        print(f"  Y 范围: [{radar_pc[:,1].min():.2f}, {radar_pc[:,1].max():.2f}] m")
        print(f"  Z 范围: [{radar_pc[:,2].min():.2f}, {radar_pc[:,2].max():.2f}] m")
    
    # 点云覆盖率分析 (用空间体素统计)
    print("\n" + "=" * 70)
    print("阶段 4：空间覆盖率分析")
    print("=" * 70)
    
    voxel_size = 0.3  # 30cm 体素
    gt_voxels = set()
    for p in pc_radar:
        vk = (int(p[0]/voxel_size), int(p[1]/voxel_size), int(p[2]/voxel_size))
        gt_voxels.add(vk)
    
    radar_voxels = set()
    if len(radar_pc) > 0:
        for p in radar_pc:
            vk = (int(p[0]/voxel_size), int(p[1]/voxel_size), int(p[2]/voxel_size))
            radar_voxels.add(vk)
    
    covered = len(gt_voxels & radar_voxels)
    print(f"  GT 占据体素: {len(gt_voxels)}")
    print(f"  雷达覆盖体素: {len(radar_voxels)}")
    print(f"  覆盖率: {covered}/{len(gt_voxels)} = {covered/max(len(gt_voxels),1)*100:.1f}%")

if __name__ == "__main__":
    main()
