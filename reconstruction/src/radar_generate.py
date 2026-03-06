'''
Author: Orange
Date: 2026-02-26 16:53
LastEditors: Orange
LastEditTime: 2026-03-06 11:29
FilePath: radar_generate.py
Description: Vectorized simulator for mmWave FMCW Radar Human Echoes
             Supports single-frame (PNG) and continuous multi-frame (GIF) modes.
'''

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import time
import os
import argparse
import io

from utils.seq_loader import PointCloudSequencePlayer
from utils.radar_config import RadarConfig
from utils.radar_dsp import simulate_adc, process_radar_data, extract_point_cloud


def transform_to_radar_frame(pc, vel, radar_pitch_deg=0):
    """
    RealSense  → Radar Coordinate
    RealSense: X right, Y down, Z forward
    Radar:     X right (azimuth), Y forward (depth), Z up (elevation)
    """
    pc_radar = np.zeros_like(pc)
    pc_radar[:, 0] = pc[:, 0]     # Radar X = RS X
    pc_radar[:, 1] = pc[:, 2]     # Radar Y = RS Z
    pc_radar[:, 2] = -pc[:, 1]    # Radar Z = -RS Y

    vel_radar = np.zeros_like(vel)
    vel_radar[:, 0] = vel[:, 0]
    vel_radar[:, 1] = vel[:, 2]
    vel_radar[:, 2] = -vel[:, 1]

    # Pitch angle compensation
    theta = np.deg2rad(radar_pitch_deg)
    # Rotation matrix
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta),  np.cos(theta)]
    ])
    pc_radar = pc_radar @ R_x.T
    vel_radar = vel_radar @ R_x.T

    return pc_radar, vel_radar


def process_single_frame(pc, vel, rcs, config):
    """
    Single frame processing
    
    Returns: (pc_radar_frame, rdm, radar_pc, range_axis, doppler_axis)
    """
    pc_radar, vel_radar = transform_to_radar_frame(pc, vel)
    rcs = rcs * np.random.uniform(0.8, 1.2, size=rcs.shape)

    adc_cube = simulate_adc(pc_radar, vel_radar, rcs, config)
    rdm, array_info, range_axis, doppler_axis = process_radar_data(adc_cube, config)
    radar_pc = extract_point_cloud(array_info, range_axis, doppler_axis, rdm)

    return pc_radar, vel_radar, rdm, radar_pc, range_axis, doppler_axis


def render_rd_map(rdm, range_axis, doppler_axis, frame_idx=None):
    """
    Save Range-Doppler Map to PIL Image
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    rdm_db = 10 * np.log10(rdm + 1e-10)
    im = ax.imshow(rdm_db, aspect='auto',
                   extent=[range_axis[0], range_axis[-1], doppler_axis[0], doppler_axis[-1]],
                   cmap='jet', origin='lower', vmin=5, vmax=50)
    title = 'Range-Doppler Map'
    if frame_idx is not None:
        title += f' (Frame {frame_idx})'
    ax.set_title(title)
    ax.set_xlabel('Range (m)')
    ax.set_ylabel('Velocity (m/s)')
    fig.colorbar(im, label='Power (dB)')
    fig.tight_layout()

    img = _fig_to_pil(fig)
    plt.close(fig)
    return img


def render_rs_pointcloud(pc_radar, vel_radar, frame_idx=None):
    """
    Save RealSense Point Cloud to PIL Image
    """
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pc_radar[:, 0], pc_radar[:, 1], pc_radar[:, 2],
               c='steelblue', s=2, alpha=0.1)

    # Draw velocity arrows (downsample to avoid excessive density)
    stride = 30
    p_s = pc_radar[::stride]
    v_s = vel_radar[::stride]
    if len(p_s) > 0:
        ax.quiver(p_s[:, 0], p_s[:, 1], p_s[:, 2],
                  v_s[:, 0], v_s[:, 1], v_s[:, 2],
                  length=0.3, normalize=False, color='red', alpha=0.6,
                  arrow_length_ratio=0.3, linewidth=0.5)

    title = 'RealSense Point Cloud (Radar Frame)'
    if frame_idx is not None:
        title += f' (Frame {frame_idx})'
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_xlim(-2, 2)
    ax.set_ylim(0, 8)
    ax.set_zlim(-1, 2)
    ax.view_init(elev=20, azim=-60)
    fig.tight_layout()

    img = _fig_to_pil(fig)
    plt.close(fig)
    return img


def render_radar_pointcloud(pc_radar, radar_pc, frame_idx=None):
    """
    Save Radar Point Cloud to PIL Image (Base: Ground Truth)
    """
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')

    # Ground Truth
    ax.scatter(pc_radar[:, 0], pc_radar[:, 1], pc_radar[:, 2],
               c='gray', s=2, alpha=0.2, label='Ground Truth', vmin=-3, vmax=3)

    # Radar Point Cloud
    if len(radar_pc) > 0:
        scatter = ax.scatter(radar_pc[:, 0], radar_pc[:, 1], radar_pc[:, 2],
                             c=radar_pc[:, 3], cmap='coolwarm', s=20,
                             edgecolors='black', alpha=0.9)
        fig.colorbar(scatter, label='Velocity (m/s)', ax=ax, shrink=0.5, pad=0.1)

    title = 'Simulated Radar Point Cloud'
    if frame_idx is not None:
        title += f' (Frame {frame_idx})'
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.legend(fontsize=8)
    ax.set_xlim(-2, 2)
    ax.set_ylim(0, 8)
    ax.set_zlim(-1, 2)
    ax.view_init(elev=20, azim=-60)
    fig.tight_layout()

    img = _fig_to_pil(fig)
    plt.close(fig)
    return img


def _fig_to_pil(fig):
    """
    save matplotlib figure to PIL Image
    """
    # use buffer to accelerate
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120)
    buf.seek(0)
    img = Image.open(buf).convert('RGB')
    return img


def save_gif(images, output_path, fps):
    """
    Save PIL Image list to GIF
    """
    if not images:
        print(f"  [跳过] 没有可用帧，未生成 {output_path}")
        return
    duration_ms = int(1000.0 / fps)
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0
    )
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  -> 已保存 {output_path}  ({len(images)} 帧, {size_mb:.1f} MB)")


def run_single_frame(player, config, target_frame, output_dir):
    """
    Single frame mode
    """
    print(f"读取第 {target_frame} 帧 RS 点云数据")
    pc, vel, rcs = player.get_all_as_radar_targets(frame_idx=target_frame)

    if pc is None:
        print(f"载入失败: 请检查文件路径下是否有起码 {target_frame} 帧以上的 .ply 文件")
        return

    pc_radar, vel_radar, rdm, radar_pc, range_axis, doppler_axis = process_single_frame(pc, vel, rcs, config)
    print(f"DSP 完成, 检出 {len(radar_pc)} 个雷达反射点.")

    # Figure draw
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    rdm_db = 10 * np.log10(rdm + 1e-10)
    plt.imshow(rdm_db, aspect='auto',
               extent=[range_axis[0], range_axis[-1], doppler_axis[0], doppler_axis[-1]],
               cmap='jet', origin='lower', vmin=5, vmax=50)
    plt.title('Range-Doppler Map (RDM)')
    plt.xlabel('Range (m)')
    plt.ylabel('Velocity (m/s)')
    plt.colorbar(label='Power (dB)')

    ax = plt.subplot(1, 2, 2, projection='3d')
    ax.scatter(pc_radar[:, 0], pc_radar[:, 1], pc_radar[:, 2],
               c='gray', s=2, alpha=0.2, label='Ground Truth PC')
    if len(radar_pc) > 0:
        scatter = ax.scatter(radar_pc[:, 0], radar_pc[:, 1], radar_pc[:, 2],
                             c=radar_pc[:, 3], cmap='coolwarm', s=30,
                             edgecolors='black', alpha=0.9)
        plt.colorbar(scatter, label='Velocity (m/s)', ax=ax, shrink=0.5, pad=0.1)
    ax.set_title('3D Radar Point Cloud')
    ax.set_xlabel('Azimuth/X (m)')
    ax.set_ylabel('Depth/Y (m)')
    ax.set_zlabel('Height/Z (m)')
    ax.legend()
    ax.set_xlim(-2, 2)
    ax.set_ylim(0, 8)
    ax.set_zlim(-1, 2)
    ax.view_init(elev=20, azim=-60)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'radar_simulation_result.png')
    plt.savefig(output_path, dpi=300)
    print(f" -> 成功！结果图已存至 {output_path}")
    plt.close()


def run_continuous(player, config, output_dir):
    """
    Continuous mode
    """
    total_frames = player.get_frame_count()
    print(f"连续模式: 共 {total_frames} 帧, 开始逐帧仿真...")

    rd_images = []
    rs_images = []
    radar_images = []

    t0 = time.time()

    for frame_idx, pc, vel, rcs in player.iter_all_as_radar_targets():
        frame_t0 = time.time()

        pc_radar, vel_radar, rdm, radar_pc, range_axis, doppler_axis = process_single_frame(pc, vel, rcs, config)

        # RD-Map, RS-Pointcloud, Radar-Pointcloud
        rd_images.append(render_rd_map(rdm, range_axis, doppler_axis, frame_idx))
        rs_images.append(render_rs_pointcloud(pc_radar, vel_radar, frame_idx))
        radar_images.append(render_radar_pointcloud(pc_radar, radar_pc, frame_idx))

        elapsed = time.time() - frame_t0
        print(f"  帧 {frame_idx:4d}/{total_frames} | "
              f"雷达点: {len(radar_pc):4d} | "
              f"耗时: {elapsed:.2f}s")

    total_time = time.time() - t0
    print(f"\n全部 {len(rd_images)} 帧处理完成, 总耗时 {total_time:.1f}s")

    # Save GIF
    print("正在生成 GIF ...")
    fps = player.fps
    save_gif(rd_images, os.path.join(output_dir, 'rd_spectrum.gif'), fps)
    save_gif(rs_images, os.path.join(output_dir, 'rs_pointcloud.gif'), fps)
    save_gif(radar_images, os.path.join(output_dir, 'radar_pointcloud.gif'), fps)
    print("全部完成!")


def main():
    parser = argparse.ArgumentParser(description='mmWave FMCW Radar Simulation from RealSense Point Cloud')
    parser.add_argument('--mode', type=str, default='continuous', choices=['single', 'continuous'],
                        help='运行模式: single (单帧, 输出 PNG) 或 continuous (全序列, 输出 GIF)')
    parser.add_argument('--frame', type=int, default=50,
                        help='单帧模式下的目标帧号 (默认: 50)')
    parser.add_argument('--seq-path', type=str, default=None,
                        help='点云序列目录路径 (默认: datas/rs_pointcloud/seq_20260227_150659)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='输出目录 (默认: images/)')
    args = parser.parse_args()

    print("初始化雷达配置参数")
    config = RadarConfig()

    # 路径解析
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if args.seq_path:
        seq_path = args.seq_path
    else:
        seq_path = os.path.join(base_dir, "datas", "rs_pointcloud", "seq_20260227_150659")

    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(base_dir, 'images')

    os.makedirs(output_dir, exist_ok=True)

    player = PointCloudSequencePlayer(seq_path, fps=6.0)
    print(f"已加载序列: {seq_path} ({player.get_frame_count()} 帧)")

    if args.mode == 'single':
        run_single_frame(player, config, args.frame, output_dir)
    else:
        run_continuous(player, config, output_dir)


if __name__ == "__main__":
    main()
