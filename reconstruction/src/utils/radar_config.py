'''
Author: Orange
Date: 2026-02-27 10:51
LastEditors: Orange
LastEditTime: 2026-02-27 17:06
FilePath: radar_config.py
Description: 
    Radar Config: Configuration for radar system
'''

import numpy as np
import scipy.constants as const

class RadarConfig:
    def __init__(self):
        self.c = const.c         # 光速 (m/s)
        self.fc = 60e9           # 载波频率 (Hz)
        self.K_slope = 200e12    # 调频斜率 (Hz/s), 200MHz/us = 200e6 * 1e6 = 200e12 Hz/s
        self.B = 3e9             # 扫频带宽 (Hz)
        self.Tc = self.B / self.K_slope          # 单个 Chirp 扫频时间 (s) = 15us
        self.Fs = 10e6           # 采样率 (Hz) 
        
        self.NumSamples = 256    # 快时间采样点数
        self.NumChirps = 128     # 慢时间 Chirp 数
        
        # 物理时长与常数
        self.K = self.K_slope    # 调频斜率
        self.PRT = 50e-6         # 脉冲重复时间 (Chirp 周期)
        
        # 虚拟天线阵列配置 (MIMO)
        # 根据用户要求的 4Tx 4Rx，采用 TDM-MIMO 模式产生 16 根虚拟接收天线
        self.NumTx = 4
        self.NumRx = 4
        
        self.wavelength = self.c / self.fc
        d = self.wavelength / 2
        
        # 发射天线 (Tx) 排布: 构造不规则面阵
        # TX1: (0, 0, 0)
        # TX2: (4*d, 0, 0)
        # TX3: (4*d, 0, d)
        # TX4: (4*d, 0, 2*d)
        self.TxPos = np.zeros((self.NumTx, 3))
        self.TxPos[0] = [0, 0, 0]
        self.TxPos[1] = [4 * d, 0, 0]
        self.TxPos[2] = [4 * d, 0, d]
        self.TxPos[3] = [4 * d, 0, 2 * d]
        
        # 接收天线 (Rx) 排布: 间隔为 d 均匀排布于 X 轴
        self.RxPos = np.zeros((self.NumRx, 3))
        self.RxPos[:, 0] = np.arange(self.NumRx) * d
