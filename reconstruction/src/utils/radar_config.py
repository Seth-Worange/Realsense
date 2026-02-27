import numpy as np
import scipy.constants as const

class RadarConfig:
    def __init__(self):
        self.c = const.c         # 光速 (m/s)
        self.fc = 77e9           # 载波频率 (Hz) (77GHz mmWave)
        self.B = 4e9             # 扫频带宽 (Hz) (4 GHz)
        self.Tc = 40e-6          # 单个 Chirp 扫频时间 (s) (40 us)
        self.Fs = 10e6           # 采样率 (Hz) (10 MHz)
        
        self.NumSamples = 256    # 快时间采样点数
        self.NumChirps = 128     # 慢时间 Chirp 数
        
        # 物理时长与常数
        self.K = self.B / self.Tc # 调频斜率
        self.PRT = 50e-6         # 脉冲重复时间 (Chirp 周期)
        
        # 虚拟天线阵列配置 (MIMO)
        # 这里为了简化和高效，默认定义 1Tx, 8Rx 的虚拟均匀线阵 (ULA) 用于方位角估计
        self.TxPos = np.array([[0, 0, 0]])
        self.NumRx = 8
        self.wavelength = self.c / self.fc
        d = self.wavelength / 2
        
        self.RxPos = np.zeros((self.NumRx, 3))
        self.RxPos[:, 0] = np.arange(self.NumRx) * d # Rx 沿 X 轴排布 (间距 lambda/2)
