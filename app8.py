import librosa
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.signal import fftconvolve


def difference_function2(x, N, tau_max):
    """计算差分函数（修复广播错误版本）"""
    x = np.array(x, dtype=np.float64)
    w = len(x)
    tau_max = min(tau_max, w)  # 确保tau_max不超过信号长度
    
    # 1. 计算累积平方和（优化索引范围）
    x_cumsum = np.concatenate(([0.], (x**2).cumsum()))
    
    # 2. 使用FFT计算自相关（修正填充逻辑）
    x_padded = np.concatenate([x, np.zeros(w)])
    fft_x = np.fft.rfft(x_padded)
    autocorr = np.fft.irfft(fft_x * fft_x.conjugate())[:w]
    
    # 3. 向量化计算差分函数（关键修复点）
    tau_range = np.arange(1, tau_max+1)
    sum_x = x_cumsum[w] - x_cumsum[tau_range] - x_cumsum[w - tau_range] + x_cumsum[0]
    df = sum_x - 2 * autocorr[tau_range]
    
    return df

def difference_function(x, N, tau_max):
    """计算差分函数（核心步骤1）- 已修复广播错误"""
    x = np.array(x, dtype=np.float64)
    w = x.size
    tau_max = min(tau_max, w)
    
    # 1. 计算累积平方和（索引从1开始）
    x_cumsum = np.concatenate(([0.], (x**2).cumsum()))
    
    # 2. 修正FFT填充方式（关键修复点）
    # 使用两倍长度避免循环卷积，替代原size_pad逻辑
    x_padded = np.concatenate([x, np.zeros_like(x)])
    fft_x = np.fft.rfft(x_padded)
    autocorr = np.fft.irfft(fft_x * fft_x.conjugate())[:w]
    
    # 3. 向量化计算差分函数（统一数组长度）
    tau_range = np.arange(1, tau_max+1)  # τ从1开始到tau_max
    df = (x_cumsum[w] - x_cumsum[tau_range] -  # 修正索引切片方式
          (x_cumsum[w - tau_range] - x_cumsum[0])) - 2 * autocorr[tau_range]
    
    return df

def cumulative_mean_normalized_difference(df, tau_max):
    """累积均值归一化（核心步骤2）"""
    cmndf = df[1:] * np.arange(1, tau_max) / np.cumsum(df[1:]).clip(1e-12)
    return np.insert(cmndf, 0, 1)  # 插入d'(0)=1

def parabolic_interpolation(y):
    """抛物线插值（核心步骤4）"""
    if y.size < 3: return 0
    parabola = np.polyfit(np.arange(3), y, 2)
    return -parabola[1] / (2 * parabola[0])

def yin_pitch(x, sr, frame_length=2048, hop_length=512, fmin=50, fmax=500, threshold=0.1):
    """完整YIN算法实现"""
    # 参数初始化
    min_period = max(1, int(sr / fmax))
    max_period = min(int(sr / fmin), frame_length // 2)
    
    # 分帧处理
    frames = np.lib.stride_tricks.sliding_window_view(
        x, frame_length, axis=0)[::hop_length]
    
    f0 = []
    for frame in frames:
        # 步骤1：计算差分函数
        df = difference_function(frame, frame_length, max_period)
        
        # 步骤2：累积均值归一化
        cmndf = cumulative_mean_normalized_difference(df, max_period)
        
        # 步骤3：阈值检测
        tau = np.argmin(cmndf[min_period:max_period]) + min_period
        if cmndf[tau] < threshold:
            # 步骤4：抛物线插值
            if tau+1 < len(cmndf):
                y = cmndf[tau-1:tau+2]
                tau += parabolic_interpolation(y)
            f0.append(sr / tau if tau > 0 else 0)
        else:
            f0.append(0)  # 无声段
    
    return np.array(f0)


def calculate_f0_range(y, sr, fmin=80, fmax=400, valid_probs = 0.5):
    
    # 使用PYIN算法提取基频
    f0, voiced_flag, voiced_probs = librosa.pyin(y, 
                                                sr=sr,
                                                fmin=fmin, 
                                                fmax=fmax,
                                                frame_length=2048,
                                                hop_length=512)
    
    # 过滤无效值（保留置信度>0.8的帧）
    valid_f0 = f0[(voiced_probs > valid_probs) & (voiced_flag)]
    valid_f0 = valid_f0[~np.isnan(valid_f0)]
    
    if len(valid_f0) == 0:
        raise ValueError("未检测到有效基频")
    
    # 计算统计指标
    stats = {
        "mean": np.mean(valid_f0),
        "std": np.std(valid_f0),
        "max": np.max(valid_f0),
        "min": np.min(valid_f0),
        "range": np.ptp(valid_f0),  # 峰峰值范围
        "jitter": np.mean(np.abs(np.diff(valid_f0)))  # 基频抖动
    }

    valid_times = (voiced_probs > valid_probs) & (voiced_flag)
    
    
    return f0, valid_f0, valid_times, stats

# 示例使用
if __name__ == "__main__":
    audio_file = f'recordings/clean_吃饭.wav'  # 替换为你的音频文件路径
    #audio_file = f'clean_audio.wav'  # 替换为你的音频文件路径
    

    y, sr = librosa.load(audio_file, sr=16000)
    
    f0, valid_f0, valid_times, stats = calculate_f0_range(y, sr=sr, valid_probs = 0.2)
    print("基频统计指标：")
    for k, v in stats.items():
        print(f"{k.upper():<8}: {v:.2f} Hz")

    print("Valid F0:", len(valid_f0), valid_f0)

    print("Calculate F0:", len(f0),f0)

    f02 = yin_pitch(y, sr, fmin=80, fmax=400)
    print("Estimated F0:", len(f02), f02)

    # 可视化基频轨迹
    plt.figure(figsize=(12, 6))
    times = librosa.times_like(f0, sr=sr, hop_length=512)
    plt.plot(times, f0, label='f0', alpha=0.5)
    plt.plot(times[valid_times], valid_f0, label='valid_f0')
    plt.plot(times[4:], f02, label='Estimated f0', alpha=0.25)
    plt.xlabel('time(s)')
    plt.ylabel('Hz')
    plt.title('Hz Analysis')
    plt.legend()
    plt.show()
