
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
from scipy.signal import find_peaks, hilbert, butter, filtfilt

def load_audio(file_path, target_sr=16000):
    """加载音频文件"""
    audio, sr = librosa.load(file_path, sr=target_sr)
    return audio, sr

def extract_envelope(audio, sr, lowpass_cutoff=20):
    """
    使用Hilbert变换提取音频包络线
    参数:
        audio: 音频信号
        sr: 采样率
        lowpass_cutoff: 包络低通滤波截止频率(Hz)
    返回:
        envelope: 包络线信号
    """
    # 应用Hilbert变换获取解析信号
    analytic_signal = hilbert(audio)
    amplitude_envelope = np.abs(analytic_signal)
    
    # 设计低通滤波器平滑包络
    nyquist = 0.5 * sr
    cutoff = lowpass_cutoff / nyquist
    b, a = butter(4, cutoff, btype='low')
    
    # 应用零相位滤波
    smooth_envelope = filtfilt(b, a, amplitude_envelope)
    
    # 归一化包络
    normalized_envelope = smooth_envelope / np.max(smooth_envelope)
    
    return normalized_envelope

def find_peaks_in_envelope(envelope, sr, prominence=0.1, distance=100):
    """
    检测包络线中的波峰
    参数:
        envelope: 包络线信号
        sr: 采样率
        prominence: 波峰显著性阈值(0-1)
        distance: 波峰间最小距离(样本数)
    返回:
        peaks: 波峰位置索引数组
        properties: 波峰属性字典
    """
    peaks, properties = find_peaks(
        envelope,
        prominence=prominence,
        distance=distance
    )
    
    return peaks, properties

def plot_envelope_with_peaks(audio, sr, envelope, peaks, title="Envelope with Peaks"):
    """绘制带波峰标记的包络线图"""
    plt.figure(figsize=(12, 6))

    audio = audio / np.max(np.abs(audio))  # 归一化音频信号
    
    # 绘制原始波形(半透明)
    librosa.display.waveshow(audio, sr=sr, alpha=0.7, label='Original Waveform')
    
    
    # 绘制包络线
    times = librosa.times_like(audio, sr=sr, hop_length=1)

    plt.plot(times, envelope, color='b', linewidth=1.5, label='Envelope')
    
    # 标记波峰
    peak_times = times[peaks]
    peak_values = envelope[peaks]
    plt.scatter(peak_times, peak_values, color='r', s=40, label='Peaks')
    
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized Amplitude')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.show()

def analyze_envelope_peaks(envelope, sr, peaks, properties):
    """分析包络线波峰特征"""
    print(f"\n检测到 {len(peaks)} 个包络波峰")
    print("波峰统计信息:")
    
    # 计算波峰间隔(秒)
    peak_intervals = np.diff(peaks) / sr
    if len(peak_intervals) > 0:
        print(f"- 平均间隔: {np.mean(peak_intervals):.3f} ± {np.std(peak_intervals):.3f} 秒")
        print(f"- 最小间隔: {np.min(peak_intervals):.3f} 秒")
        print(f"- 最大间隔: {np.max(peak_intervals):.3f} 秒")
    
    # 波峰显著性
    print(f"- 平均显著性: {np.mean(properties['prominences']):.3f}")
    
    # 计算波峰密度(每秒波峰数)
    duration = len(envelope) / sr
    peak_density = len(peaks) / duration
    print(f"- 波峰密度: {peak_density:.2f} peaks/sec")

def main(input_file):
    """主处理函数"""
    # 1. 加载音频
    audio, sr = load_audio(input_file)
    print(f"已加载音频: {input_file}, 时长: {len(audio)/sr:.2f}秒")
    
    # 2. 提取包络线
    envelope = extract_envelope(audio, sr, lowpass_cutoff=10)

    
    # 3. 检测包络波峰
    peaks, properties = find_peaks_in_envelope(envelope, sr, prominence=0.3, distance=int(0.2*sr))

    print(peaks)
    print(properties)
    
    # 4. 绘制结果
    plot_envelope_with_peaks(audio, sr, envelope, peaks)
    
    # 5. 分析波峰特征
    analyze_envelope_peaks(envelope, sr, peaks, properties)
    
    return len(peaks)

if __name__ == "__main__":
    input_file = "recordings/吃水果.wav"  # 替换为你的音频文件
    
    print("开始包络线波峰检测分析...")
    peak_count = main(input_file)
    print(f"\n最终结果: 包络线中共检测到 {peak_count} 个显著波峰")