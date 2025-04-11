
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
from scipy.signal import find_peaks, hilbert, butter, filtfilt
import parselmouth
from parselmouth.praat import call

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

    
def find_peaks_in_envelope(envelope, sr, prominence=0.1, distance=100, low_rate=0.5):
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
        width=100,
        #plateau_size=5,
        #height=0.25,
        #threshold=0.2,
        prominence=prominence,
        distance=distance
    )

    """
    检查波峰是否需要合并
    """
    while True:
        drop_peak = -1
        peak_values = envelope[peaks]
        for i in range(1, len(peaks)):
            # 计算波峰之间的 trough 高度
            trough = np.min(envelope[peaks[i-1]:peaks[i]])
            if (trough > min(peak_values[i-1], peak_values[i]) * low_rate):
                drop_peak = i if peak_values[i-1] > peak_values[i] else i-1
                break
        if drop_peak < 0:
            break
        keep = [i for i in range(len(peaks)) if i != drop_peak]
        peaks = peaks[keep]
        properties = {k: v[keep] for k, v in properties.items()}
    
    troughs = np.asarray([np.argmin(envelope[peaks[i-1]:peaks[i]]) + peaks[i-1] for i in range(1, len(peaks))])
    left_waves = np.append(np.asarray([0]), troughs+1)
    right_waves = np.append(troughs, np.asarray([len(envelope)-1]))
    properties["left_waves"] = left_waves
    properties["right_waves"] = right_waves

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

    return duration, peak_density

def time_scale_psola(audio, sr, duration, peak_density, target_peak_density=2.0):
    """使用PSOLA算法进行时间缩放(改变语速但不改变音高)"""
    snd = parselmouth.Sound(audio, sampling_frequency=sr)
    
    # 创建操作对象
    manipulation = call(snd, "To Manipulation", 0.01, 75, 600)
    
    # 创建新的持续时间层
    duration_tier = call("Create DurationTier", "duration", 0, duration)
    target_rate = peak_density / target_peak_density
    #target_rate = max(1/3, min(target_rate, 3.0))  # 限制缩放因子范围
    print(f"检测到原始语速指标: {peak_density:.2f} (目标: {target_peak_density:.2f})")
    print(f"应用时间缩放因子: {target_rate:.2f}")

    # 应用时间缩放
    call(duration_tier, "Add point", 0, target_rate)
    
    # 替换持续时间层
    call([duration_tier, manipulation], "Replace duration tier")
    
    # 重新合成语音
    scaled_sound = call(manipulation, "Get resynthesis (overlap-add)")
    scaled_audio = scaled_sound.values.T.flatten()
    
    return scaled_audio

def adaptive_gain_control(audio, target_rms = 0.1):
    """第四步：自适应增益控制"""
    current_rms = np.sqrt(np.mean(audio**2))
    
    # 计算需要的增益
    gain = target_rms / (current_rms + 1e-10)
    gain = np.clip(gain, 0.5, 4.0)  # 限制增益范围
    print(f"当前RMS: {current_rms:.4f}, 目标RMS: {target_rms:.4f}, 应用增益: {gain:.4f}")
    
    # 应用增益
    enhanced_audio = audio * gain
   
    return enhanced_audio

def main(input_file, output_file="adjusted_speech.wav"):
    """主处理函数"""
    print("开始包络线波峰检测分析...")
    # 1. 加载音频
    audio, sr = load_audio(input_file)
    print(f"已加载音频: {input_file}, 时长: {len(audio)/sr:.2f}秒")
    
    # 2. 提取包络线
    envelope = extract_envelope(audio, sr, lowpass_cutoff=10)
   
    # 3. 检测包络波峰
    peaks, properties = find_peaks_in_envelope(envelope, sr, prominence=0.2, distance=int(0.2*sr))

    print(peaks)
    print(properties)

    # 4. 分析波峰特征
    duration, peak_density = analyze_envelope_peaks(envelope, sr, peaks, properties)

    print(f"\n最终结果: 包络线中共检测到 {len(peaks)} 个显著波峰")
    
    # 5. 绘制结果
    plot_envelope_with_peaks(audio, sr, envelope, peaks)



    # 6. 智能调整语速
    print("开始智能调整语...")
    scaled_audio = time_scale_psola(audio, sr, duration, peak_density, target_peak_density=2.0)

    # 7. 自适应增益控制
    print("开始自适应增益控制...")
    enhanced_audio = adaptive_gain_control(scaled_audio, target_rms=0.15)
    
    # 7. 保存结果
    sf.write(output_file, enhanced_audio, sr)
    print(f"结果已保存到: {output_file}")


if __name__ == "__main__":
    yuyin = "上厕所99"
    input_file = f"recordings/{yuyin}.wav"  # 替换为你的音频文件
    output_file = f"recordings/adjusted13_{yuyin}.wav"
    main(input_file, output_file)
