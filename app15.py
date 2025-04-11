import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
from scipy import signal
from scipy.fft import fft, ifft, fftfreq

def load_audio(file_path, target_sr=16000):
    """加载音频文件"""
    audio, sr = librosa.load(file_path, sr=target_sr)
    return audio, sr

def lpc_analysis(audio, sr, order=14):
    """线性预测编码(LPC)分析"""
    # 预加重
    pre_emphasis = 0.97
    emphasized = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])
    
    # 分帧处理
    frame_length = int(0.025 * sr)  # 25ms帧
    hop_length = int(0.01 * sr)     # 10ms跳幅
    frames = librosa.util.frame(emphasized, frame_length=frame_length, hop_length=hop_length).T
    
    # 加窗
    windows = np.hamming(frame_length)
    frames = frames * windows
    
    # 计算LPC系数
    lpc_coeffs = []
    for frame in frames:
        # 自相关
        autocorr = np.correlate(frame, frame, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Levinson-Durbin递归
        coeffs = np.zeros(order+1)
        reflection = np.zeros(order)
        error = autocorr[0]
        
        for i in range(1, order+1):
            reflection[i-1] = autocorr[i]
            for j in range(1, i):
                reflection[i-1] -= coeffs[j] * autocorr[i-j]
            reflection[i-1] /= error
            
            coeffs[i] = reflection[i-1]
            for j in range(1, i):
                temp = coeffs[j]
                coeffs[j] -= reflection[i-1] * coeffs[i-j]
            error *= (1 - reflection[i-1]**2)
        
        coeffs[0] = 1.0
        lpc_coeffs.append(coeffs)
    
    return np.array(lpc_coeffs), frame_length, hop_length

def get_first_values(arr,lenght=3):
    # 检查数组长度
    if len(arr) < lenght:
        # 使用numpy.pad补全数组，补全的值为np.nan
        arr = np.pad(arr, (0, lenght - len(arr)), 'constant', constant_values=0)
    # 取前lenght个值
    return arr[:lenght]

def get_formants_from_lpc(lpc_coeffs, sr):
    """从LPC系数计算共振峰"""
    formants = []
    for coeffs in lpc_coeffs:
        # 求LPC多项式的根
        roots = np.roots(coeffs)
        roots = roots[np.imag(roots) >= 0]  # 保留上半平面
        
        # 计算角度得到频率
        angz = np.arctan2(np.imag(roots), np.real(roots))
        freqs = angz * (sr / (2 * np.pi))
        
        # 按频率排序并选择前几个
        freqs = np.sort(freqs)
        valid_freqs = get_first_values(freqs[(freqs > 50) & (freqs < 4000)],3)  # 取前3个有效共振峰
        formants.append(valid_freqs)
    
    return np.array(formants)

def plot_formant_contours(original_formants, enhanced_formants, sr, hop_length):
    """绘制共振峰轨迹对比图"""
    plt.figure(figsize=(12, 6))
    
    # 时间轴
    times = np.arange(len(original_formants)) * hop_length / sr
    
    # 原始共振峰
    plt.subplot(2, 1, 1)
    for i in range(3):
        plt.plot(times, original_formants[:, i], label=f'F{i+1}')
    plt.title('Original Formants')
    plt.ylabel('Frequency (Hz)')
    plt.legend()
    plt.grid()
    
    # 增强后共振峰
    plt.subplot(2, 1, 2)
    for i in range(3):
        plt.plot(times, enhanced_formants[:, i], label=f'F{i+1}')
    plt.title('Enhanced Formants')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.legend()
    plt.grid()
    
    plt.tight_layout()
    plt.show()

def overlap_add(frames, hop_length):
    """手动实现重叠相加算法"""
    frame_length = frames.shape[1]
    output_length = frame_length + hop_length * (frames.shape[0] - 1)
    output = np.zeros(output_length)
    
    for i, frame in enumerate(frames):
        start = i * hop_length
        end = start + frame_length
        output[start:end] += frame
    
    return output

def formant_enhancement(audio, sr, boost_factor=1.2):
    """共振峰增强处理"""
    # 1. LPC分析获取原始共振峰
    lpc_coeffs, frame_length, hop_length = lpc_analysis(audio, sr)
    original_formants = get_formants_from_lpc(lpc_coeffs, sr)
    
    # 2. 预加重滤波器
    pre_emphasis = 0.97
    emphasized = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])
    
    # 3. 分帧处理
    frames = librosa.util.frame(emphasized, frame_length=frame_length, hop_length=hop_length).T
    windows = np.hamming(frame_length)
    enhanced_frames = np.zeros_like(frames)
    
    # 4. 对每帧进行共振峰增强
    for i, (frame, coeffs) in enumerate(zip(frames, lpc_coeffs)):
        # 计算LPC频谱
        w, h = signal.freqz(1, coeffs, worN=512)
        lpc_spectrum = np.abs(h)
        
        # 找到峰值(共振峰)
        peaks, _ = signal.find_peaks(lpc_spectrum, height=0.1)
        peak_freqs = w[peaks] * sr / (2 * np.pi)
        peak_mags = lpc_spectrum[peaks]
        
        # 增强前3个共振峰
        frame_fft = fft(frame * windows)
        for j in range(min(3, len(peaks))):
            # 在频域增强
            center_bin = int(peak_freqs[j] * 512 / (sr/2))
            bandwidth = 50  # Hz
            start_bin = max(0, center_bin - int(bandwidth * 512 / sr))
            end_bin = min(511, center_bin + int(bandwidth * 512 / sr))
            
            # 增强选定频段
            frame_fft[start_bin:end_bin+1] *= boost_factor
            frame_fft[512-end_bin:512-start_bin+1] *= boost_factor  # 对称部分
            
        # 反变换
        enhanced_frame = np.real(ifft(frame_fft))
        enhanced_frames[i] = enhanced_frame * windows
    
    # 5. 手动实现重叠相加
    enhanced_audio = overlap_add(enhanced_frames, hop_length)
    
    # 6. 裁剪到原始长度
    enhanced_audio = enhanced_audio[:len(emphasized)]
    
    # 7. 去加重
    enhanced_audio = signal.lfilter([1], [1, -pre_emphasis], enhanced_audio)
    
    # 8. 获取增强后的共振峰
    enhanced_lpc, _, _ = lpc_analysis(enhanced_audio, sr)
    enhanced_formants = get_formants_from_lpc(enhanced_lpc, sr)
    
    return enhanced_audio, original_formants, enhanced_formants, hop_length

def evaluate_formant_changes(original_formants, enhanced_formants):
    """评估共振峰变化"""
    print("\n共振峰变化统计:")
    for i in range(3):
        orig_f = original_formants[:, i]
        enh_f = enhanced_formants[:, i]
        
        # 计算有效变化(忽略零值)
        mask = (orig_f > 0) & (enh_f > 0)
        if np.sum(mask) > 0:
            mean_orig = np.mean(orig_f[mask])
            mean_enh = np.mean(enh_f[mask])
            change = (mean_enh - mean_orig) / mean_orig * 100
            
            print(f"F{i+1}: 平均频率 {mean_orig:.1f}Hz → {mean_enh:.1f}Hz ({change:+.1f}%)")
        else:
            print(f"F{i+1}: 无有效数据")

def main(input_file, output_file):
    """主处理流程"""
    # 1. 加载音频
    original_audio, sr = load_audio(input_file)
    print(f"已加载音频: {input_file}, 时长: {len(original_audio)/sr:.2f}秒")
    
    # 2. 共振峰增强
    enhanced_audio, orig_formants, enh_formants, hop_length = formant_enhancement(original_audio, sr, 4.0)
    
    # 3. 保存结果
    sf.write(output_file, enhanced_audio, sr)
    
    # 4. 分析与可视化
    plot_formant_contours(orig_formants, enh_formants, sr, hop_length)
    evaluate_formant_changes(orig_formants, enh_formants)



    
    # 5. 频谱对比
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(original_audio)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Original Spectrum')
    
    plt.subplot(2, 1, 2)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(enhanced_audio)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Enhanced Spectrum')
    
    plt.tight_layout()
    plt.show()
    
    return enhanced_audio


if __name__ == "__main__":
    yuyin = "上厕所99"
    input_file = f"recordings/{yuyin}.wav"  # 替换为你的音频文件
    output_file = f"recordings/enhanced15_{yuyin}.wav"
    
    print("开始共振峰增强处理...")
    enhanced_audio = main(input_file, output_file)
    print(f"\n处理完成! 结果已保存到 {output_file}")