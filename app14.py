"""
参数调优建议
对于严重鼻音问题：
增加reduction_strength(0.8-0.9)
扩展鼻音频带范围(如增加150-350Hz频带)

对于轻度鼻音问题：
降低reduction_strength(0.4-0.6)
减少滤波频带数量(只处理主要鼻音频带)

针对特定患者：
可先分析其鼻音频谱特征
基于分析结果个性化设置滤波频带
"""
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
from scipy import signal
from scipy.fft import fft, ifft, fftfreq
import parselmouth
from parselmouth.praat import call

def load_audio(file_path, target_sr=16000):
    """加载音频文件"""
    audio, sr = librosa.load(file_path, sr=target_sr)
    return audio, sr

def plot_spectral_comparison(original, enhanced, sr, title=""):
    """绘制频谱对比图"""
    plt.figure(figsize=(12, 8))
    
    # 原始频谱
    plt.subplot(2, 1, 1)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(original)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Original Spectrum')
    
    # 增强后频谱
    plt.subplot(2, 1, 2)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(enhanced)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Enhanced Spectrum ' + title)
    
    plt.tight_layout()
    plt.show()

def nasal_detection(audio, sr):
    """检测鼻音区域"""
    snd = parselmouth.Sound(audio, sampling_frequency=sr)
    
    # 计算鼻音指标(基于频谱重心和带宽)
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
    
    # 鼻音区域通常有较低的频谱重心和较窄的带宽
    nasal_indicator = (spectral_centroid < np.median(spectral_centroid)) & \
                     (spectral_bandwidth < np.median(spectral_bandwidth))
    
    # 计算浊音区域(鼻音只出现在浊音段)
    pitch = snd.to_pitch()
    voiced_frames = pitch.selected_array['frequency'] > 0
    
    # 合并鼻音和浊音条件
    nasal_regions = nasal_indicator & voiced_frames[:len(nasal_indicator)]
    
    return nasal_regions

def nasal_reduction_filter(audio, sr, nasal_regions, reduction_strength=0.7):
    """
    鼻音减弱滤波器
    参数:
        reduction_strength: 减弱强度(0-1)
    """
    # 设计带阻滤波器组(针对典型鼻音频率)
    nyquist = 0.5 * sr
    nasal_bands = [
        (200, 300),   # 低频鼻音共振区
        (800, 1200),   # 中频鼻音共振区
        (2400, 2800)   # 高频鼻音共振区
    ]
    
    # 创建全局滤波器
    sos_global = []
    for low, high in nasal_bands:
        low_cut = low / nyquist
        high_cut = high / nyquist
        sos = signal.butter(4, [low_cut, high_cut], btype='bandstop', output='sos')
        sos_global.append(sos)
    
    # 应用全局滤波器
    filtered_audio = audio.copy()
    for sos in sos_global:
        filtered_audio = signal.sosfilt(sos, filtered_audio)
    
    # 创建动态调节版本(只在鼻音区域应用强滤波)
    dynamic_audio = audio.copy()
    frame_size = int(0.02 * sr)  # 20ms帧
    hop_size = frame_size // 2
    
    for i in range(0, len(audio) - frame_size, hop_size):
        frame_end = min(i + frame_size, len(audio))
        region_idx = i // hop_size
        
        if region_idx < len(nasal_regions) and nasal_regions[region_idx]:
            # 鼻音区域应用强滤波
            frame = audio[i:frame_end]
            for sos in sos_global:
                frame = signal.sosfilt(sos, frame)
            dynamic_audio[i:frame_end] = frame * reduction_strength + \
                                        audio[i:frame_end] * (1 - reduction_strength)
    
    return dynamic_audio

def formant_enhancement(audio, sr, nasal_regions, boost_strength=1.5):
    """增强非鼻音区域的共振峰"""
    snd = parselmouth.Sound(audio, sampling_frequency=sr)
    enhanced_audio = np.zeros_like(audio)
    frame_size = int(0.1 * sr)  # 20ms帧
    hop_size = frame_size // 2
    
    for i in range(0, len(audio) - frame_size, hop_size):
        frame_end = min(i + frame_size, len(audio))
        region_idx = i // hop_size
        frame = audio[i:frame_end]
        
        if region_idx >= len(nasal_regions) or not nasal_regions[region_idx]:
            # 非鼻音区域增强共振峰
            frame_snd = parselmouth.Sound(frame, sampling_frequency=sr)
            formant = frame_snd.to_formant_burg(max_number_of_formants=5)
            
            # 获取并增强主要共振峰
            manipulation = call(frame_snd, "To Manipulation", 0.01, 75, 600)
            formant_tier = call(manipulation, "Extract formant tier")
            
            mid_time = 0.5 * len(frame) / sr
            for formant_num in [1, 2, 3]:  # 前三个共振峰
                freq = call(formant, "Get value at time", formant_num, mid_time, "HERTZ", "LINEAR")
                if not np.isnan(freq):
                    new_freq = freq * boost_strength
                    call(formant_tier, "Add point", mid_time, new_freq)
            
            call([formant_tier, manipulation], "Replace formant tier")
            enhanced_frame = call(manipulation, "Get resynthesis (overlap-add)")
            frame = enhanced_frame.values.T.flatten()
        
        enhanced_audio[i:frame_end] += frame * 0.5  # 重叠相加
    
    return enhanced_audio

def evaluate_enhancement(original, enhanced, sr):
    """评估增强效果"""
    # 计算鼻音区域的能量变化
    nasal_regions = nasal_detection(original, sr)
    original_nasal_energy = np.mean(original[nasal_regions[:len(original)]]**2)
    enhanced_nasal_energy = np.mean(enhanced[nasal_regions[:len(enhanced)]]**2)
    nasal_reduction = (original_nasal_energy - enhanced_nasal_energy) / original_nasal_energy * 100
    
    # 计算整体语音质量指标
    def calculate_hnr(audio):
        snd = parselmouth.Sound(audio, sampling_frequency=sr)
        pitch = snd.to_pitch()
        return call(pitch, "Get mean", "HNR", 0, 0)
    
    hnr_improvement = calculate_hnr(enhanced) - calculate_hnr(original)
    
    print("\n增强效果评估:")
    print(f"鼻音能量降低: {nasal_reduction:.1f}%")
    print(f"谐噪比(HNR)提升: {hnr_improvement:.1f} dB")
    
    # 绘制波形对比
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(original, sr=sr, alpha=0.7, label='Original')
    plt.title("Original Audio")
    plt.subplot(2, 1, 2)
    librosa.display.waveshow(enhanced, sr=sr, alpha=0.7, color='r', label='Enhanced')
    plt.title("Enhanced Audio")
    plt.tight_layout()
    plt.show()
    
    return nasal_reduction, hnr_improvement

def main(input_file, output_file):
    """主处理流程"""
    # 1. 加载音频
    original_audio, sr = load_audio(input_file)
    print(f"已加载音频: {input_file}, 时长: {len(original_audio)/sr:.2f}秒")
    
    # 2. 检测鼻音区域
    nasal_regions = nasal_detection(original_audio, sr)
    print(f"检测到鼻音区域占比: {np.mean(nasal_regions)*100:.1f}%")
    
    # 3. 鼻音减弱处理
    reduced_nasal_audio = nasal_reduction_filter(original_audio, sr, nasal_regions)
    
    # 4. 共振峰增强
    enhanced_audio = formant_enhancement(reduced_nasal_audio, sr, nasal_regions)
    
    # 5. 保存结果
    sf.write(output_file, enhanced_audio, sr)
    
    # 6. 评估与可视化
    plot_spectral_comparison(original_audio, enhanced_audio, sr, "Nasal Reduction")
    nasal_reduction, hnr_improvement = evaluate_enhancement(original_audio, enhanced_audio, sr)
    
    return enhanced_audio, nasal_reduction, hnr_improvement

if __name__ == "__main__":
    yuyin = "上厕所99"
    input_file = f"recordings/{yuyin}.wav"  # 替换为你的音频文件
    output_file = f"recordings/enhanced14_{yuyin}.wav"
    
    print("开始处理鼻音过重问题...")
    enhanced_audio, nasal_reduction, hnr_improvement = main(input_file, output_file)
    
    print("\n处理完成!")
    print(f"结果已保存到: {output_file}")
    print(f"鼻音能量降低: {nasal_reduction:.1f}%")
    print(f"谐噪比提升: {hnr_improvement:.1f} dB")