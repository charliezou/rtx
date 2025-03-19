import librosa
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import noisereduce as nr  # 需要单独安装的降噪库

# --------------------------
# 步骤1：加载音频并预处理
# --------------------------
def load_audio(path, sr=16000):
    """加载音频并统一采样率"""
    current_recording_path = f'recordings/喝茶.wav'
    y, _ = librosa.load(current_recording_path, sr=sr)
    return y

# --------------------------
# 步骤2：噪声抑制 (使用谱减法)
# --------------------------
def reduce_noise(y, sr=16000, noise_start=0, noise_end=0.5):
    """
    基于噪声样本的谱减法降噪
    :param noise_start: 噪声样本开始时间（秒）
    :param noise_end: 噪声样本结束时间（秒）
    """
    # 提取噪声样本
    noise_clip = y[int(noise_start*sr):int(noise_end*sr)]
    
    # 使用noisereduce库降噪
    y_clean = nr.reduce_noise(
        y=y, 
        y_noise=noise_clip,
        sr=sr,
        stationary=True  # 假设噪声是稳态的
    )
    return y_clean

# --------------------------
# 步骤3：静音检测 (基于能量阈值)
# --------------------------
def detect_silence(y, top_db=25, frame_length=2048, hop_length=512):
    """
    检测非静音区域
    :param top_db: 低于该阈值视为静音（分贝）
    :return: 有效音频段的起始和结束索引
    """
    # 计算分贝值
    rms = librosa.feature.rms(
        y=y,
        frame_length=frame_length,
        hop_length=hop_length
    )
    
    # 转换为分贝
    db = librosa.amplitude_to_db(rms, ref=np.max)
    
    # 检测非静音区域
    non_silent = librosa.effects.split(
        y,
        top_db=top_db,
        frame_length=frame_length,
        hop_length=hop_length
    )
    return non_silent

# --------------------------
# 步骤4：可视化分析
# --------------------------
def plot_audio(y, y_clean, non_silent, sr=16000):
    """绘制原始/降噪音频及静音检测结果"""
    plt.figure(figsize=(15, 8))
    
    # 原始音频波形
    plt.subplot(3, 1, 1)
    librosa.display.waveshow(y, sr=sr)
    plt.title("Original Audio")
    
    # 降噪后波形
    plt.subplot(3, 1, 2)
    librosa.display.waveshow(y_clean, sr=sr)
    plt.title("Denoised Audio")
    
    # 静音检测标记
    plt.subplot(3, 1, 3)
    librosa.display.waveshow(y_clean, sr=sr)
    for start, end in non_silent:
        start_time = start / sr
        end_time = end / sr
        plt.axvspan(start_time, end_time, color="red", alpha=0.3)
    plt.title("Voice Activity Detection")
    
    plt.tight_layout()
    plt.show()

# --------------------------
# 主程序
# --------------------------
if __name__ == "__main__":
    # 加载音频
    audio_path = "test_audio.wav"
    y = load_audio(audio_path)
    
    # 降噪处理
    y_clean = reduce_noise(y)
    
    # 静音检测
    non_silent = detect_silence(y_clean)
    
    # 提取有效音频段
    valid_audio = np.concatenate([y_clean[start:end] for start, end in non_silent])
    
    # 保存结果
    sf.write("clean_audio.wav", valid_audio, 16000)
    
    # 可视化
    plot_audio(y, y_clean, non_silent)