import numpy as np
import librosa
import soundfile as sf
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import noisereduce as nr
import torch
import torchaudio

def load_audio(path, sr=16000):
    """加载音频并统一采样率"""
    
    y, _ = librosa.load(path, sr=sr)
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

def clean_audio(y, top_db=25):
    # 降噪处理
    y_clean = reduce_noise(y)
    
    # 静音检测
    non_silent = detect_silence(y_clean, top_db=top_db)
    
    # 提取有效音频段
    valid_audio = np.concatenate([y_clean[start:end] for start, end in non_silent])

    return valid_audio


def process_recognition_audio(audio_data, xd=32768.0):
    # 复用特征提取逻辑
    y = audio_data.astype(np.float32) / xd
    mfcc = librosa.feature.mfcc(y=y, sr=16000, n_mfcc=13)

    # 添加一阶/二阶差分
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    combined_features = np.hstack([mfcc, mfcc_delta, mfcc_delta2])
    return combined_features

# --------------------------
# 步骤2：DTW相似度计算
# --------------------------
def calculate_similarity(mfcc1, mfcc2):
    """
    使用DTW计算MFCC序列的相似度
    :param mfcc1: 输入语音MFCC
    :param mfcc2: 目标语音卡MFCC
    :return: 相似度得分 (0~100)
    """
    # 计算DTW最优路径和累计距离
    distance, _ = fastdtw(mfcc1, mfcc2, dist=euclidean)
    
    # 归一化处理（可根据具体需求调整）
    max_length = max(len(mfcc1), len(mfcc2))
    normalized_score = 1 / (1 + distance/max_length)
    
    return round(normalized_score, 4)

def plot_audio(y1, y2,y3, y1_clean, y2_clean, y3_clean, similarity1_2, similarity1_3,similarity2_3, sr=16000):
    """绘制原始/降噪音频及静音检测结果"""
    plt.figure(figsize=(15, 8))
    
    plt.subplot(7, 1, 7)
    plt.title(f"Voice similarity: {similarity1_2}-----{similarity1_3}-----{similarity2_3}")
    
    # y1原始音频波形
    plt.subplot(7, 1, 1)
    librosa.display.waveshow(y1, sr=sr)
    plt.title("Original Audio y1")
    
    # y1降噪后波形
    plt.subplot(7, 1, 2)
    librosa.display.waveshow(y1_clean, sr=sr)
    plt.title("Denoised Audio y1")

    # y1原始音频波形
    plt.subplot(7, 1, 3)
    librosa.display.waveshow(y2, sr=sr)
    plt.title("Original Audio y2")
    
    # y1降噪后波形
    plt.subplot(7, 1, 4)
    librosa.display.waveshow(y2_clean, sr=sr)
    plt.title("Denoised Audio y2")
      
    # y1原始音频波形
    plt.subplot(7, 1, 5)
    librosa.display.waveshow(y3, sr=sr)
    plt.title("Original Audio y3")
    
    # y1降噪后波形
    plt.subplot(7, 1, 6)
    librosa.display.waveshow(y3_clean, sr=sr)
    plt.title("Denoised Audio y3")

    plt.tight_layout()
    plt.show()



# --------------------------
# 主程序
# --------------------------
if __name__ == "__main__":
    # 加载音频
    y1 = load_audio('recordings/睡觉.wav')
    y2 = load_audio('recordings/睡觉2.wav')
    y3 = load_audio('recordings/睡觉3.wav')

    # 降噪处理
    y1_clean = clean_audio(y1,25)
    y2_clean = clean_audio(y2,25)
    y3_clean = clean_audio(y3,25)

    # 提取MFCC特征
    mfcc1 = process_recognition_audio(y1_clean)
    mfcc2 = process_recognition_audio(y2_clean)
    mfcc3 = process_recognition_audio(y3_clean)

    # 计算相似度
    similarity1_2 = calculate_similarity(mfcc1.T, mfcc2.T)
    similarity1_3 = calculate_similarity(mfcc1.T, mfcc3.T)
    similarity2_3 = calculate_similarity(mfcc2.T, mfcc3.T)

    plot_audio(y1, y2,y3, y1_clean, y2_clean, y3_clean, similarity1_2, similarity1_3,similarity2_3)