import librosa
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import noisereduce as nr
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import soundfile as sf


def load_audio(path, sr=16000):
    """加载音频并统一采样率"""
    y, _ = librosa.load(path, sr=sr)
    return y

# --------------------------
# 步骤2：噪声抑制 (使用谱减法)
# --------------------------
def reduce_noise(y, sr=16000, noise_start=0, noise_end=0.3):
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

    # 提取噪声样本
    noise_clip = y[len(y)-int(noise_end*sr):]
    
    # 使用noisereduce库降噪
    y_clean2 = nr.reduce_noise(
        y=y_clean, 
        y_noise=noise_clip,
        sr=sr,
        stationary=True  # 假设噪声是稳态的
    )

    return y_clean2

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

    # 提取有效音频段
    valid_audio = np.concatenate([y[start:end] for start, end in non_silent])
    return valid_audio


def process_recognition_audio(y, sr=16000):
    # 降噪处理
    y_clean = reduce_noise(y)
    # 静音检测
    y_clean = detect_silence(y_clean)

    # 预加重处理
    pre_emphasis = 0.97
    #y = y_clean.astype(np.float32) / xd
    y_preemphasized = np.append(y_clean[0], y_clean[1:] - pre_emphasis * y_clean[:-1])

    # 提取MFCC特征
    mfccs = librosa.feature.mfcc(y=y_preemphasized, sr=sr, n_mfcc=13)
    # 提取pitch特征
    pitch = librosa.yin(y=y_preemphasized, fmin=50, fmax=2000).reshape(1, -1)
    # 提取energy特征
    energy = librosa.feature.rms(y=y_preemphasized)
    # 合并特征
    combined_feat = np.concatenate([mfccs, pitch, energy], axis=0)

    # 特征归一化
    mfccs_normalized = librosa.util.normalize(mfccs, axis=1)

    return y_clean, mfccs_normalized

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
   
    return round(normalized_score, 4), distance


# --------------------------
# 主程序
# --------------------------
if __name__ == "__main__":
    yuyins = ['吃饭', '吃饭3','吃水果','饮茶','饮水','饮茶2','吃水果2','吃水蛋','睡觉','睡觉2','睡觉3']
    mfccs = []

    for yuyin in yuyins:
        # 加载音频
        y = load_audio(f'recordings/{yuyin}.wav')
        y_clean, mfccs_normalized= process_recognition_audio(y) 
        mfccs.append(mfccs_normalized)
        # 保存结果
        sf.write(f'recordings/clean_{yuyin}.wav', y_clean, 16000)

    for i in range(len(mfccs)):
        # 打印归一化后的MFCC特征形状
        print(f"Normalized MFCCs shape for {yuyins[i]}:", mfccs[i].shape)
        for j in range(len(mfccs)):
            if i == j:
                continue
            similarity, distance = calculate_similarity(mfccs[i].T, mfccs[j].T)
            print(f"Similarity between {yuyins[i]} and {yuyins[j]}:", similarity, distance)

    # 设置全局字体为支持中文的字体
    matplotlib.rcParams['font.family'] = 'STHeiti'  # 黑体
    matplotlib.rcParams['font.size'] = 9
    matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号
    
    # 可视化原始MFCC特征
    plt.figure(figsize=(15, 7))
    # 可视化归一化后的MFCC特征
    for i in range(len(mfccs)):
        plt.subplot(len(mfccs), 1, i+1)
        plt.imshow(mfccs[i], cmap='viridis', origin='lower', aspect='auto')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Normalized MFCCs for {yuyins[i]}')
    plt.tight_layout()
    plt.show()
    
   