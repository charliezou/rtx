import numpy as np
import librosa
import soundfile as sf
import parselmouth
from parselmouth.praat import call
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def load_audio(file_path, target_sr=16000):
    """加载音频文件并统一采样率"""
    audio, sr = librosa.load(file_path, sr=target_sr)
    return audio, sr

def plot_waveform(audio, title, color='b'):
    """绘制波形图"""
    plt.figure(figsize=(10, 3))
    librosa.display.waveshow(audio, sr=16000, color=color)
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.show()

def analyze_speech_rate(audio, sr=16000):
    """分析语音速率特征"""
    snd = parselmouth.Sound(audio, sampling_frequency=sr)
    
    # 计算基频(判断浊音段)
    pitch = snd.to_pitch(time_step=0.01) #步长0.01s
    f0 = pitch.selected_array['frequency']

    print(f0.shape)
    print(f0)
    
    # 计算强度(能量)
    intensity = snd.to_intensity()
    intensity_values = intensity.values[0]

    
    # 检测语音活动
    voiced_frames = np.where(f0 > 0)[0]
    voiced_ratio = len(voiced_frames) / len(f0) if len(f0) > 0 else 0
    
    # 计算平均语速指标
    speaking_rate = voiced_ratio * (sr / len(audio)) * 100  # 自定义指标
    
    return {
        'voiced_ratio': voiced_ratio,
        'speaking_rate': speaking_rate,
        'intensity_profile': intensity_values,
        'f0_profile': f0
    }

def time_scale_psola(audio, sr, rate_factor):
    """使用PSOLA算法进行时间缩放(改变语速但不改变音高)"""
    snd = parselmouth.Sound(audio, sampling_frequency=sr)
    
    # 创建操作对象
    manipulation = call(snd, "To Manipulation", 0.01, 75, 600)
    
    # 获取原始音标层
    #original_duration = call(manipulation, "Get total duration")
    original_duration = len(audio)/sr
    
    # 创建新的持续时间层
    duration_tier = call("Create DurationTier", "duration", 0, original_duration)
    call(duration_tier, "Add point", 0, 1/3)
    
    # 替换持续时间层
    call([duration_tier, manipulation], "Replace duration tier")
    
    # 重新合成语音
    scaled_sound = call(manipulation, "Get resynthesis (overlap-add)")
    scaled_audio = scaled_sound.values.T.flatten()
    
    return scaled_audio

def dynamic_time_scaling(audio, sr, target_rate=3.0, max_rate=1.5):
    """动态时间缩放 - 智能调整语速"""
    # 分析原始语音特征
    analysis = analyze_speech_rate(audio, sr)
    #print("原始语音分析结果:")
    #print(analysis)


    original_rate = analysis['speaking_rate']
    
    print(f"检测到原始语速指标: {original_rate:.2f} (目标: {target_rate:.2f})")
    
    # 计算需要的速率调整因子
    rate_factor = min(original_rate / target_rate, max_rate)
    
    # 如果原始语速已经接近目标，不做调整
    if abs(original_rate - target_rate) < 0.2:
        print("语速接近正常范围，无需调整")
        return audio
    
    print(f"应用时间缩放因子: {rate_factor:.2f}")
    
    # 应用PSOLA时间缩放
    scaled_audio = time_scale_psola(audio, sr, rate_factor)
    
    return scaled_audio

def evaluate_speech_rate(audio_before, audio_after, sr=16000):
    """评估语速调整效果"""
    before = analyze_speech_rate(audio_before, sr)
    after = analyze_speech_rate(audio_after, sr)
    
    improvement = (after['speaking_rate'] - before['speaking_rate']) / before['speaking_rate'] * 100
    
    print("\n语速调整评估结果:")
    print(f"原始语速指标: {before['speaking_rate']:.2f}")
    print(f"调整后语速指标: {after['speaking_rate']:.2f}")
    print(f"语速提升: {improvement:.1f}%")
    
    # 绘制对比图
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(audio_before, sr=sr, color='b')
    plt.title(f"原始语音 (语速指标: {before['speaking_rate']:.2f})")
    
    plt.subplot(2, 1, 2)
    librosa.display.waveshow(audio_after, sr=sr, color='r')
    plt.title(f"调整后语音 (语速指标: {after['speaking_rate']:.2f})")
    
    plt.tight_layout()
    plt.show()
    
    return improvement

def main(input_file, output_file):
    """主处理函数"""
    # 1. 加载音频
    audio, sr = load_audio(input_file)
    print(f"加载音频: {input_file}, 时长: {len(audio)/sr:.2f}秒")
    
    # 2. 显示原始波形
    #plot_waveform(audio, "原始语音波形")
    
    # 3. 智能调整语速
    scaled_audio = dynamic_time_scaling(audio, sr, target_rate=3.0)
    
    # 4. 保存结果
    sf.write(output_file, scaled_audio, sr)
    print(f"结果已保存到: {output_file}")
    
    # 5. 评估效果
    #improvement = evaluate_speech_rate(audio, scaled_audio, sr)
    
    return 1

if __name__ == "__main__":
    input_file = "recordings/clean_睡觉3.wav"  # 替换为你的输入文件
    output_file = "adjusted_speech.wav"
    
    print("开始处理构音障碍患者语速调整...")
    improvement = main(input_file, output_file)
    
    if improvement > 0:
        print(f"成功提升语速 {improvement:.1f}%")
    else:
        print("语速已在正常范围内，无需显著调整")

        