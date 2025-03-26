import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import noisereduce as nr
import librosa
import librosa.display
import pywt
from scipy import signal
from scipy import interpolate
from scipy.io import wavfile
from pypesq import pesq
from pystoi import stoi
import parselmouth
from parselmouth.praat import call

# 全局参数配置
DISEASE_TYPE = "dysarthria"  # 可选项: "dysarthria", "apraxia", "spasmodic"
SAMPLE_RATE = 16000
FRAME_LENGTH = 512
HOP_LENGTH = 128
THRESHOLD_DB = -40  # 静音检测阈值

def load_audio(file_path):
    """加载音频文件"""
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    return audio, sr

def plot_waveform(audio, title, subplot_pos, color='b'):
    """绘制波形图"""
    plt.subplot(4, 2, subplot_pos)
    librosa.display.waveshow(audio, sr=SAMPLE_RATE, color=color)
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')

def step1_noise_reduction(audio):
    """第一步：噪声抑制(谱减法)"""
    # 使用noisereduce进行噪声抑制
    reduced_noise = nr.reduce_noise(y=audio, sr=SAMPLE_RATE, stationary=True, prop_decrease=0.9)
    return reduced_noise

def step2_silence_detection(audio):
    """第二步：基于能量阈值的静音检测"""
    # 计算短时能量
    frames = librosa.util.frame(audio, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
    energy = np.sum(frames**2, axis=0) / FRAME_LENGTH
    
    # 转换为dB
    energy_db = 10 * np.log10(energy + 1e-10)
    
    # 检测非静音段
    non_silence = energy_db > THRESHOLD_DB
    
    # 找到非静音段的起始和结束点
    changes = np.diff(non_silence.astype(int))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    
    # 处理边界情况
    if len(ends) == 0 or (len(starts) > 0 and starts[0] > ends[0]):
        starts = np.insert(starts, 0, 0)
    if len(starts) == 0 or (len(ends) > 0 and ends[-1] < starts[-1]):
        ends = np.append(ends, len(non_silence)-1)
    
    # 合并相邻的有效段
    valid_segments = []
    for start, end in zip(starts, ends):
        if end - start > 2:  # 至少3帧才认为是有效语音
            valid_segments.append((start, end))
    
    # 如果没有检测到有效段，返回整个音频
    if not valid_segments:
        return audio
    
    # 提取最长的有效段
    longest_segment = max(valid_segments, key=lambda x: x[1]-x[0])
    start_sample = longest_segment[0] * HOP_LENGTH
    end_sample = longest_segment[1] * HOP_LENGTH
    
    return audio[start_sample:end_sample]

def step3_abnormality_detection(audio):
    """第三步：异常检测并自动配置参数"""
    # 计算基频特征
    snd = parselmouth.Sound(audio, sampling_frequency=SAMPLE_RATE)
    pitch = snd.to_pitch()
    f0 = pitch.selected_array['frequency']
    f0[f0 == 0] = np.nan
    mean_f0 = np.nanmean(f0)
    std_f0 = np.nanstd(f0)
    
    # 计算能量特征
    rms = librosa.feature.rms(y=audio, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    mean_rms = np.mean(rms)
    std_rms = np.std(rms)
    
    # 计算谐噪比(HNR)
    harmonicity = snd.to_harmonicity()
    hnr = harmonicity.values[harmonicity.values != -200].mean()
    
    # 根据疾病类型和异常检测结果配置参数
    params = {
        'agc_gain': 1.0,
        'f0_smoothing': 1.0,
        'formant_shift': 1.0
    }
    
    if DISEASE_TYPE == "dysarthria":
        # 构音障碍通常表现为基频不稳定和能量低
        if std_f0 > 20:
            params['f0_smoothing'] = 2.0
        if mean_rms < 0.05:
            params['agc_gain'] = 1.5
        if hnr < 10:
            params['formant_shift'] = 0.9  # 降低共振峰频率
    
    elif DISEASE_TYPE == "apraxia":
        # 言语失用症通常表现为不规则的发音间隔
        if std_rms / mean_rms > 1.5:
            params['agc_gain'] = 2.0
            params['f0_smoothing'] = 3.0
    
    elif DISEASE_TYPE == "spasmodic":
        # 痉挛性发音障碍通常表现为基频和能量突变
        if std_f0 > 30:
            params['f0_smoothing'] = 3.0
        if std_rms > 0.1:
            params['agc_gain'] = 1.8
    
    print(f"异常检测结果 - 平均基频: {mean_f0:.1f}Hz, 基频标准差: {std_f0:.1f}")
    print(f"平均能量: {mean_rms:.3f}, 能量标准差: {std_rms:.3f}, 谐噪比: {hnr:.1f}dB")
    print(f"自动配置参数: {params}")
    
    return params

def step4_adaptive_gain_control(audio, params):
    """第四步：自适应增益控制"""
    target_rms = 0.1  # 目标RMS值
    current_rms = np.sqrt(np.mean(audio**2))
    
    # 计算需要的增益
    gain = params['agc_gain'] * (target_rms / (current_rms + 1e-10))
    gain = np.clip(gain, 0.5, 3.0)  # 限制增益范围
    
    # 应用增益
    enhanced_audio = audio * gain
    
    # 防止削波
    max_sample = np.max(np.abs(enhanced_audio))
    if max_sample > 0.99:
        enhanced_audio = enhanced_audio * 0.99 / max_sample
    
    return enhanced_audio

def step5_pitch_correction(audio, params):
    """第五步：基频校正"""
    # 使用PSOLA算法进行基频平滑
    snd = parselmouth.Sound(audio, sampling_frequency=SAMPLE_RATE)
    
    # 获取基频轮廓
    pitch = snd.to_pitch()
    original_f0 = pitch.selected_array['frequency']
    
    # 平滑基频轮廓
    smoothed_f0 = np.array(original_f0, dtype=float)
    window_size = int(0.1 * SAMPLE_RATE / HOP_LENGTH)  # 100ms窗口
    
    for i in range(len(smoothed_f0)):
        if original_f0[i] == 0:
            continue
            
        start = max(0, i - window_size)
        end = min(len(smoothed_f0), i + window_size + 1)
        neighborhood = original_f0[start:end]
        neighborhood = neighborhood[neighborhood != 0]
        
        if len(neighborhood) > 0:
            smoothed_f0[i] = np.median(neighborhood) * params['f0_smoothing']
        else:
            smoothed_f0[i] = original_f0[i]
    
    # 应用PSOLA修改基频
    manipulation = call(snd, "To Manipulation", 0.01, 75, 600)
    pitch_tier = call(manipulation, "Extract pitch tier")
    
    # 更新基频
    for i in range(len(smoothed_f0)):
        if smoothed_f0[i] > 0:
            time = i * HOP_LENGTH / SAMPLE_RATE
            call(pitch_tier, "Add point", time, smoothed_f0[i])
    
    call([pitch_tier, manipulation], "Replace pitch tier")
    corrected_sound = call(manipulation, "Get resynthesis (overlap-add)")
    corrected_audio = corrected_sound.values.T.flatten()
    
    return corrected_audio, original_f0, smoothed_f0

def step6_formant_correction(audio, params):
    """第六步：共振峰校正（使用librosa实现）"""
    # 计算STFT
    stft = librosa.stft(audio, n_fft=2048, hop_length=HOP_LENGTH)
    
    # 获取幅度和相位谱
    magnitude, phase = librosa.magphase(stft)
    
    # 计算频谱包络
    formant_envelope = librosa.amplitude_to_db(magnitude, ref=np.max)
    
    # 计算共振峰频率
    formant_freqs = librosa.fft_frequencies(sr=SAMPLE_RATE, n_fft=2048)
    
    # 找到前三个共振峰
    peaks = signal.find_peaks(formant_envelope.mean(axis=1))[0]
    if len(peaks) >= 3:
        f1_idx, f2_idx, f3_idx = peaks[:3]
        f1, f2, f3 = formant_freqs[f1_idx], formant_freqs[f2_idx], formant_freqs[f3_idx]
    else:
        # 如果没有检测到足够的共振峰，使用默认值
        f1, f2, f3 = 500, 1500, 2500
    
    # 根据参数调整共振峰
    shift_factor = params['formant_shift']
    new_f1 = f1 * shift_factor
    new_f2 = f2 * shift_factor
    new_f3 = f3 * shift_factor
    
    # 创建新的频谱包络
    new_envelope = np.zeros_like(formant_envelope)
    
    # 应用共振峰偏移
    for i in range(formant_envelope.shape[1]):
        # 获取当前帧的频谱
        frame = formant_envelope[:, i]
        
        # 创建插值函数
        interp_func = interpolate.interp1d(formant_freqs, frame, kind='cubic', fill_value="extrapolate")
        
        # 计算新的频率轴
        new_freqs = formant_freqs * shift_factor
        
        # 应用插值
        new_frame = interp_func(new_freqs)
        
        # 存储结果
        new_envelope[:, i] = new_frame
    
    # 转换回线性幅度
    new_magnitude = librosa.db_to_amplitude(new_envelope)
    
    # 重建STFT
    new_stft = new_magnitude * phase
    
    # 使用ISTFT重建音频
    enhanced_audio = librosa.istft(new_stft, hop_length=HOP_LENGTH)
    
    return enhanced_audio

def step6_formant_correction_3(audio, params):
    """第六步：共振峰校正"""
    snd = parselmouth.Sound(audio, sampling_frequency=SAMPLE_RATE)
    
    # 获取共振峰信息
    formant = snd.to_formant_burg(max_number_of_formants=5)
    
    # 创建新的Sound对象用于存储修改后的音频
    modified_snd = parselmouth.Sound(snd)
    
    # 逐帧处理
    n_frames = call(formant, "Get number of frames")
    for i in range(1, n_frames + 1):
        time = call(formant, "Get time from frame number", i)
        
        # 获取前三个共振峰
        f1 = formant.get_value_at_time(1, time)
        f2 = formant.get_value_at_time(2, time)
        f3 = formant.get_value_at_time(3, time)
        
        if np.isnan(f1) or np.isnan(f2) or np.isnan(f3):
            continue
            
        # 根据参数调整共振峰
        new_f1 = f1 * params['formant_shift']
        new_f2 = f2 * params['formant_shift']
        new_f3 = f3 * params['formant_shift']
        
        # 使用LPC方法修改共振峰
        lpc = call(snd, "To LPC (burg)", 16, 0.025,0.01,50)  
        #lpc = call(lpc, "Shift frequencies", time, new_f1, new_f2, new_f3, "HERTZ")
        modified_snd = call(lpc, "To Sound (filter)")
    
    # 获取修改后的音频数据
    enhanced_audio = modified_snd.values.T.flatten()
    
    return enhanced_audio

def step6_formant_correction_2(audio, params):
    """第六步：共振峰校正"""
    snd = parselmouth.Sound(audio, sampling_frequency=SAMPLE_RATE)
    
    # 获取共振峰信息
    formant = snd.to_formant_burg(max_number_of_formants=5)
    
    # 创建操作对象
    manipulation = call(snd, "To Manipulation", 0.01, 75, 600)
    
    # 获取并修改共振峰层
    formant_tier = call(manipulation, "Extract formant tier")
    
    # 调整共振峰频率
    n_frames = call(formant, "Get number of frames")
    for i in range(1, n_frames + 1):
        time = call(formant, "Get time from frame number", i)
        
        # 获取前三个共振峰
        f1 = call(formant, "Get value at time", 1, time, "HERTZ", "LINEAR")
        f2 = call(formant, "Get value at time", 2, time, "HERTZ", "LINEAR")
        f3 = call(formant, "Get value at time", 3, time, "HERTZ", "LINEAR")
        
        if np.isnan(f1) or np.isnan(f2) or np.isnan(f3):
            continue
            
        # 根据参数调整共振峰
        new_f1 = f1 * params['formant_shift']
        new_f2 = f2 * params['formant_shift']
        new_f3 = f3 * params['formant_shift']
        
        # 更新共振峰层
        call(formant_tier, "Add point", time, new_f1, new_f2, new_f3)
    
    # 替换共振峰层并重新合成
    call([formant_tier, manipulation], "Replace formant tier")
    enhanced_sound = call(manipulation, "Get resynthesis (overlap-add)")
    enhanced_audio = enhanced_sound.values.T.flatten()
    
    return enhanced_audio


def evaluate_enhancement(original, enhanced):
    """评估语音增强质量"""
    # 确保长度相同
    print("111111111111")
    min_len = min(len(original), len(enhanced))
    original = original[:min_len]
    enhanced = enhanced[:min_len]
    
    # 计算信噪比提升
    noise = original - enhanced
    original_snr = 10 * np.log10(np.sum(original**2) / np.sum(noise**2 + 1e-10))
    enhanced_snr = 10 * np.log10(np.sum(enhanced**2) / np.sum(noise**2 + 1e-10))
    snr_improvement = enhanced_snr - original_snr

    print(f"SNR提升: {snr_improvement:.2f} dB")




    # 计算STOI (语音可懂度评估)
    stoi_score = stoi(original, enhanced, SAMPLE_RATE, extended=False)
    print(f"STOI: {stoi_score:.2f}")    
    # 计算基频稳定性
    snd_orig = parselmouth.Sound(original, sampling_frequency=SAMPLE_RATE)
    pitch_orig = snd_orig.to_pitch()
    f0_orig = pitch_orig.selected_array['frequency']
    f0_orig = f0_orig[f0_orig != 0]
    
    snd_enh = parselmouth.Sound(enhanced, sampling_frequency=SAMPLE_RATE)
    pitch_enh = snd_enh.to_pitch()
    f0_enh = pitch_enh.selected_array['frequency']
    f0_enh = f0_enh[f0_enh != 0]
    
    if len(f0_orig) > 1 and len(f0_enh) > 1:
        f0_std_orig = np.std(f0_orig)
        f0_std_enh = np.std(f0_enh)
        f0_stability_improvement = (f0_std_orig - f0_std_enh) / f0_std_orig * 100
    else:
        f0_stability_improvement = 0

    print(f"基频稳定性提升: {f0_stability_improvement:.2f}%")
        
    # 计算PESQ (语音质量感知评估)
    pesq_score = pesq(original, enhanced, SAMPLE_RATE)
    print(f"PESQ: {pesq_score:.2f}")
    
    return {
        'snr_improvement': snr_improvement,
        'pesq': pesq_score,
        'stoi': stoi_score,
        'f0_stability_improvement': f0_stability_improvement
    }

def main(input_file, output_file):
    """主处理函数"""
    plt.figure(figsize=(15, 20))
    
    # 1. 加载原始音频
    original_audio, sr = load_audio(input_file)
    plot_waveform(original_audio, "Original Audio", 1)
    
    # 2. 第一步：噪声抑制
    denoised_audio = step1_noise_reduction(original_audio)
    plot_waveform(denoised_audio, "Step1: Noise Reduced", 2, 'g')
    sf.write("step1_denoised.wav", denoised_audio, SAMPLE_RATE)
    
    # 3. 第二步：静音检测
    voiced_audio = step2_silence_detection(denoised_audio)
    plot_waveform(voiced_audio, "Step2: Silence Removed", 3, 'r')
    sf.write("step2_voiced.wav", voiced_audio, SAMPLE_RATE)
    
    # 4. 第三步：异常检测和参数配置
    params = step3_abnormality_detection(voiced_audio)
    
    # 5. 第四步：自适应增益控制
    agc_audio = step4_adaptive_gain_control(voiced_audio, params)
    plot_waveform(agc_audio, "Step4: Gain Adjusted", 4, 'c')
    sf.write("step4_agc.wav", agc_audio, SAMPLE_RATE)
    
    # 6. 第五步：基频校正
    pitch_corrected_audio, original_f0, smoothed_f0 = step5_pitch_correction(agc_audio, params)
    plot_waveform(pitch_corrected_audio, "Step5: Pitch Corrected", 5, 'm')
    sf.write("step5_pitch_corrected.wav", pitch_corrected_audio, SAMPLE_RATE)
    
    # 绘制基频对比图
    plt.subplot(4, 2, 6)
    times = np.arange(len(original_f0)) * HOP_LENGTH / SAMPLE_RATE
    plt.plot(times, original_f0, label='Original F0')
    plt.plot(times, smoothed_f0, label='Smoothed F0')
    plt.title("Pitch Contour Comparison")
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.legend()
    
    # 7. 第六步：共振峰校正
    enhanced_audio = step6_formant_correction(pitch_corrected_audio, params)
    plot_waveform(enhanced_audio, "Step6: Formant Corrected", 7, 'y')
    sf.write(output_file, enhanced_audio, SAMPLE_RATE)
    
    # 8. 评估增强效果
    evaluation = evaluate_enhancement(original_audio[:len(enhanced_audio)], enhanced_audio)
    print( evaluation)
    
    # 显示评估结果
    plt.subplot(4, 2, 8)
    plt.axis('off')
    eval_text = (
        f"Enhancement Evaluation:\n"
        f"SNR Improvement: {evaluation['snr_improvement']:.2f} dB\n"
        f"PESQ Score: {evaluation['pesq']:.2f}\n"
        f"STOI Score: {evaluation['stoi']:.2f}\n"
        f"F0 Stability Improvement: {evaluation['f0_stability_improvement']:.1f}%"
    )
    plt.text(0.1, 0.5, eval_text, fontsize=12)
    
    plt.tight_layout()
    plt.savefig("enhancement_process.png")
    plt.show()

    return evaluation

if __name__ == "__main__":
    input_file = "recordings/睡觉3.wav"  # 替换为你的输入文件
    output_file = "enhanced_speech.wav"
    
    print(f"Processing {input_file} for {DISEASE_TYPE}...")
    evaluation = main(input_file, output_file)
    
    print("\nEnhancement Evaluation Results:")
    print(f"SNR Improvement: {evaluation['snr_improvement']:.2f} dB")
    print(f"PESQ Score (Quality): {evaluation['pesq']:.2f}")
    print(f"STOI Score (Intelligibility): {evaluation['stoi']:.2f}")
    print(f"F0 Stability Improvement: {evaluation['f0_stability_improvement']:.1f}%")
    