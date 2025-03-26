import torch
import torchaudio
import librosa
import numpy as np
from torchmetrics.functional.audio import signal_noise_ratio
from torchmetrics.functional.audio.stoi import short_time_objective_intelligibility

class PathologicalVoiceEnhancer:
    def __init__(self, sr=16000):
        self.sr = sr
        self.f0_min = 80
        self.f0_max = 400
        
        # 加载预训练的LSTM基频模型
        #self.f0_model = torch.jit.load('f0_predictor.pt') 
        
        # 共振峰校正系数 (示例数据)
        self.formant_ratio = [1.2, 0.9, 1.1]  # F1/F2/F3校正比例

    def _extract_f0(self, audio):
        # 高精度基频提取
        f0, voiced_flag, voiced_probs = librosa.pyin(audio.numpy(), 
                         fmin=self.f0_min,
                         fmax=self.f0_max,
                         frame_length=2048,
                         hop_length=512)
        #return torch.from_numpy(f0).float()
        return torch.from_numpy(f0).float()
    """
    def _lstm_f0_correction(self, f0):
        # 输入: (seq_len,) 输出: (seq_len,)
        seq_len = len(f0)
        with torch.no_grad():
            pred_f0 = self.f0_model(f0.view(1,1,-1)) 
        return pred_f0.squeeze()
    """
    def _dynamic_range_compression(self, f0):
        # 动态基频范围压缩
        median = torch.nanmedian(f0)
        f0_norm = (f0 - median) * 0.5 + median
        return torch.clamp(f0_norm, self.f0_min, self.f0_max)

    def process_audio(self, input_wave):
        # 输入: (1, samples) 波形数据
        # 特征提取
        f0 = self._extract_f0(input_wave)
        
        # 基频处理流程
        #if f0.std() > 50:  # 抖动过大时启用预测
        #    f0 = self._lstm_f0_correction(f0)
        if np.nanmax(f0.numpy()) - np.nanmin(f0.numpy()) > 10:  # 范围过宽时压缩
            f0 = self._dynamic_range_compression(f0)

        print(f0)

        # 基频平滑
        pitch_shift = self._calc_pitch_shift(f0)
        print(pitch_shift)
        
        # 语音重构（PSOLA算法简化版）
        enhanced = torchaudio.functional.pitch_shift(
            input_wave, self.sr, n_steps=pitch_shift
        )
        
        # 共振峰校正（线性预测编码）
        enhanced = self._formant_correction(enhanced)
        
        return enhanced

    def _calc_pitch_shift(self, f0):
        # 计算相对于目标基频的音调偏移量
        target_f0 = 150  # 女性设为220
        return 12 * np.log2(target_f0 / np.nanmedian(f0.numpy()))

    def _formant_correction(self, wave):
        # 共振峰线性缩放
        n_fft = 512
        spec = torch.stft(wave, n_fft, return_complex=True)
        freqs = torch.fft.rfftfreq(n_fft, 1/self.sr)
        
        # 调整共振峰位置
        for i, ratio in enumerate(self.formant_ratio):
            peak_idx = int((500*(i+1)*ratio)/self.sr * n_fft)
            spec[:, peak_idx] *= 2  # 增强目标频段
            
        return torch.istft(spec, n_fft)

    def evaluate_enhancement(self, original, enhanced):
        # 信噪比提升量
        snr_improve = signal_noise_ratio(enhanced, original, sr)

        
        # 语音可懂度评估
        # 短时客观可懂度
        stoi_score = short_time_objective_intelligibility(enhanced, original, fs = self.sr, extended=False)
        
        # 基频稳定性
        f0_std = np.nanstd(self._extract_f0(enhanced).numpy())
        
        return {
            "SNR_improvement": snr_improve,
            "STOI": stoi_score,
            "F0_stability": f0_std
        }

if __name__ == "__main__":
 
    audio_path = 'recordings/睡觉3.wav'

    # 加载音频文件
    original, sr = torchaudio.load(audio_path)

    enhancer = PathologicalVoiceEnhancer()
    # 音频增强
    enhanced = enhancer.process_audio(original)
    # 评估增强效果
    metrics = enhancer.evaluate_enhancement(original, enhanced)
    print(metrics)

    # 保存结果
    torchaudio.save("enhanced_audio.wav", enhanced, enhancer.sr)

