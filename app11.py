import torch
import torchaudio
import numpy as np
import librosa
from torchaudio.transforms import MelSpectrogram

class MedicalVoiceEnhancer:
    def __init__(self, sr=16000, device='cpu'):
        self.sr = sr
        self.device = device
        
        # 噪声抑制模块
        self.noise_gate = NoiseGate(sr=sr)
        
        # 自适应增益控制
        self.agc = AdaptiveGainControl(sr=sr)
        
        # 基频增强模块
        self.pitch_corrector = PitchCorrector(sr=sr)
        
        # 共振峰校正模块
        self.formant_shifter = FormantShifter(sr=sr)

    def enhance(self, input_wave):
        # 输入音频标准化
        waveform = self._preprocess(input_wave)
        
        # 处理流程
        cleaned = self.noise_gate.apply(waveform)         # 步骤1: 降噪
        leveled = self.agc.process(cleaned)               # 步骤2: 增益控制
        pitched = self.pitch_corrector(leveled)           # 步骤3: 基频稳定
        output = self.formant_shifter(pitched)            # 步骤4: 共振峰校正
        
        return output.cpu()

    def _preprocess(self, waveform):
        # 格式转换与归一化
        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform).float()
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        return waveform.to(self.device)

class NoiseGate:
    def __init__(self, sr, threshold_db=-30, n_fft=512):
        self.sr = sr
        self.threshold = 10**(threshold_db / 20)
        self.n_fft = n_fft
        self.hop = n_fft // 4
        
        # 创建Mel滤波器组
        self.mel_filter = MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=self.hop,
            n_mels=40
        ).to('cpu')

    def apply(self, waveform):
        # 计算Mel能量
        spec = self.mel_filter(waveform)
        energy = torch.mean(spec, dim=1)
        
        # 生成噪声门掩码
        mask = (energy > self.threshold).float()
        mask = torch.nn.functional.interpolate(
            mask.unsqueeze(0), 
            size=waveform.shape[-1],
            mode='linear'
        ).squeeze()
        
        return waveform * mask

class AdaptiveGainControl:
    def __init__(self, sr, target_dbfs=-20, attack=0.01, release=0.1):
        self.sr = sr
        self.target_amplitude = 10**(target_dbfs / 20)
        self.attack = int(attack * sr)
        self.release = int(release * sr)
        
    def process(self, waveform):
        # 计算短时能量包络
        envelope = torch.sqrt(torch.mean(waveform**2, dim=1, keepdim=True))
        
        # 动态增益计算
        gain = self.target_amplitude / (envelope + 1e-8)
        gain = torch.clamp(gain, 0.5, 2.0)
        
        # 增益平滑处理
        smoothed_gain = torch.nn.functional.avg_pool1d(
            gain, 
            kernel_size=self.attack + self.release,
            stride=1,
            padding=(self.attack + self.release)//2
        )
        smoothed_gain = torch.clamp(smoothed_gain, 0.5, 2.0)
        print(smoothed_gain)
        print(gain)
        return waveform * gain

class PitchCorrector(torch.nn.Module):
    def __init__(self, sr):
        super().__init__()
        self.sr = sr
        self.lstm = torch.nn.LSTM(1, 64, 2)
        self.fc = torch.nn.Linear(64, 1)
        
    def forward(self, waveform):
        # 基频提取
        f0 = self._extract_f0(waveform)
        
        # 验证基频有效性
        if torch.all(f0 == 0):
            return waveform
        
        # LSTM基频预测
        f0_in = f0.reshape(1, -1, 1)
        f0_pred, _ = self.lstm(f0_in)
        f0_pred = self.fc(f0_pred).squeeze()
        
        # 计算音调偏移并验证
        try:
            ratio = f0_pred.median() / f0.median()
            if not torch.isfinite(ratio) or ratio <= 0:
                return waveform
            n_steps = 12 * torch.log2(ratio)
            if not torch.isfinite(n_steps):
                return waveform
            
            # 音高校正
            shifted = torchaudio.functional.pitch_shift(
                waveform,
                self.sr,
                n_steps=n_steps
            )
            return shifted
        except Exception as e:
            print(f"音调偏移计算错误: {str(e)}")
            return waveform

    def _extract_f0(self, waveform):
        try:
            # 确保输入波形格式正确
            if waveform.dim() != 2 or waveform.shape[0] != 1:
                waveform = waveform.unsqueeze(0) if waveform.dim() == 1 else waveform
                waveform = waveform[:1] if waveform.shape[0] > 1 else waveform
            
            # 提取基频并处理异常值
            f0 = librosa.pyin(
                waveform.cpu().numpy().squeeze(),
                fmin=80,
                fmax=400,
                sr=self.sr,
                frame_length=2048
            )[0]
            f0 = torch.from_numpy(f0).float()
            return f0.nan_to_num().to(waveform.device)
        except Exception as e:
            print(f"基频提取错误: {str(e)}")
            return torch.zeros(waveform.shape[-1] // 512).to(waveform.device)

class FormantShifter(torch.nn.Module):
    def __init__(self, sr, shift_ratios=[1.2, 0.9, 1.1]):
        super().__init__()
        self.sr = sr
        self.shift_ratios = torch.tensor(shift_ratios)
        
    def forward(self, waveform):
        n_fft = 512
        spec = torch.stft(waveform, n_fft, return_complex=True)
        
        # 共振峰线性偏移
        freqs = torch.fft.rfftfreq(n_fft, 1/self.sr)
        for i, ratio in enumerate(self.shift_ratios):
            src_band = [300*(i+1), 500*(i+1)]
            tgt_band = [int(src_band[0]*ratio), int(src_band[1]*ratio)]
            
            # 频段重映射
            src_mask = (freqs >= src_band[0]) & (freqs <= src_band[1])
            tgt_pos = ((freqs - tgt_band[0]) / (tgt_band[1]-tgt_band[0])).clamp(0,1)
            spec[:, src_mask] = torch.nn.functional.interpolate(spec[:, src_mask].unsqueeze(0).unsqueeze(0), 
                                           size=(1, sum(src_mask)), 
                                           mode='bilinear', 
                                           align_corners=False).squeeze()
        
        return torch.istft(spec, n_fft)

# 使用示例
if __name__ == "__main__":
    enhancer = MedicalVoiceEnhancer()
    
    # 加载病理语音样本
    input_audio, sr = torchaudio.load("recordings/睡觉3.wav")
    if sr != enhancer.sr:
        resampler = torchaudio.transforms.Resample(sr, enhancer.sr)
        input_audio = resampler(input_audio)
    
    # 执行增强处理
    enhanced = enhancer.enhance(input_audio)
    
    # 保存结果
    torchaudio.save("enhanced_voice.wav", enhanced, enhancer.sr)