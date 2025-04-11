#针对构音障碍患者将声母"p"发成"b"的问题，可通过增强送气音特征进行音频修复。以下是基于Python的完整解决方案：

import numpy as np  
import librosa  
import soundfile as sf  
#from pydub import AudioSegment  
from scipy.signal import butter, lfilter

def enhance_p_sound(input_path, output_path):  
    # 加载音频  
    y, sr = librosa.load(input_path, sr=None)  
      
    # 1. 延长嗓音起始时间(VOT)  
    vot_extension = int(0.05 * sr)  # 延长50ms  
    silence = np.zeros(vot_extension)  
    y = np.concatenate([y[:len(y)//2], silence, y[len(y)//2:]])  
      
    # 2. 增强高频成分(2000-4000Hz)  
    b, a = butter(4, [2000/(sr/2), 4000/(sr/2)], btype='band')  
    #b, a = butter(4, [1000, 2000], btype='band')  
    high_freq = lfilter(b, a, y)  
    y = y + 0.5 * high_freq  
      
    # 3. 添加送气白噪声  
    noise = np.random.normal(0, 0.02, len(y))  
    noise_env = np.linspace(0.5, 0, len(y))  # 噪声衰减包络  
    y = y + noise * noise_env  
      
    # 4. 动态范围压缩  
    y = np.tanh(2 * y)  # 模拟压缩效果  
      
    # 保存处理后的音频  
    sf.write(output_path, y, sr)

if __name__ == "__main__":
    yuyin = "上厕所99"
    input_file = f"recordings/{yuyin}.wav"  # 替换为你的音频文件
    output_file = f"recordings/enhanced17_{yuyin}.wav"

    print("开始增强p声...")
    enhance_p_sound(input_file, output_file)
    print(f"增强p声完成，保存为: {output_file}")
