import parselmouth
from parselmouth.praat import call

def change_pitch(sound, factor):
    manipulation = call(sound, "To Manipulation", 0.01, 75, 600)
    pitch_tier = call(manipulation, "Extract pitch tier")
    call(pitch_tier, "Multiply frequencies", sound.xmin, sound.xmax, factor)
    call([pitch_tier, manipulation], "Replace pitch tier")
    return call(manipulation, "Get resynthesis (overlap-add)")

if __name__ == "__main__":
    yuyin = "上厕所99"
    input_file = f"recordings/{yuyin}.wav"  # 替换为你的音频文件
    output_file = f"recordings/enhanced16_{yuyin}.wav"

    sound = parselmouth.Sound(input_file)
  
    print("开始音高增强处理...")
    enhanced_audio = change_pitch(sound, 2.0)
    enhanced_audio.save(output_file, "WAV")
    print(f"\n处理完成! 结果已保存到 {output_file}")