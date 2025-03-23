import torch
import torchaudio
from speechbrain.pretrained import EncoderClassifier
from sklearn.metrics.pairwise import cosine_similarity

# 配置参数
MODEL_NAME = "speechbrain/spkrec-ecapa-voxceleb"
MODEL_SAVE_DIR = "pretrained_models/ecapa_tdnn"  # 模型保存目录

def preprocess_audio(audio_path):
    """音频预处理：统一为16kHz单声道格式"""
    waveform, orig_sr = torchaudio.load(audio_path)
    
    # 重采样至16kHz
    if orig_sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_sr, 16000)
        waveform = resampler(waveform)
    
    # 合并为单声道
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    return waveform.squeeze(0)  # 输出形状: (num_samples,)

def normalize_embedding(embedding):
    """归一化嵌入向量"""
    return embedding / torch.norm(embedding)

def get_speech_embedding(audio_path, classifier):
    """提取语音嵌入向量"""
    waveform = preprocess_audio(audio_path)
    
    # 添加批次维度: (batch_size, num_samples)
    waveform_batch = waveform.unsqueeze(0).to(classifier.device)
    
    # 提取嵌入特征
    with torch.no_grad():
        embedding = classifier.encode_batch(waveform_batch)

    # 归一化嵌入向量
    embedding = normalize_embedding(embedding)
    
    return embedding.squeeze().cpu().numpy()  # 形状: (192,)

def get_speech_embedding2(audio_path, classifier):
    signal, sr = torchaudio.load(audio_path)
    assert sr == 16000, "需重采样至16kHz"
    embeddings = classifier.encode_batch(signal)
    return embeddings.squeeze().numpy()

def main():
    # 加载预训练模型（自动下载）
    classifier = EncoderClassifier.from_hparams(
        source=MODEL_NAME,
        savedir=MODEL_SAVE_DIR,
        run_opts={"device": "cpu"}  # 可选"cuda"使用GPU
    )

    yuyins = ['饮水1','吃饭', '吃饭2','喝茶','饮茶','饮水2','吃饭3']
    embs = []
    for yuyin in yuyins:   
        # 提取音频的嵌入向量
        emb = get_speech_embedding2(f'recordings/clean_{yuyin}.wav', classifier)
        embs.append(emb)

    for i in range(len(yuyins)):
        # 打印归一化后的embedding特征形状
        print(f"Normalized embedding shape for {yuyins[i]}:", embs[i].shape)
        for j in range(len(yuyins)):
            if i == j:
                continue   
            # 计算余弦相似度
            similarity = cosine_similarity([embs[i]], [embs[j]])[0][0]
            print(f"Similarity between {yuyins[i]} and {yuyins[j]}:", similarity)   

if __name__ == "__main__":
    main()