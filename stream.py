import sounddevice as sd
import numpy as np
import torch
from encodec import EncodecModel
from encodec.utils import convert_audio
import typing as tp
import queue

# 初始化模型
model = EncodecModel.encodec_model_48khz()
model.set_target_bandwidth(12.0)
sample_rate = model.sample_rate
channels = model.channels

# 音频缓冲区
audio_buffer = queue.Queue()
block_size = 4096  # 每次处理的样本数


def audio_callback(indata: np.ndarray, outdata: np.ndarray,
                   frames: int, time, status):
    """实时音频回调函数"""
    if status:
        print(status)

    # 将输入数据放入缓冲区
    audio_buffer.put(indata.copy())

    # 如果缓冲区有足够数据，处理并播放
    if audio_buffer.qsize() >= 2:  # 调整这个值以获得最佳延迟
        # 获取数据
        chunks = []
        while not audio_buffer.empty():
            chunks.append(audio_buffer.get())
        audio_np = np.concatenate(chunks, axis=0)

        # 转换为模型输入格式
        audio_torch = torch.from_numpy(audio_np.T).unsqueeze(0)  # [1, C, T]
        audio_torch = convert_audio(audio_torch, sample_rate, sample_rate, channels)

        # 编码
        with torch.no_grad():
            encoded_frames = model.encode(audio_torch)
            # 解码
            decoded = decode_frames(encoded_frames)

        # 转换为numpy输出
        outdata[:] = decoded.squeeze().T.numpy()[:frames]
    else:
        outdata.fill(0)


def decode_frames(encoded_frames):
    """解码帧"""
    segment_length = model.segment_length
    if segment_length is None:
        assert len(encoded_frames) == 1
        return model._decode_frame(encoded_frames[0])

    frames = [model._decode_frame(frame) for frame in encoded_frames]
    return model._linear_overlap_add(frames, model.segment_stride or 1)


# 开始实时处理
print("Starting real-time encodec test...")
with sd.Stream(samplerate=sample_rate, blocksize=block_size,
               channels=channels, dtype='float32',
               latency='low',
               callback=audio_callback):
    print("Press Enter to stop...")
    input()
