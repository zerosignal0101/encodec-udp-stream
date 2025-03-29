import socket
import threading
import time
from queue import Queue
from encodec import EncodecModel
from encodec.utils import convert_audio
import torchaudio
import torch

# UDP配置
UDP_IP = "127.0.0.1"
UDP_PORT = 9527
PACKET_SIZE = 192  # 字节

# 初始化Encodec模型
model = EncodecModel.encodec_model_48khz()
model.set_target_bandwidth(12.0)

# 创建UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024 * 1024)  # 增加发送缓冲区

# 音频处理参数
CHUNK_DURATION = 0.1  # 每次处理的音频时长(秒)
SAMPLE_RATE = model.sample_rate
CHANNELS = model.channels
CHUNK_SAMPLES = int(CHUNK_DURATION * SAMPLE_RATE)

# Debug
print("CHUNK_SAMPLES", CHUNK_SAMPLES)
print("SAMPLE_RATE", SAMPLE_RATE)
print("CHANNELS", CHANNELS)
print("CHUNK_SAMPLES", CHUNK_SAMPLES)

# 数据队列
audio_queue = Queue(maxsize=10)
code_queue = Queue(maxsize=10)


def audio_loader(audio_path):
    """实时加载音频文件并分块"""
    wav, sr = torchaudio.load(audio_path)
    wav = convert_audio(wav, sr, SAMPLE_RATE, CHANNELS)

    total_samples = wav.shape[-1]
    pointer = 0

    while pointer < total_samples:
        end = min(pointer + CHUNK_SAMPLES, total_samples)
        chunk = wav[:, pointer:end]

        # 如果不足一个chunk，填充静音
        if chunk.shape[-1] < CHUNK_SAMPLES:
            padding = torch.zeros((CHANNELS, CHUNK_SAMPLES - chunk.shape[-1]))
            chunk = torch.cat([chunk, padding], dim=-1)

        audio_queue.put(chunk.unsqueeze(0))  # 添加batch维度
        pointer = end
        time.sleep(CHUNK_DURATION)  # 模拟实时

    audio_queue.put(None)  # 结束信号


def encode_worker():
    """编码工作线程"""
    while True:
        chunk = audio_queue.get()
        if chunk is None:
            code_queue.put(None)
            break

        with torch.no_grad():
            encoded_frames = model.encode(chunk)
            codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)  # [B, n_q, T]
            code_queue.put(codes.cpu().numpy())  # 转换为numpy数组


def udp_sender():
    """UDP发送线程"""
    sequence_num = 0

    while True:
        codes = code_queue.get()
        if codes is None:
            break

        # 将codes转换为字节流
        codes_bytes = codes.tobytes()
        total_length = len(codes_bytes)

        # 分片发送
        for offset in range(0, total_length, PACKET_SIZE):
            chunk = codes_bytes[offset:offset + PACKET_SIZE]

            # 添加简单的包头 (序列号+分片号)
            header = sequence_num.to_bytes(4, 'big') + (offset // PACKET_SIZE).to_bytes(2, 'big')
            packet = header + chunk

            # # Debug
            # print("Length: ", len(packet))
            # print("Header Length: ", len(header))
            # print("Chunk Size: ", len(chunk))

            sock.sendto(packet, (UDP_IP, UDP_PORT))

        sequence_num += 1


if __name__ == "__main__":
    # 启动工作线程
    loader_thread = threading.Thread(target=audio_loader, args=("data/Zeraphym - Lifeline_cut_02_stereo.wav",))
    encoder_thread = threading.Thread(target=encode_worker)
    sender_thread = threading.Thread(target=udp_sender)

    loader_thread.start()
    encoder_thread.start()
    sender_thread.start()

    # 等待完成
    loader_thread.join()
    encoder_thread.join()
    sender_thread.join()

    sock.close()
    print("传输完成")
