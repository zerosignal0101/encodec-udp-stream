import socket
import threading
import time
import numpy as np
import torch
import torchaudio
from queue import Queue
from encodec import EncodecModel
import pyaudio  # 用于播放音频

# UDP配置
UDP_IP = "127.0.0.1"
UDP_PORT = 9527
PACKET_SIZE = 192  # 字节，与发送端一致

# 初始化Encodec模型
model = EncodecModel.encodec_model_48khz()
model.set_target_bandwidth(12.0)

# 音频参数
SAMPLE_RATE = model.sample_rate  # 48000
CHANNELS = model.channels  # 2 (立体声)
CHUNK_DURATION = 0.1  # 与发送端一致
CHUNK_SAMPLES = int(CHUNK_DURATION * SAMPLE_RATE)

# 创建UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.settimeout(1.0)  # 设置超时时间

# 数据队列
packet_queue = Queue(maxsize=100)
audio_queue = Queue(maxsize=10)

# PyAudio配置
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paFloat32,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                output=True,
                frames_per_buffer=CHUNK_SAMPLES)


def udp_receiver():
    """接收UDP数据包并放入队列"""
    buffer = {}  # 用于重组数据包 {sequence_num: {fragment_num: data}}
    current_seq = -1

    while True:
        try:
            data, _ = sock.recvfrom(PACKET_SIZE + 6)  # 6字节包头
            if not data:
                continue

            # 解析包头 (4字节序列号 + 2字节分片号)
            seq_num = int.from_bytes(data[:4], 'big')
            frag_num = int.from_bytes(data[4:6], 'big')
            payload = data[6:]

            # 初始化缓冲区
            if seq_num not in buffer:
                buffer[seq_num] = {}

            buffer[seq_num][frag_num] = payload

            # 检查是否收到完整帧
            if seq_num != current_seq:
                if current_seq != -1 and current_seq in buffer:
                    # 检查前一帧是否完整
                    fragments = buffer[current_seq]
                    max_frag = max(fragments.keys())
                    expected_frags = set(range(max_frag + 1))
                    if expected_frags.issubset(fragments.keys()):
                        # 重组完整数据
                        fragments = [fragments[i] for i in sorted(fragments.keys())]
                        full_data = b''.join(fragments)
                        packet_queue.put((current_seq, full_data))
                        del buffer[current_seq]

                current_seq = seq_num

        except socket.timeout:
            continue
        except Exception as e:
            print(f"接收错误: {e}")
            continue


def decode_worker():
    """解码工作线程"""
    while True:
        packet = packet_queue.get()
        if packet is None:  # 结束信号
            audio_queue.put(None)
            break

        seq_num, data = packet

        # 将字节流转换为numpy数组
        codes = np.frombuffer(data, dtype=np.int32)
        # 转换为Encodec需要的格式 [B, n_q, T]
        codes = torch.from_numpy(codes).reshape(1, model.quantizer.n_q, -1)

        # 解码
        with torch.no_grad():
            # 构造EncodedFrame格式: (codes, None)
            encoded_frame = (codes, None)
            decoded = model.decode([encoded_frame])

        # 放入音频队列
        audio_queue.put(decoded.squeeze(0))  # 移除batch维度


def audio_player():
    """音频播放线程"""
    while True:
        audio = audio_queue.get()
        if audio is None:
            break

        # 转换为numpy数组并确保是float32
        audio_np = audio.cpu().numpy().astype(np.float32)

        # 播放音频
        stream.write(audio_np.tobytes())


if __name__ == "__main__":
    # 启动工作线程
    receiver_thread = threading.Thread(target=udp_receiver)
    decoder_thread = threading.Thread(target=decode_worker)
    player_thread = threading.Thread(target=audio_player)

    receiver_thread.start()
    decoder_thread.start()
    player_thread.start()

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("正在停止...")
        # 发送结束信号
        packet_queue.put(None)

        # 等待线程结束
        receiver_thread.join()
        decoder_thread.join()
        player_thread.join()

        # 关闭资源
        stream.stop_stream()
        stream.close()
        p.terminate()
        sock.close()
        print("接收端已关闭")
