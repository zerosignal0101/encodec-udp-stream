import torch
import torchaudio
import typing as tp
import socket
import threading
import time
import io
import struct
import numpy as np
import argparse
import math
import sounddevice as sd
import encodec.binary as binary
from encodec import EncodecModel
from encodec.utils import convert_audio
from encodec.compress import compress_to_file
from encodec.quantization.ac import ArithmeticCoder, ArithmeticDecoder, build_stable_quantized_cdf

MODELS = {
    'encodec_24khz': EncodecModel.encodec_model_24khz,
    'encodec_48khz': EncodecModel.encodec_model_48khz,
}

EncodedFrame = tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]


class AudioUDPSender:
    def __init__(self, model_name='encodec_48khz', use_lm=False, chunk_size=48000, target_ip='127.0.0.1',
                 target_port=12345):
        """
        初始化UDP音频发送器

        Args:
            model_name: Encodec模型名称
            use_lm: 是否使用语言模型进一步压缩
            chunk_size: 音频分块大小（样本数）
            target_ip: 目标IP地址
            target_port: 目标端口
        """
        self.model_name = model_name
        self.use_lm = use_lm
        self.chunk_size = chunk_size
        self.target_ip = target_ip
        self.target_port = target_port

        # 初始化Encodec模型
        self.device = torch.device("cpu")
        # torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MODELS[f'{self.model_name}']().to(self.device)
        self.sample_rate = self.model.sample_rate
        self.channels = self.model.channels

        # 48kHz 模型设置为 12kbps
        if self.model_name == 'encodec_48khz':
            self.model.set_target_bandwidth(12)

        # 24kHz 模型设置为 6kbps
        if self.model_name == 'encodec_24khz':
            self.model.set_target_bandwidth(6)

        # 初始化UDP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # 序列号
        self.seq_num = 0

    def compress_chunk(self, wav_chunk):
        """压缩单个音频块"""
        # 使用内存缓冲区存储压缩数据
        fo = io.BytesIO()

        compression_start_time = time.time()

        # 压缩音频
        compress_to_file(self.model, wav_chunk, fo, use_lm=self.use_lm)

        compression_end_time = time.time()

        print(f"Compression took {compression_end_time - compression_start_time:.6f} seconds")

        # 获取压缩后的数据
        fo.seek(0)
        compressed_data = fo.read()

        return compressed_data

    def send_audio_file(self, audio_file_path):
        """发送音频文件"""
        # 加载音频文件
        wav, sr = torchaudio.load(audio_file_path)

        # 转换到目标格式
        wav = convert_audio(wav, sr, self.sample_rate, self.channels)

        # 分块处理
        num_samples = wav.shape[-1]

        print(f"发送音频: {audio_file_path}")
        print(f"总样本数: {num_samples}, 采样率: {self.sample_rate}")

        # 按块处理和发送
        for start in range(0, num_samples, self.chunk_size):
            end = min(start + self.chunk_size, num_samples)
            chunk = wav[:, start:end]

            # 压缩音频块
            compressed_data = self.compress_chunk(chunk)

            # 创建UDP包头 (序列号 + 数据长度)
            header = struct.pack("!II", self.seq_num, len(compressed_data))

            # 发送数据包
            packet = header + compressed_data
            self.sock.sendto(packet, (self.target_ip, self.target_port))

            print(f"发送数据包 #{self.seq_num}: {len(compressed_data)} 字节")
            self.seq_num += 1

            # 模拟实时流式传输的延迟
            time.sleep(chunk.shape[-1] / self.sample_rate * 0.90)  # 略快一点发送，防止播放端缓冲区空

        # 发送结束标记
        end_header = struct.pack("!II", self.seq_num, 0)
        self.sock.sendto(end_header, (self.target_ip, self.target_port))
        print("音频传输完成")


class AudioUDPReceiver:
    def __init__(self, model_name='encodec_48khz', buffer_size=16, listen_port=12345):
        """
        初始化UDP音频接收器

        Args:
            model_name: Encodec模型名称
            buffer_size: 音频缓冲区大小（帧数）
            listen_port: 监听端口
        """
        self.model_name = model_name
        self.buffer_size = buffer_size
        self.listen_port = listen_port

        # 初始化Encodec模型
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MODELS[f'{self.model_name}']().to(self.device)
        self.sample_rate = self.model.sample_rate
        self.channels = self.model.channels

        # 初始化UDP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('0.0.0.0', listen_port))

        # 音频缓冲区
        self.audio_buffer = []
        self.buffer_lock = threading.Lock()

        # 控制标志
        self.running = False
        self.playback_started = False

        # 接收到的数据包序列号
        self.received_packets = set()

        # 最后播放的序列号
        self.last_played_seq = -1

    def decompress_chunk(self, compressed_data):
        """解压缩音频块"""
        fo = io.BytesIO(compressed_data)
        metadata = binary.read_ecdc_header(fo)
        model_name = metadata['m']
        audio_length = metadata['al']
        num_codebooks = metadata['nc']
        use_lm = metadata['lm']
        assert isinstance(audio_length, int)
        assert isinstance(num_codebooks, int)
        if model_name not in MODELS:
            raise ValueError(f"The audio was compressed with an unsupported model {model_name}.")
        if model_name != self.model_name:
            self.model_name = model_name
            self.model = MODELS[f'{self.model_name}']().to(self.device)
            self.sample_rate = self.model.sample_rate
            self.channels = self.model.channels

        if use_lm:
            lm = self.model.get_lm_model()

        frames: tp.List[EncodedFrame] = []
        segment_length = self.model.segment_length or audio_length
        segment_stride = self.model.segment_stride or audio_length
        for offset in range(0, audio_length, segment_stride):
            this_segment_length = min(audio_length - offset, segment_length)
            frame_length = int(math.ceil(this_segment_length / self.model.sample_rate * self.model.frame_rate))
            if self.model.normalize:
                scale_f, = struct.unpack('!f', binary._read_exactly(fo, struct.calcsize('!f')))
                scale = torch.tensor(scale_f, device=self.device).view(1)
            else:
                scale = None
            if use_lm:
                decoder = ArithmeticDecoder(fo)
                states: tp.Any = None
                offset = 0
                input_ = torch.zeros(1, num_codebooks, 1, dtype=torch.long, device=self.device)
            else:
                unpacker = binary.BitUnpacker(self.model.bits_per_codebook, fo)
            frame = torch.zeros(1, num_codebooks, frame_length, dtype=torch.long, device=self.device)
            for t in range(frame_length):
                if use_lm:
                    with torch.no_grad():
                        probas, states, offset = lm(input_, states, offset)
                code_list: tp.List[int] = []
                for k in range(num_codebooks):
                    if use_lm:
                        q_cdf = build_stable_quantized_cdf(
                            probas[0, :, k, 0], decoder.total_range_bits, check=False)
                        code = decoder.pull(q_cdf)
                    else:
                        code = unpacker.pull()
                    if code is None:
                        raise EOFError("The stream ended sooner than expected.")
                    code_list.append(code)
                codes = torch.tensor(code_list, dtype=torch.long, device=self.device)
                frame[0, :, t] = codes
                if use_lm:
                    input_ = 1 + frame[:, :, t: t + 1]
            frames.append((frame, scale))
        with torch.no_grad():
            wav = self.model.decode(frames)
        wav = wav[0, :, :audio_length]
        # # Debug
        # print("Wav shape: ", wav.shape)
        return wav

    def receive_thread(self):
        """接收线程"""
        print(f"开始接收音频数据, 监听端口: {self.listen_port}")

        while self.running:
            try:
                # 接收数据包
                data, addr = self.sock.recvfrom(65536)  # UDP包最大大小

                # 解析头部
                header_size = struct.calcsize("!II")
                seq_num, data_len = struct.unpack("!II", data[:header_size])

                # 如果是结束标记，则退出
                if data_len == 0:
                    print(f"收到结束标记 (序列号 {seq_num})")
                    # 等待所有数据播放完毕
                    time.sleep(2)
                    self.running = False
                    break

                # 提取音频数据
                compressed_data = data[header_size:]

                # 检查是否已经接收过此包
                if seq_num in self.received_packets:
                    print(f"重复数据包 #{seq_num}，跳过")
                    continue

                print(f"接收数据包 #{seq_num}: {data_len} 字节")
                self.received_packets.add(seq_num)

                # 解压缩
                wav_chunk = self.decompress_chunk(compressed_data)

                # 添加到缓冲区
                with self.buffer_lock:
                    # 添加 (seq_num, wav_chunk) 元组到缓冲区
                    self.audio_buffer.append((seq_num, wav_chunk))
                    # 按序列号排序
                    self.audio_buffer.sort(key=lambda x: x[0])

                # 如果缓冲区足够大，开始播放
                if not self.playback_started and len(self.audio_buffer) >= self.buffer_size // 2:
                    self.playback_started = True
                    threading.Thread(target=self.playback_thread).start()

            except Exception as e:
                print(f"接收错误: {e}")

    def playback_thread(self):
        """播放线程"""
        print("开始音频播放")

        # 初始化音频播放设备
        stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype='float32'
        )
        stream.start()

        while self.running or self.audio_buffer:
            with self.buffer_lock:
                if not self.audio_buffer:
                    continue

                # 获取下一个要播放的音频帧
                next_seq, next_chunk = self.audio_buffer[0]

                # 检查序列号是否连续
                if next_seq != self.last_played_seq + 1 and self.last_played_seq != -1:
                    print(f"检测到丢包: 期望 {self.last_played_seq + 1}, 收到 {next_seq}")

                # 从缓冲区移除
                self.audio_buffer.pop(0)

            # 播放
            audio_np = np.ascontiguousarray(next_chunk.cpu().numpy().T)
            stream.write(audio_np)

            self.last_played_seq = next_seq
            print(f"播放音频帧 #{next_seq}, 缓冲区大小: {len(self.audio_buffer)}")

        # 关闭流
        stream.stop()
        stream.close()
        print("音频播放结束")

    def start(self):
        """开始接收和播放"""
        self.running = True
        threading.Thread(target=self.receive_thread).start()

    def stop(self):
        """停止接收和播放"""
        self.running = False


def main():
    parser = argparse.ArgumentParser(description="UDP流式音频传输")
    parser.add_argument('--mode', type=str, choices=['send', 'receive'], required=True, help='运行模式: send或receive')
    parser.add_argument('--input', type=str, help='输入音频文件路径 (仅发送模式)')
    parser.add_argument('--ip', type=str, default='127.0.0.1', help='接收端IP地址 (仅发送模式)')
    parser.add_argument('--port', type=int, default=12345, help='UDP端口')
    parser.add_argument('--no-lm', action='store_true', help='不使用语言模型压缩')
    args = parser.parse_args()

    if args.mode == 'send':
        if not args.input:
            parser.error("发送模式需要指定输入音频文件 (--input)")

        sender = AudioUDPSender(
            use_lm=not args.no_lm,
            target_ip=args.ip,
            target_port=args.port
        )
        sender.send_audio_file(args.input)

    elif args.mode == 'receive':
        receiver = AudioUDPReceiver(listen_port=args.port)
        receiver.start()

        try:
            # 保持主线程运行
            while receiver.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("接收中断")
        finally:
            receiver.stop()


if __name__ == "__main__":
    main()



