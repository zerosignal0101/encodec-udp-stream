# **Encodec-Based Real-Time Audio Streaming System**

## **Overview**
This project implements a low-latency UDP-based audio streaming system leveraging Meta's **Encodec neural codec** for efficient compression. The system supports:
- **Real-time streaming** of high-quality audio (24kHz/48kHz)
- **Adaptive bitrate** (12kbps default)
- **Packet loss resilience** through fragmentation/reassembly
- **Seamless playback** with overlap-add (OLA) windowing

## **Key Features**

### **1. Compression Pipeline**
- Utilizes Encodec's **neural audio codec** with optional arithmetic coding
- Supports both **24kHz** (mono) and **48kHz** (stereo) models
- Configurable bandwidth (6/12/24 kbps)

### **2. Network Layer**
- **UDP transport** with custom packet structure:
  ```python
  struct Packet {
      uint32_t seq_num;       // Sequence number
      uint32_t total_length;  // Compressed data size
      uint32_t has_overlap;   // Overlap flag
      uint32_t frag_id;       // Fragment ID
      uint32_t total_frags;   // Total fragments
      uint32_t frag_index;    // Current fragment index
      uint8_t  payload[];     // Encoded audio data
  }
  ```
- **220-byte MTU** fragmentation for NAT traversal
- Sequential reassembly with duplicate detection

### **3. Audio Processing**
- **Overlap-Add (OLA) smoothing**:
  $$
  y[n] = x_{prev}[n] \cdot w_{fade-out} + x_{current}[n] \cdot w_{fade-in}
  $$
  where $w$ is a triangular window (1-50% overlap configurable)
- **Jitter buffer** (16-frame default) for stable playback

## **Usage Examples**
### **Sender**
```bash
python stream.py --mode send \
    --input sample.wav \
    --ip 127.0.0.1 \
    --port 9527 \
    --no-lm \
    --overlap 25
```

### **Receiver**
```bash
python stream.py --mode receive \
    --port 6000 \
    --buffer-size 24 \
    --overlap 25
```

## **Dependencies**
- Encodec (`facebookresearch/encodec`)
- SoundDevice 0.4+
- NumPy

## **Limitations**
- No FEC (Forward Error Correction)
- Requires GPU for optimal performance

## **Future Work**
- [ ] Stream audio from mic source
- [ ] Multi-track audio transmission