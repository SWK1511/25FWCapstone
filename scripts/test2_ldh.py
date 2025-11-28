import adi
import numpy as np
import time
import sys

# --- 사용자 설정 ---
SDR_IP = "ip:192.168.2.1"
CENTER_FREQ = 2.4e9
SAMPLE_RATE = 6e6
BANDWIDTH = 50e6
NUM_SAMPLES = 2**14

try:
    sdr = adi.Pluto(uri=SDR_IP)
    sdr.sample_rate = int(SAMPLE_RATE)
    sdr.rx_buffer_size = NUM_SAMPLES
    sdr.tx_lo = int(CENTER_FREQ)
    sdr.rx_lo = int(CENTER_FREQ)
    sdr.tx_cyclic_buffer = True
    
    # [수정 1] 게인을 다시 안전한 수준으로 낮춤 (50 -> 20)
    # 일단 신호를 잡는 게 중요하니 욕심을 버리고 낮춥니다.
    sdr.tx_hardwaregain_chan0 = 0 
    sdr.rx_gain_mode_chan0 = "manual"
    sdr.rx_hardwaregain_chan0 = 20 
    
    print(f">> Pluto SDR 연결 성공. (Gain: 20)")

except Exception as e:
    print(f"!! 연결 실패: {e}")
    sys.exit(1)

# --- 파형 송신 ---
t = np.arange(NUM_SAMPLES) / SAMPLE_RATE
chirp_signal = np.exp(1j * np.pi * (BANDWIDTH/SAMPLE_RATE) * (t**2))
chirp_signal *= 2**14
sdr.tx(chirp_signal)

# --- 배경 제거 (Calibration) ---
print("\n" + "="*50)
print(">> [캘리브레이션] 2초간 꼼짝 마세요! (배경 학습 중)")
print("="*50)

clutter_sum = np.zeros(NUM_SAMPLES, dtype=complex)
for i in range(30):
    rx = sdr.rx()
    mixed = rx * np.conj(chirp_signal)
    clutter_sum += mixed
    time.sleep(0.01)
clutter_avg = clutter_sum / 30
print(">> 학습 완료. 이제 손을 흔들어보세요.\n")

# --- 메인 루프 ---
try:
    while True:
        rx_data = sdr.rx()
        
        # MTI (배경 제거)
        mixed_signal = rx_data * np.conj(chirp_signal)
        moving_signal = mixed_signal - clutter_avg 
        
        fft_out = np.fft.fft(moving_signal)
        fft_mag = 20 * np.log10(np.abs(fft_out) + 1e-12)
        
        # [수정 2] 블라인드 존 설정 (Blind Zone)
        # 0~5번이 아니라 0~20번 Bin까지 과감하게 버립니다.
        # 이렇게 하면 'Leakage' 신호를 무시하고 그 뒤에 있는 손만 봅니다.
        start_bin = 20 
        end_bin = 300
        
        valid_range = fft_mag[start_bin:end_bin]
        
        if len(valid_range) == 0:
            peak_val = 0
            peak_idx = 0
        else:
            peak_val = np.max(valid_range)
            peak_idx = np.argmax(valid_range) + start_bin
        
        # [수정 3] 시각화 임계값 조정
        # 게인을 20으로 낮췄으므로, 감지 기준도 낮춥니다 (40dB 이상이면 감지)
        threshold = 40 
        
        bar_len = int(peak_val - threshold)
        if bar_len < 0: bar_len = 0
        if bar_len > 40: bar_len = 40
        
        bar = "#" * bar_len
        space = " " * (40 - bar_len)
        
        # Bin 정보 출력 (이 숫자가 중요함)
        dist_info = f"Bin: {peak_idx:3d}"
            
        sys.stdout.write(f"\rLevel: {peak_val:5.1f} dB | {dist_info} | [{bar}{space}]")
        sys.stdout.flush()
        
        time.sleep(0.05)

except KeyboardInterrupt:
    sdr.tx_destroy_buffer()
    del sdr
    print("\n>> 종료")