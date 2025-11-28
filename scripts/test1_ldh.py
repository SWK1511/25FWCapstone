import adi
import numpy as np
import matplotlib.pyplot as plt
import time

# --- 1. 사용자 설정 (User Configuration) ---
SDR_IP = "ip:192.168.2.1" # Pluto+의 기본 IP (USB 연결 시)
CENTER_FREQ = 2.4e9       # 2.4 GHz (안테나 주파수에 맞춤)
SAMPLE_RATE = 6e6         # 6 MHz (샘플링 레이트, 높을수록 해상도 증가)
BANDWIDTH = 50e6           # 4 MHz (대역폭, 거리 해상도 결정)
NUM_SAMPLES = 2**14       # 한 번에 처리할 샘플 수 (FFT 크기 영향)

# --- 2. SDR 하드웨어 설정 ---
try:
    sdr = adi.Pluto(uri=SDR_IP)
    sdr.sample_rate = int(SAMPLE_RATE)
    
    # [!!! 중요 !!!] 수신 버퍼 크기를 우리가 만든 샘플 수와 똑같이 맞춰줘야 합니다.
    sdr.rx_buffer_size = NUM_SAMPLES 
    
    # 채널 0번 사용 (Tx1, Rx1)
    sdr.tx_lo = int(CENTER_FREQ)
    sdr.rx_lo = int(CENTER_FREQ)
    sdr.tx_cyclic_buffer = True 
    
    # Gain 설정
    sdr.tx_hardwaregain_chan0 = -10 
    sdr.rx_gain_mode_chan0 = "manual"
    sdr.rx_hardwaregain_chan0 = 30 
    
    print(">> Pluto+ SDR 설정 완료")

except Exception as e:
    print(f"SDR 연결 오류: {e}")
    exit()

# --- 3. FMCW 파형(Chirp) 생성 ---
# 선형 주파수 변조 신호 생성
t = np.arange(NUM_SAMPLES) / SAMPLE_RATE
# -BW/2 에서 +BW/2 로 주파수가 변하는 신호
f_chirp = np.linspace(-BANDWIDTH/2, BANDWIDTH/2, NUM_SAMPLES)
# 위상 적분하여 신호 생성 (I/Q 복소수 신호)
chirp_signal = np.exp(1j * np.pi * (BANDWIDTH/SAMPLE_RATE) * (t**2)) # 근사식
chirp_signal *= 2**14 # Pluto의 DAC 범위에 맞게 스케일링 (14비트)

# --- 4. 신호 송수신 및 처리 ---
# 송신 버퍼에 데이터 로드 (Cyclic Buffer로 인해 계속 반복 전송됨)
sdr.tx(chirp_signal)

# 그래프 초기화
plt.ion()
fig, ax = plt.subplots()
line, = ax.plot([], [])
ax.set_ylim(0, 100) # Y축 범위 (임의 설정)
ax.set_xlim(0, NUM_SAMPLES/2)
ax.set_xlabel('Range Bins')
ax.set_ylabel('Amplitude (dB)')
plt.title('Real-time Radar Magnitude Response')

print(">> 레이더 작동 시작... (Ctrl+C로 중지)")

try:
    while True:
        # 1. 데이터 수신 (Rx)
        rx_data = sdr.rx()
        
        # 2. 믹싱 (De-chirping)
        # 수신된 신호와 송신 신호의 켤레복소수를 곱함 (Beat Frequency 추출)
        # 주의: 실제로는 하드웨어 지연시간 보정이 필요할 수 있음
        mixed_signal = rx_data * np.conj(chirp_signal)
        
        # 3. FFT 처리 (주파수 도메인 변환 -> 거리 정보가 됨)
        fft_out = np.fft.fft(mixed_signal)
        fft_mag = 20 * np.log10(np.abs(fft_out)) # dB 스케일 변환
        
        # 4. 시각화 업데이트
        # DC 성분(0번 빈)과 누설 신호(Tx Leakage)가 강하므로 일부 앞쪽 빈을 무시할 수 있음
        line.set_xdata(np.arange(len(fft_mag)))
        line.set_ydata(fft_mag)
        fig.canvas.draw()
        fig.canvas.flush_events()
        
        time.sleep(0.05) # CPU 과부하 방지

except KeyboardInterrupt:
    # 종료 처리
    sdr.tx_destroy_buffer()
    del sdr
    print("\n>> 레이더 종료")