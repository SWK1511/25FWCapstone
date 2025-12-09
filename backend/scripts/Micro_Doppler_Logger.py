import adi
import numpy as np
import sys
import time
import csv
import os
from collections import deque

# SSH 환경에서 그래프 저장을 위해 백엔드 설정 (창 안 띄움)
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# 운영체제에 따른 논블로킹 키보드
if sys.platform == 'win32':
    import msvcrt
    def kbhit(): return msvcrt.kbhit()
    def getch(): return msvcrt.getch().decode('utf-8')
else:
    import termios, tty, select
    def kbhit():
        dr,dw,de = select.select([sys.stdin], [], [], 0)
        return dr != []
    def getch():
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        return ch

# ==========================================
# 1. 설정 (run_motion_fmcw.py 참고)
# ==========================================
SDR_IP = "ip:192.168.2.1"
SAMPLE_RATE = 2000000     # 2MHz
CENTER_FREQ = 2400000000  # 2.4GHz
BANDWIDTH = 50000000      # 50MHz (변경됨)
CHIRP_DURATION = 1e-4     # 0.1ms (변경됨)

NUM_CHIRPS = 128          # 속도(Doppler) 축 (변경됨)
N_SAMPLES = 1024          # 거리(Range) 축

# 데이터 저장 파일명
DATA_FILENAME = "micro_doppler_walk_stand.csv"

# 스냅샷 저장용 버퍼 설정
TIME_WINDOW = 100
spectrogram_buffer = np.zeros((NUM_CHIRPS, TIME_WINDOW))

# ==========================================
# 2. PlutoSDR 연결
# ==========================================
print(f">>> PlutoSDR({SDR_IP}) 연결 중...")
try:
    sdr = adi.Pluto(SDR_IP)
    sdr.sample_rate = int(SAMPLE_RATE)
    sdr.rx_lo = int(CENTER_FREQ)
    sdr.tx_lo = int(CENTER_FREQ)
    sdr.rx_rf_bandwidth = int(BANDWIDTH)
    sdr.tx_rf_bandwidth = int(BANDWIDTH)
    sdr.rx_buffer_size = N_SAMPLES * NUM_CHIRPS
    sdr.gain_control_mode_chan0 = 'manual'
    sdr.rx_hardwaregain_chan0 = 70 # 게인 70으로 설정
    sdr.tx_hardwaregain_chan0 = 0
    sdr.tx_cyclic_buffer = True

    t = np.arange(N_SAMPLES) / SAMPLE_RATE
    chirp = np.exp(1j * np.pi * (BANDWIDTH / CHIRP_DURATION) * t**2) * (2**14)
    tx_waveform = np.tile(chirp, NUM_CHIRPS)
    sdr.tx(tx_waveform)
    
except Exception as e:
    print("❌ 연결 실패.")
    sys.exit()

# 배경 학습
print(">>> 배경 학습 중 (3초)...")
clutter_avg = np.zeros((NUM_CHIRPS, N_SAMPLES), dtype=complex)
for _ in range(30):
    rx = sdr.rx()
    frame = rx.reshape(NUM_CHIRPS, N_SAMPLES)
    clutter_avg += frame
clutter_avg /= 30
print(">>> 준비 완료!")

# ==========================================
# 3. CSV 초기화
# ==========================================
if not os.path.exists(DATA_FILENAME):
    with open(DATA_FILENAME, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = ["label"] + [f"doppler_{i}" for i in range(NUM_CHIRPS)]
        writer.writerow(header)

print("------------------------------------------------------------")
print(" [SSH용 마이크로 도플러 수집기] ")
print(" - '0' 누르면: [걷기 Walking] 2초간 녹화")
print(" - '1' 누르면: [서 있기 Standing] 2초간 녹화")
print(" - 's' 누르면: 현재 스펙트로그램을 그림파일(PNG)로 저장")
print(" - 'q' 누르면: 종료")
print("------------------------------------------------------------")

# 녹화 제어 변수
recording_label = None
recording_frames_left = 0
labels = ["Walking", "Standing"]

try:
    while True:
        # 1. 데이터 수신 & 배경 제거
        rx = sdr.rx()
        frame = rx.reshape(NUM_CHIRPS, N_SAMPLES)
        frame = frame - clutter_avg 
        
        # 2. 2D FFT (Range-Doppler)
        range_fft = np.fft.fft(frame, axis=1)
        doppler_fft = np.fft.fft(range_fft, axis=0)
        doppler_shifted = np.fft.fftshift(doppler_fft, axes=0)
        
        # 3. 마이크로 도플러 추출 (Range 축 Sum)
        mag_data = np.abs(doppler_shifted)
        velocity_profile = np.sum(mag_data, axis=1)
        velocity_profile_db = 20 * np.log10(velocity_profile + 1e-9)
        
        # 4. 스냅샷용 버퍼 업데이트
        spectrogram_buffer = np.roll(spectrogram_buffer, -1, axis=1)
        spectrogram_buffer[:, -1] = velocity_profile_db
        
        # 5. 움직임 강도 계산 (터미널 표시용)
        # 도플러 맵 전체 에너지의 평균을 대략적인 '움직임'으로 표시
        motion_energy = np.mean(velocity_profile_db)
        # 게이지 정규화 (대략 80~120dB 사이)
        bar_len = int((motion_energy - 80) / 40 * 20)
        if bar_len < 0: bar_len = 0
        if bar_len > 20: bar_len = 20
        bar_str = "█" * bar_len + " " * (20 - bar_len)

        # 6. 상태 메시지 및 녹화
        status = " [대기] "
        
        if recording_frames_left > 0:
            status = f"\033[91m [REC: {labels[recording_label]}]\033[0m"
            
            # 파일 저장
            with open(DATA_FILENAME, mode='a', newline='') as f:
                writer = csv.writer(f)
                row = [recording_label] + velocity_profile_db.tolist()
                writer.writerow(row)
            
            recording_frames_left -= 1
            if recording_frames_left == 0:
                print("\n>>> 녹화 완료!")

        # 터미널 출력
        sys.stdout.write(f"\r{status} 움직임 강도: {motion_energy:5.1f}dB [{bar_str}] [0:걷기/1:정지/s:캡쳐/q:종료]  ")
        sys.stdout.flush()

        # 7. 키보드 입력
        if kbhit():
            key = getch()
            
            if key == 'q':
                break
            
            elif key in ['0', '1']:
                recording_label = int(key)
                recording_frames_left = 60 # 약 2초
                print(f"\n>>> 녹화 시작: {labels[recording_label]}...")
            
            elif key == 's':
                print("\n>>> 스펙트로그램 캡쳐 중...")
                plt.figure(figsize=(10, 5))
                plt.imshow(spectrogram_buffer, aspect='auto', cmap='jet', origin='lower')
                plt.title("Snapshot: Micro-Doppler Spectrogram")
                plt.ylabel("Velocity (Doppler)")
                plt.xlabel("Time")
                plt.colorbar()
                plt.savefig("spectrogram_snapshot.png")
                plt.close()
                print(">>> 저장됨: spectrogram_snapshot.png")

except KeyboardInterrupt:
    print("\n종료합니다.")
finally:
    sdr.tx_destroy_buffer()