import adi
import numpy as np
import sys
import time

# ==========================================
# 1. 설정 (튜닝 영역)
# ==========================================
SDR_IP = "ip:192.168.2.1"

# [하드웨어 설정]
SAMPLE_RATE = 2000000   # 2MHz
CENTER_FREQ = 2380000000 # 2.4GHz
BANDWIDTH = 50000000    # 50MHz
CHIRP_DURATION = 1e-4   # 0.1ms

# [dB 기준값 설정]
# 이 값은 환경에 따라 조절하세요.
MIN_DB_FOR_BAR = 80.0   # 최소 감지 신호 (이보다 작으면 대기중)
MAX_DB_FOR_BAR = 105.0  # 최대 감지 신호 (가장 가까울 때)

# [게이지 안정화 필터 설정]
ALPHA_PROFILE = 0.3 
ALPHA_RISE = 0.3   # 게이지가 올라갈 때 속도
ALPHA_FALL = 0.02  # 게이지가 내려갈 때 속도 (떨림 방지)

# ==========================================
# 2. 초기화
# ==========================================
N_SAMPLES = 1024 
NUM_CHIRPS = 128  

# ==========================================
# 3. PlutoSDR 연결
# ==========================================
print(f">>> PlutoSDR({SDR_IP}) 연결 중...")
try:
    sdr = adi.Pluto(SDR_IP)
except Exception as e:
    print("❌ 연결 실패. IP나 케이블을 확인하세요.")
    sys.exit()

sdr.sample_rate = int(SAMPLE_RATE)
sdr.rx_lo = int(CENTER_FREQ)
sdr.tx_lo = int(CENTER_FREQ)
sdr.rx_rf_bandwidth = int(BANDWIDTH)
sdr.tx_rf_bandwidth = int(BANDWIDTH)
sdr.rx_buffer_size = N_SAMPLES * NUM_CHIRPS
sdr.gain_control_mode_chan0 = 'manual'
sdr.rx_hardwaregain_chan0 = 70
sdr.tx_hardwaregain_chan0 = 0  
sdr.tx_cyclic_buffer = True    

# ==========================================
# 4. 파형 송신
# ==========================================
# FMCW 파형 생성
t = np.arange(N_SAMPLES) / SAMPLE_RATE
chirp = np.exp(1j * np.pi * (BANDWIDTH / CHIRP_DURATION) * t**2) * (2**14)
tx_waveform = np.tile(chirp, NUM_CHIRPS)
sdr.tx(tx_waveform)

# ==========================================
# 5. 배경 학습
# ==========================================
print("\n>>> 안정화 대기 중... (2초)")
time.sleep(2)
print(">>> 배경 학습 중... (주변을 비워주세요)")

clutter_map = np.zeros(N_SAMPLES)
for i in range(30):
    rx = sdr.rx()
    frame = rx.reshape(NUM_CHIRPS, N_SAMPLES)
    win = np.hanning(N_SAMPLES)
    fft_data = np.fft.fft(frame * win, axis=1)
    mag_data = np.abs(fft_data)
    clutter_map += np.mean(mag_data, axis=0)
    time.sleep(0.01)

clutter_map /= 30
print(">>> 학습 완료! 시작합니다.")
print("------------------------------------------------------------")

smoothed_profile = np.zeros(N_SAMPLES)
stable_peak_val = MIN_DB_FOR_BAR 

# ==========================================
# 6. 메인 루프
# ==========================================
try:
    while True:
        t0 = time.time()
        
        # 1. 데이터 수신 및 FFT
        rx = sdr.rx()
        frame = rx.reshape(NUM_CHIRPS, N_SAMPLES)
        win = np.hanning(N_SAMPLES)
        fft_data = np.fft.fft(frame * win, axis=1)
        raw_profile = np.mean(np.abs(fft_data), axis=0)
        
        # 2. 프로파일 필터링
        smoothed_profile = (smoothed_profile * (1 - ALPHA_PROFILE)) + (raw_profile * ALPHA_PROFILE)
        
        # 3. 배경 제거
        diff_profile = np.abs(smoothed_profile - clutter_map)
        
        # 4. dB 변환
        valid_data = diff_profile[1 : N_SAMPLES//2]
        diff_db = 20 * np.log10(valid_data + 1e-9)
        
        # 5. 피크 찾기
        current_peak_idx = np.argmax(diff_db) 
        current_peak_val = diff_db[current_peak_idx]    
        
        # [참고] FFT 기반 거리 계산은 USB 지연으로 인해 생략합니다.
        
        # ---------------------------------------------------------
        # [안정화 필터]
        # ---------------------------------------------------------
        if current_peak_val > stable_peak_val:
            # 값이 커질 때 (반응 속도 조절)
            stable_peak_val = (stable_peak_val * (1 - ALPHA_RISE)) + (current_peak_val * ALPHA_RISE)
        else:
            # 값이 작아질 때 (떨림 방지)
            stable_peak_val = (stable_peak_val * (1 - ALPHA_FALL)) + (current_peak_val * ALPHA_FALL)

        # ---------------------------------------------------------
        # [화면 출력 로직]
        # ---------------------------------------------------------
        
        # 1. 감지 판정
        is_detected = (stable_peak_val >= MIN_DB_FOR_BAR)

        # 2. 상태 메시지 및 게이지 계산
        if not is_detected:
            # [대기중 상태]
            status = "\033[90m⏳ 대기중..\033[0m" # 회색
            msg_label = "      "
            bar_str = ""
            space_str = " " * 30
            
            # 배경 업데이트 (미감지 시에만)
            if current_peak_val < MIN_DB_FOR_BAR:
                 clutter_map = (clutter_map * 0.98) + (smoothed_profile * 0.02)
        else:
            # [감지됨 상태]
            status = "\033[91m🚨 감지됨!\033[0m" # 빨강
            
            # 비율 계산 (0.0 ~ 1.0)
            ratio = (stable_peak_val - MIN_DB_FOR_BAR) / (MAX_DB_FOR_BAR - MIN_DB_FOR_BAR)
            if ratio > 1.0: ratio = 1.0
            
            # [거리 라벨링] 신호 강도에 따른 텍스트 표시
            if ratio >= 0.9:
                msg_label = "\033[91m⚠️ 초근접!!\033[0m" # 빨간색 강조
            elif ratio >= 0.5:
                msg_label = "가까움"
            else:
                msg_label = "멀리 있음"
            
            # [게이지]
            bar_len = int(ratio * 30)
            if bar_len < 1: bar_len = 1
            if bar_len > 30: bar_len = 30
            
            bar_str = "█" * bar_len
            space_str = " " * (30 - bar_len)

        # 3. 최종 출력 (거리 숫자 대신 상태 라벨 출력)
        # 라벨 출력 시 문자열 길이를 맞추기 위해 탭(\t)이나 고정폭 사용을 고려할 수 있으나,
        # 여기서는 간단히 배치합니다.
        sys.stdout.write(f"\r{status} | 상태: {msg_label:10s} | 강도:{stable_peak_val:5.1f}dB | [{bar_str}{space_str}]   ")
        sys.stdout.flush()

except KeyboardInterrupt:
    print("\n\n>>> 종료합니다.")
finally:
    sdr.tx_destroy_buffer()