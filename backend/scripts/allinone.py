# 파일명: jetson_sdr_client.py (젯슨나노에서 실행)
import adi
import numpy as np
import sys
import time
import socket  # [추가] 통신 라이브러리

# ==========================================
# 1. 설정 및 통신 준비
# ==========================================
# ★ 중요: 여기에 라즈베리파이 IP 주소를 적으세요 ★
RPI_IP = "10.204.220.184"  # 예: "192.168.0.15" (따옴표 필수)
RPI_PORT = 5005          # 라즈베리파이 코드와 같은 포트 번호

SDR_IP = "ip:192.168.2.1"
THRESHOLD = 15.0 
DETECT_LIMIT = 10.0 
MAX_SCORE = 20.0 
ADAPTATION_RATE = 0.05 

# [추가] 통신 소켓 생성 (우체부 준비)
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# ==========================================
# 2. PlutoSDR 초기화 (기존 코드 동일)
# ==========================================
print(f">>> PlutoSDR({SDR_IP}) 연결 및 설정 중...")
try:
    sdr = adi.Pluto(SDR_IP)
except Exception as e:
    print("❌ 연결 실패: 케이블을 확인하거나 IP를 확인하세요.")
    sys.exit()

sdr.sample_rate = int(2e6)
sdr.rx_lo = int(2400e6)
sdr.tx_lo = int(2400e6)
sdr.rx_rf_bandwidth = int(2e6)
sdr.tx_rf_bandwidth = int(2e6)
sdr.rx_buffer_size = 1024 * 16

sdr.gain_control_mode_chan0 = 'manual'
sdr.rx_hardwaregain_chan0 = 60  
sdr.tx_hardwaregain_chan0 = 0 
sdr.tx_cyclic_buffer = True 

fs = int(sdr.sample_rate)
t = np.arange(0, sdr.rx_buffer_size) / fs
fc = 100000 
tx_signal = np.exp(1j * 2 * np.pi * fc * t) * (2**14)
sdr.tx(tx_signal)

# ==========================================
# 3. 초기 캘리브레이션
# ==========================================
print("\n" + "="*60)
print("   [ 젯슨나노 -> 라즈베리파이 원격 제어 ]")
print("="*60)
print(">>> 안정화 대기 중... (3초)")
time.sleep(3)

print(">>> 기준값 측정 중... (사람은 가만히 계세요!)")
baseline_list = []
for i in range(50):
    data = sdr.rx()
    energy = np.mean(np.abs(data))
    baseline_list.append(energy)
    time.sleep(0.01)

current_baseline = np.mean(baseline_list)
print(f">>> 기준값 설정 완료: {current_baseline:.2f}")
print(f">>> 라즈베리파이({RPI_IP})로 신호를 전송합니다.")

# ==========================================
# 4. 메인 루프
# ==========================================
current_score = 0.0

try:
    while True:
        # 데이터 수신 및 에너지 계산
        data = sdr.rx()
        current_energy = np.mean(np.abs(data))
        diff = abs(current_energy - current_baseline)
        
        # 점수 판정 로직
        if diff > THRESHOLD:
            current_score += 2.0 
        else:
            current_score -= 1.0
            if current_score <= 0:
                current_baseline = (current_baseline * (1 - ADAPTATION_RATE)) + (current_energy * ADAPTATION_RATE)

        if current_score < 0: current_score = 0
        if current_score > MAX_SCORE: current_score = MAX_SCORE
            
        is_detected = current_score > DETECT_LIMIT
        
        # === [추가된 부분: 라즈베리파이로 신호 쏘기] ===
        if is_detected:
            msg = "DETECTED"
            status = "🚨 DETECTED!"
            color = "\033[91m"
            bar_color = "\033[91m"
        else:
            msg = "SECURE"
            status = "   Secure   "
            color = "\033[92m"
            bar_color = "\033[90m"

        # 라즈베리파이 IP로 메시지 전송 (한 줄로 끝!)
        sock.sendto(msg.encode(), (RPI_IP, RPI_PORT))
        
        # 화면 출력 (기존 시각화 유지)
        bar_len = int(current_score * 2.0)
        if bar_len > 40: bar_len = 40
        bar = "█" * bar_len
        space = " " * (40 - bar_len)
        reset = "\033[0m"
        
        info = f"전송중:{msg} | 점수:{current_score:4.1f}"
        print(f"\r{color}[{status}]{reset} {info} |{bar_color}{bar}{space}{reset}|", end="")

except KeyboardInterrupt:
    print("\n\n>>> 시스템을 종료합니다.")
finally:
    sdr.tx_destroy_buffer()
    sock.close()