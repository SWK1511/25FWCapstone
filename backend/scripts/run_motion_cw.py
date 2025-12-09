import adi
import numpy as np
import sys
import time

# ==========================================
# 1. 설정 (Sensitivity Tuning)
# ==========================================
SDR_IP = "ip:192.168.2.1"

# [감도 조절]
# 아까 노이즈가 11.2였으므로, 그보다 높은 15.0을 넘어야만 움직임으로 인정합니다.
THRESHOLD = 15.0       
DETECT_LIMIT = 10.0    # 점수가 10점을 넘어야 "사람 있음" 판정
MAX_SCORE = 20.0       # 점수 최대치

# 기준값 자동 보정 속도 (사람 없을 때 기준값이 환경에 맞춰 변하는 속도)
ADAPTATION_RATE = 0.05 

# ==========================================
# 2. PlutoSDR 초기화
# ==========================================
print(f">>> PlutoSDR({SDR_IP}) 연결 및 설정 중...")
try:
    sdr = adi.Pluto(SDR_IP)
except Exception as e:
    print("❌ 연결 실패: 케이블을 확인하거나 IP를 확인하세요.")
    sys.exit()

sdr.sample_rate = int(2e6)
sdr.rx_lo = int(2380e6)
sdr.tx_lo = int(2380e6)
sdr.rx_rf_bandwidth = int(2e6)
sdr.tx_rf_bandwidth = int(2e6)
sdr.rx_buffer_size = 1024 * 16

# [수정] 게인을 50 -> 40으로 낮춤 (노이즈 감소 목적)
sdr.gain_control_mode_chan0 = 'manual'
sdr.rx_hardwaregain_chan0 = 60  
sdr.tx_hardwaregain_chan0 = 0 
sdr.tx_cyclic_buffer = True 


# 송신 신호 생성
fs = int(sdr.sample_rate)
t = np.arange(0, sdr.rx_buffer_size) / fs
fc = 100000 
tx_signal = np.exp(1j * 2 * np.pi * fc * t) * (2**14)
sdr.tx(tx_signal)

# ==========================================
# 3. 초기 캘리브레이션 (0점 잡기)
# ==========================================
print("\n" + "="*60)
print("   [ 스마트 모션 감지기 (노이즈 필터링 적용됨) ]")
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
print(">>> 감지 시작! (Ctrl+C로 종료)\n")

# ==========================================
# 4. 메인 루프
# ==========================================
current_score = 0.0

try:
    while True:
        # 1. 데이터 수신 및 에너지 계산
        data = sdr.rx()
        current_energy = np.mean(np.abs(data))
        
        # 2. 변화량 계산 (현재값 - 기준값)
        diff = abs(current_energy - current_baseline)
        
        # 3. 점수 판정 로직
        if diff > THRESHOLD:
            # 변화량이 15.0을 넘으면 점수 추가 (사람 움직임)
            current_score += 2.0 
        else:
            # 변화량이 적으면 점수 감소
            current_score -= 1.0
            
            # [적응형 기준값]
            # 사람이 없다고 판단될 때(점수 0 이하), 기준값을 현재 환경에 맞게 미세 조정
            if current_score <= 0:
                current_baseline = (current_baseline * (1 - ADAPTATION_RATE)) + (current_energy * ADAPTATION_RATE)

        # 점수 범위 제한
        if current_score < 0: current_score = 0
        if current_score > MAX_SCORE: current_score = MAX_SCORE
            
        # 4. 시각화
        is_detected = current_score > DETECT_LIMIT
        
        # 게이지 바 길이 설정
        bar_len = int(current_score * 2.0)
        if bar_len > 40: bar_len = 40
        
        bar = "█" * bar_len
        space = " " * (40 - bar_len)
        
        if is_detected:
            status = "🚨 DETECTED!"
            color = "\033[91m"     # 빨강
            bar_color = "\033[91m"
        else:
            status = "   Secure   "
            color = "\033[92m"     # 녹색
            bar_color = "\033[90m" # 회색
            
        reset = "\033[0m"
        
        # 출력 (변화량 diff가 THRESHOLD(15.0) 보다 큰지 확인하세요)
        info = f"기준:{current_baseline:5.1f} | 변화:{diff:4.1f} | 점수:{current_score:4.1f}"
        print(f"\r{color}[{status}]{reset} {info} |{bar_color}{bar}{space}{reset}|", end="")

        # 반응 속도 최적화를 위해 sleep 제거함

except KeyboardInterrupt:
    print("\n\n>>> 시스템을 종료합니다.")
finally:
    sdr.tx_destroy_buffer()