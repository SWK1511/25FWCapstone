import numpy as np
import time
import sys

# 하드웨어 라이브러리 체크
try:
    import adi
    HAS_HARDWARE = True
except ImportError:
    HAS_HARDWARE = False

class MotionDetector:
    def __init__(self, ip="ip:192.168.2.1"):
        # ==========================================
        # 1. 설정 (run_motion_cw.py 설정값 100% 동일)
        # ==========================================
        self.SDR_IP = ip
        self.THRESHOLD = 15.0       
        self.DETECT_LIMIT = 10.0    
        self.MAX_SCORE = 20.0       
        self.ADAPTATION_RATE = 0.05 
        
        self.sdr = None
        self.current_baseline = 0.0
        self.current_score = 0.0

    def connect(self):
        """PlutoSDR 연결 및 설정 (run_motion_cw.py 코드 그대로)"""
        if not HAS_HARDWARE:
            print("⚠️ [Mock] 하드웨어 없음 -> 가상 모드로 동작")
            return True

        print(f">>> PlutoSDR({self.SDR_IP}) 연결 및 설정 중...")
        try:
            self.sdr = adi.Pluto(self.SDR_IP)
            
            # CW 모드 설정 (동일)
            self.sdr.sample_rate = int(2e6)
            self.sdr.rx_lo = int(2400e6)
            self.sdr.tx_lo = int(2400e6)
            self.sdr.rx_rf_bandwidth = int(2e6)
            self.sdr.tx_rf_bandwidth = int(2e6)
            self.sdr.rx_buffer_size = 1024 * 16
            
            # 게인 설정 (동일: 60 / 0)
            self.sdr.gain_control_mode_chan0 = 'manual'
            self.sdr.rx_hardwaregain_chan0 = 60  
            self.sdr.tx_hardwaregain_chan0 = 0 
            self.sdr.tx_cyclic_buffer = True 
            
            # 송신 신호 생성 (동일)
            fs = int(self.sdr.sample_rate)
            t = np.arange(0, self.sdr.rx_buffer_size) / fs
            fc = 100000 
            tx_signal = np.exp(1j * 2 * np.pi * fc * t) * (2**14)
            self.sdr.tx(tx_signal)
            
            print("✅ PlutoSDR 연결 성공")
            return True
        except Exception as e:
            print(f"❌ 연결 실패: {e}")
            self.sdr = None
            return False

    def calibrate(self):
        """초기 캘리브레이션 (동일 로직)"""
        print(">>> 기준값 측정 중... (사람은 가만히 계세요!)")
        
        if not self.sdr:
            self.current_baseline = 500.0 # Mock 값
            return

        baseline_list = []
        for i in range(50):
            data = self.sdr.rx()
            energy = np.mean(np.abs(data))
            baseline_list.append(energy)
            time.sleep(0.01)

        self.current_baseline = np.mean(baseline_list)
        print(f">>> 기준값 설정 완료: {self.current_baseline:.2f}")

    def process_frame(self):
        """메인 루프의 '한 바퀴' 로직"""
        # 1. 데이터 수신
        if self.sdr:
            raw_data = self.sdr.rx()
        else:
            # Mock 데이터 (테스트용)
            noise = np.random.normal(500, 50, 4096)
            if np.random.rand() > 0.95: noise += 1000 
            raw_data = noise

        if len(raw_data) == 0: return None

        # 2. 에너지 계산 (동일)
        current_energy = np.mean(np.abs(raw_data))
        
        # 3. 변화량 계산 (동일)
        diff = abs(current_energy - self.current_baseline)
        
        # 4. 점수 판정 로직 (동일)
        if diff > self.THRESHOLD:
            # 변화량이 15.0을 넘으면 점수 추가
            self.current_score += 2.0 
        else:
            # 변화량이 적으면 점수 감소
            self.current_score -= 1.0
            
            # [적응형 기준값] (동일)
            if self.current_score <= 0:
                self.current_baseline = (self.current_baseline * (1 - self.ADAPTATION_RATE)) + (current_energy * self.ADAPTATION_RATE)

        # 점수 범위 제한 (동일)
        if self.current_score < 0: self.current_score = 0
        if self.current_score > self.MAX_SCORE: self.current_score = self.MAX_SCORE
            
        # 5. 결과 반환 (웹으로 보낼 데이터)
        is_detected = self.current_score > self.DETECT_LIMIT
        
        return {
            "signal": np.abs(raw_data)[::8].tolist(), # 그래프용
            "score": self.current_score,
            "max_score": self.MAX_SCORE,
            "is_detected": is_detected,
            "diff": diff,
            "baseline": self.current_baseline
        }

    def close(self):
        if self.sdr:
            self.sdr.tx_destroy_buffer()