import numpy as np
import time
import sys

try:
    import adi
    HAS_HARDWARE = True
except ImportError:
    HAS_HARDWARE = False


class MotionDetector:
    def __init__(self, ip="ip:192.168.2.1"):
        self.SDR_IP = ip
        self.THRESHOLD = 15.0       # 감지 민감도
        self.DETECT_LIMIT = 10.0    # 감지 판정 점수
        self.MAX_SCORE = 20.0       # 점수 최대값
        self.ADAPTATION_RATE = 0.05 # baseline 적응 비율

        self.sdr = None
        self.current_baseline = 0.0
        self.current_score = 0.0

    def connect(self):
        if not HAS_HARDWARE:
            return True

        print(f">>> [CW] PlutoSDR({self.SDR_IP}) 연결 중...")
        try:
            self.sdr = adi.Pluto(self.SDR_IP)

            # 혹시 남아 있을지 모르는 이전 버퍼 제거
            try:
                self.sdr.tx_destroy_buffer()
            except:
                pass
            try:
                self.sdr.rx_destroy_buffer()
            except:
                pass

            # CW 설정
            self.sdr.sample_rate = int(2e6)
            self.sdr.rx_lo = int(2400e6)
            self.sdr.tx_lo = int(2400e6)
            self.sdr.rx_rf_bandwidth = int(2e6)
            self.sdr.tx_rf_bandwidth = int(2e6)
            self.sdr.rx_buffer_size = 1024 * 16

            self.sdr.gain_control_mode_chan0 = "manual"
            self.sdr.rx_hardwaregain_chan0 = 60
            self.sdr.tx_hardwaregain_chan0 = 0

            fs = int(self.sdr.sample_rate)
            t = np.arange(0, self.sdr.rx_buffer_size) / fs
            fc = 100000
            tx_signal = np.exp(1j * 2 * np.pi * fc * t) * (2**14)

            # 전송 시작 전에 cyclic 모드 설정
            self.sdr.tx_cyclic_buffer = True
            self.sdr.tx(tx_signal)

            print("✅ [CW] 하드웨어 설정 완료")
            return True
        except Exception as e:
            print(f"❌ [CW] 연결 실패: {e}")
            self.sdr = None
            return False

    def calibrate(self):
        print(">>> [CW] 기준값 측정 중... (3초)")
        if not self.sdr:
            self.current_baseline = 500.0
            return

        baseline_list = []
        for i in range(50):
            try:
                data = self.sdr.rx()
                energy = np.mean(np.abs(data))
                baseline_list.append(energy)
                time.sleep(0.01)
            except:
                continue

        self.current_baseline = np.mean(baseline_list)
        print(f">>> [CW] 측정 완료: {self.current_baseline:.2f}")

    def process_frame(self):
        try:
            if self.sdr:
                raw_data = self.sdr.rx()
            else:
                noise = np.random.normal(500, 50, 4096)
                if np.random.rand() > 0.95:
                    noise += 1000
                raw_data = noise

            if len(raw_data) == 0:
                return None

            current_energy = np.mean(np.abs(raw_data))
            diff = abs(current_energy - self.current_baseline)

            if diff > self.THRESHOLD:
                self.current_score += 2.0
            else:
                self.current_score -= 1.0
                if self.current_score <= 0:
                    self.current_baseline = (
                        self.current_baseline * (1 - self.ADAPTATION_RATE)
                        + current_energy * self.ADAPTATION_RATE
                    )

            if self.current_score < 0:
                self.current_score = 0
            if self.current_score > self.MAX_SCORE:
                self.current_score = self.MAX_SCORE

            is_detected = self.current_score > self.DETECT_LIMIT

            return {
                "signal": np.abs(raw_data)[::8].tolist(),
                "score": self.current_score,
                "max_score": self.MAX_SCORE,
                "is_detected": bool(is_detected),
                "diff": diff,
                "baseline": self.current_baseline,
            }
        except:
            return None

    def close(self):
        if self.sdr:
            try:
                self.sdr.tx_destroy_buffer()
            except:
                pass
            try:
                self.sdr.rx_destroy_buffer()
            except:
                pass
            try:
                del self.sdr
            except:
                pass
            self.sdr = None