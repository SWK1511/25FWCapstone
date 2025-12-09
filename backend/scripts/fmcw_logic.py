import numpy as np
import time
import sys

try:
    import adi
    HAS_HARDWARE = True
except ImportError:
    HAS_HARDWARE = False


class FMCWDetector:
    def __init__(self, ip="ip:192.168.2.1"):
        self.SDR_IP = ip
        self.SAMPLE_RATE = 2_000_000
        self.CENTER_FREQ = 2_400_000_000
        self.BANDWIDTH = 50_000_000
        self.CHIRP_DURATION = 1e-4

        # 시각화 / 감지 파라미터
        self.MIN_DB_FOR_BAR = 80.0
        self.MAX_DB_FOR_BAR = 105.0
        self.ALPHA_PROFILE = 0.3
        self.ALPHA_RISE = 0.3
        self.ALPHA_FALL = 0.02

        # FFT/버퍼 설정
        self.N_SAMPLES = 1024
        self.NUM_CHIRPS = 128
        self.TOTAL_SAMPLES = self.N_SAMPLES * self.NUM_CHIRPS

        # 상태 변수
        self.sdr = None
        self.clutter_map = None
        self.smoothed_profile = np.zeros(self.N_SAMPLES)
        self.stable_peak_val = self.MIN_DB_FOR_BAR

    def connect(self):
        if not HAS_HARDWARE:
            # 하드웨어 없으면 시뮬레이션 모드로 그냥 True
            return True

        print(f">>> [FMCW] PlutoSDR({self.SDR_IP}) 연결 중...")
        try:
            self.sdr = adi.Pluto(self.SDR_IP)

            # 혹시 기존 버퍼가 살아있다면 정리
            try:
                self.sdr.tx_destroy_buffer()
            except:
                pass
            try:
                self.sdr.rx_destroy_buffer()
            except:
                pass

            # 기본 RF 설정
            self.sdr.sample_rate = int(self.SAMPLE_RATE)
            self.sdr.rx_lo = int(self.CENTER_FREQ)
            self.sdr.tx_lo = int(self.CENTER_FREQ)
            self.sdr.rx_rf_bandwidth = int(self.BANDWIDTH)
            self.sdr.tx_rf_bandwidth = int(self.BANDWIDTH)

            # RX 버퍼 크기 (한 프레임 = NUM_CHIRPS × N_SAMPLES)
            self.sdr.rx_buffer_size = int(self.TOTAL_SAMPLES)

            # 이득 설정
            self.sdr.gain_control_mode_chan0 = "manual"
            self.sdr.rx_hardwaregain_chan0 = 70
            self.sdr.tx_hardwaregain_chan0 = 0

            # FMCW chirp 생성
            t = np.arange(self.N_SAMPLES) / self.SAMPLE_RATE
            k = self.BANDWIDTH / self.CHIRP_DURATION  # sweep rate
            chirp = np.exp(1j * np.pi * k * t**2) * (2**14)
            tx_waveform = np.tile(chirp, self.NUM_CHIRPS)

            # ✅ 버퍼 생성 전에 cyclic 모드 설정
            self.sdr.tx_cyclic_buffer = True
            self.sdr.tx(tx_waveform)

            print("✅ [FMCW] 하드웨어 설정 완료")
            return True
        except Exception as e:
            print(f"❌ [FMCW] 연결 실패: {e}")
            self.sdr = None
            return False

    def calibrate(self):
        print(">>> [FMCW] 배경 학습 시작 (3초 대기)...")
        if not self.sdr:
            # 하드웨어 없으면 그냥 0으로 초기화
            self.clutter_map = np.zeros(self.N_SAMPLES)
            return

        time.sleep(2)

        clutter_sum = np.zeros(self.N_SAMPLES)
        for _ in range(20):
            try:
                rx = self.sdr.rx()
                if len(rx) != self.TOTAL_SAMPLES:
                    continue

                frame = rx.reshape(self.NUM_CHIRPS, self.N_SAMPLES)
                win = np.hanning(self.N_SAMPLES)
                fft_data = np.fft.fft(frame * win, axis=1)
                mag_data = np.abs(fft_data)

                clutter_sum += np.mean(mag_data, axis=0)
                time.sleep(0.01)
            except:
                continue

        self.clutter_map = clutter_sum / 20
        print(">>> [FMCW] 학습 완료!")

    def process_frame(self):
        try:
            # 1) 데이터 수신
            if self.sdr:
                rx = self.sdr.rx()
            else:
                rx = np.random.normal(0, 10, self.TOTAL_SAMPLES)

            if len(rx) != self.TOTAL_SAMPLES:
                return None

            # 2) 프레임 reshape & FFT
            frame = rx.reshape(self.NUM_CHIRPS, self.N_SAMPLES)
            win = np.hanning(self.N_SAMPLES)
            fft_data = np.fft.fft(frame * win, axis=1)
            raw_profile = np.mean(np.abs(fft_data), axis=0)

            # 3) 프로파일 smoothing
            self.smoothed_profile = (
                self.smoothed_profile * (1 - self.ALPHA_PROFILE)
                + raw_profile * self.ALPHA_PROFILE
            )

            # 4) 클러터 제거
            if self.clutter_map is not None:
                diff_profile = np.abs(self.smoothed_profile - self.clutter_map)
            else:
                diff_profile = self.smoothed_profile

            # 5) 유효 구간(양쪽 대칭 중 절반만 사용)
            valid_len = self.N_SAMPLES // 2
            valid_data = diff_profile[1:valid_len]
            diff_db = 20 * np.log10(np.maximum(valid_data, 1e-9))

            # 6) 피크 탐지 및 지수적 추적
            current_peak_idx = int(np.argmax(diff_db))
            current_peak_val = float(diff_db[current_peak_idx])

            if current_peak_val > self.stable_peak_val:
                # 상승은 빠르게
                self.stable_peak_val = (
                    self.stable_peak_val * (1 - self.ALPHA_RISE)
                    + current_peak_val * self.ALPHA_RISE
                )
            else:
                # 하강은 천천히
                self.stable_peak_val = (
                    self.stable_peak_val * (1 - self.ALPHA_FALL)
                    + current_peak_val * self.ALPHA_FALL
                )

            # 7) 감지 여부 & bar 비율
            is_detected = self.stable_peak_val >= self.MIN_DB_FOR_BAR
            ratio = (self.stable_peak_val - self.MIN_DB_FOR_BAR) / (
                self.MAX_DB_FOR_BAR - self.MIN_DB_FOR_BAR
            )
            if ratio < 0:
                ratio = 0.0
            if ratio > 1:
                ratio = 1.0

            # 8) 감지 안 된 상태에서 약한 신호가 계속 들어오면 clutter 업데이트
            if (
                not is_detected
                and current_peak_val < self.MIN_DB_FOR_BAR
                and self.clutter_map is not None
            ):
                self.clutter_map = self.clutter_map * 0.98 + self.smoothed_profile * 0.02

            return {
                "mode": "FMCW",
                "signal": diff_db.tolist(),
                "peak_val": float(self.stable_peak_val),
                "ratio": float(ratio),
                "is_detected": bool(is_detected),
                "peak_idx": int(current_peak_idx),
            }

        except Exception:
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