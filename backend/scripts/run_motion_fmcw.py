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
        # 설정값 (run_motion_fmcw.py 원본)
        self.SDR_IP = ip
        self.SAMPLE_RATE = 2000000
        self.CENTER_FREQ = 2380000000
        self.BANDWIDTH = 50000000
        self.CHIRP_DURATION = 1e-4
        
        self.MIN_DB_FOR_BAR = 80.0
        self.MAX_DB_FOR_BAR = 105.0
        self.ALPHA_PROFILE = 0.3 
        self.ALPHA_RISE = 0.3
        self.ALPHA_FALL = 0.02 

        self.N_SAMPLES = 1024 
        self.NUM_CHIRPS = 128  
        
        self.sdr = None
        self.clutter_map = None
        self.smoothed_profile = np.zeros(self.N_SAMPLES)
        self.stable_peak_val = self.MIN_DB_FOR_BAR 

    def connect(self):
        if not HAS_HARDWARE:
            print("⚠️ [FMCW Mock] 하드웨어 없음")
            return True

        print(f">>> [FMCW] PlutoSDR({self.SDR_IP}) 연결 중...")
        try:
            self.sdr = adi.Pluto(self.SDR_IP)
            self.sdr.sample_rate = int(self.SAMPLE_RATE)
            self.sdr.rx_lo = int(self.CENTER_FREQ)
            self.sdr.tx_lo = int(self.CENTER_FREQ)
            self.sdr.rx_rf_bandwidth = int(self.BANDWIDTH)
            self.sdr.tx_rf_bandwidth = int(self.BANDWIDTH)
            self.sdr.rx_buffer_size = self.N_SAMPLES * self.NUM_CHIRPS
            self.sdr.gain_control_mode_chan0 = 'manual'
            self.sdr.rx_hardwaregain_chan0 = 70
            self.sdr.tx_hardwaregain_chan0 = 0
            self.sdr.tx_cyclic_buffer = True
            
            # 파형 송신
            t = np.arange(self.N_SAMPLES) / self.SAMPLE_RATE
            chirp = np.exp(1j * np.pi * (self.BANDWIDTH / self.CHIRP_DURATION) * t**2) * (2**14)
            tx_waveform = np.tile(chirp, self.NUM_CHIRPS)
            self.sdr.tx(tx_waveform)
            
            print("✅ [FMCW] 하드웨어 설정 완료")
            return True
        except Exception as e:
            print(f"❌ [FMCW] 연결 실패: {e}")
            self.sdr = None
            return False

    def calibrate(self):
        """배경 학습 (Clutter Map)"""
        print(">>> [FMCW] 배경 학습 시작 (3초 대기)...")
        
        if not self.sdr:
            time.sleep(1)
            self.clutter_map = np.zeros(self.N_SAMPLES)
            return

        time.sleep(2) # 안정화 대기

        clutter_sum = np.zeros(self.N_SAMPLES)
        for i in range(30):
            rx = self.sdr.rx()
            frame = rx.reshape(self.NUM_CHIRPS, self.N_SAMPLES)
            win = np.hanning(self.N_SAMPLES)
            fft_data = np.fft.fft(frame * win, axis=1)
            mag_data = np.abs(fft_data)
            clutter_sum += np.mean(mag_data, axis=0)
            time.sleep(0.01)
            
        self.clutter_map = clutter_sum / 30
        print(">>> [FMCW] 배경 학습 완료!")

    def process_frame(self):
        """한 프레임 처리"""
        # 1. 데이터 수신
        if self.sdr:
            rx = self.sdr.rx()
        else:
            # Mock Data (FMCW 흉내 - 특정 주파수 튐)
            noise = np.random.normal(0, 10, self.N_SAMPLES * self.NUM_CHIRPS)
            if np.random.rand() > 0.9: 
                # 가짜 타겟 신호 생성
                t = np.linspace(0, 100, self.N_SAMPLES * self.NUM_CHIRPS)
                target = np.sin(2 * np.pi * 0.1 * t) * 5000
                rx = noise + target
            else:
                rx = noise

        if len(rx) == 0: return None

        # 2. 신호 처리 (FFT)
        frame = rx.reshape(self.NUM_CHIRPS, self.N_SAMPLES)
        win = np.hanning(self.N_SAMPLES)
        fft_data = np.fft.fft(frame * win, axis=1)
        raw_profile = np.mean(np.abs(fft_data), axis=0)
        
        # 3. 필터링
        self.smoothed_profile = (self.smoothed_profile * (1 - self.ALPHA_PROFILE)) + (raw_profile * self.ALPHA_PROFILE)
        
        # 4. 배경 제거
        if self.clutter_map is not None:
            diff_profile = np.abs(self.smoothed_profile - self.clutter_map)
        else:
            diff_profile = self.smoothed_profile
        
        # 5. dB 변환 (유효 데이터만)
        valid_len = self.N_SAMPLES // 2
        valid_data = diff_profile[1 : valid_len]
        diff_db = 20 * np.log10(valid_data + 1e-9)
        
        # 6. 피크 찾기
        current_peak_idx = np.argmax(diff_db)
        current_peak_val = diff_db[current_peak_idx]
        
        # 7. 안정화
        if current_peak_val > self.stable_peak_val:
            self.stable_peak_val = (self.stable_peak_val * (1 - self.ALPHA_RISE)) + (current_peak_val * self.ALPHA_RISE)
        else:
            self.stable_peak_val = (self.stable_peak_val * (1 - self.ALPHA_FALL)) + (current_peak_val * self.ALPHA_FALL)

        # 8. 감지 판정
        is_detected = (self.stable_peak_val >= self.MIN_DB_FOR_BAR)

        # 9. 비율 계산
        ratio = (self.stable_peak_val - self.MIN_DB_FOR_BAR) / (self.MAX_DB_FOR_BAR - self.MIN_DB_FOR_BAR)
        if ratio < 0: ratio = 0
        if ratio > 1: ratio = 1

        # 배경 업데이트 (미감지 시)
        if not is_detected and current_peak_val < self.MIN_DB_FOR_BAR and self.clutter_map is not None:
             self.clutter_map = (self.clutter_map * 0.98) + (self.smoothed_profile * 0.02)

        return {
            "mode": "FMCW",
            # 그래프 데이터: dB 값 배열 전체 전송 (다운샘플링 없음)
            "signal": diff_db.tolist(), 
            "peak_val": float(self.stable_peak_val), 
            "ratio": ratio, 
            "is_detected": bool(is_detected),
            "peak_idx": int(current_peak_idx)
        }

    def close(self):
        if self.sdr:
            self.sdr.tx_destroy_buffer()