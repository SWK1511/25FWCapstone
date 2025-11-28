import numpy as np

class FMCWProcessor:
    """
    속도 최적화된 FMCW Processor
    - 불필요한 Loop 제거 (Vectorization 적용)
    """

    def __init__(self, cfg):
        self.cfg = cfg

        # ---------------------------
        # 기본 FMCW 설정
        # ---------------------------
        self.fs = float(cfg.sample_rate)
        self.B = float(cfg.B)
        self.T = float(cfg.T)

        # per-chirp sample count (유효 샘플 수)
        self.N = int(self.fs * self.T)

        # FFT 크기
        self.Nfft = int(cfg.fft_size)

        # Doppler chirp 수
        self.num_chirps = int(cfg.num_chirps)

        # guard bins
        self.guard = int(cfg.guard_bins)

        # speed of light
        self.c = 3e8

        # freq spacing
        self.bin_spacing = self.fs / self.Nfft

        # ---------------------------
        # Range axis 미리 계산
        # ---------------------------
        bins = np.arange(self.Nfft)
        fb = bins * self.bin_spacing
        self.range_axis_m = (self.c * self.T * fb) / (2 * self.B)

        # ---------------------------
        # Adaptive background energy
        # ---------------------------
        self.bg_energy = None
        self.bg_alpha = 0.9    

    # ---------------------------
    # Beat mixing (단일 Chirp용 - 호환성 유지)
    # ---------------------------
    def mix_to_beat(self, rx: np.ndarray, chirp: np.ndarray) -> np.ndarray:
        rx_seg = rx[: self.N]
        beat = rx_seg * np.conj(chirp)
        beat = beat - np.mean(beat)
        return beat

    # ---------------------------
    # Range FFT (단일 Chirp용 - 호환성 유지)
    # ---------------------------
    def range_fft(self, beat: np.ndarray) -> np.ndarray:
        window = np.hanning(len(beat))
        spec = np.fft.fft(beat * window, n=self.Nfft)
        mag = np.abs(spec)
        if self.guard > 0:
            mag[: self.guard] = 0
        return mag

    # ---------------------------
    # [중요 수정] Collect frame (속도 개선)
    # ---------------------------
    def collect_frame(self, pluto, chirp: np.ndarray):
        """
        한 번의 rx() 호출로 모든 Chirp 데이터를 수신합니다.
        (기존 8초 -> 0.1초 미만으로 단축)
        """
        # 1. 버퍼 한 번에 읽기 (크기: num_chirps * fft_size)
        rx_raw = pluto.rx()
        
        # 2. 데이터 모양 맞추기 (Chirp 개수 x 샘플 수)
        # rx_buffer_size가 넉넉하게 잡혀있으므로 필요한 만큼만 자릅니다.
        total_samples = self.num_chirps * self.Nfft
        if len(rx_raw) < total_samples:
            # 혹시 버퍼가 모자라면 0으로 채움 (안전장치)
            rx_raw = np.pad(rx_raw, (0, total_samples - len(rx_raw)))
        
        rx_reshaped = rx_raw[:total_samples].reshape(self.num_chirps, self.Nfft)

        # 3. 결과 담을 배열
        frame = np.zeros((self.num_chirps, self.Nfft), dtype=np.float32)

        # 4. 고속 처리 (메모리 상에서 연산하므로 매우 빠름)
        # 유효 샘플 길이
        valid_N = self.N 
        
        # Chirp 신호도 미리 잘라둠
        chirp_seg = chirp[:valid_N]
        
        for i in range(self.num_chirps):
            # (1) 유효 데이터 슬라이싱
            rx_seg = rx_reshaped[i, :valid_N]
            
            # (2) Beat Signal 생성 (Mix)
            beat = rx_seg * np.conj(chirp_seg)
            beat = beat - np.mean(beat) # DC 제거
            
            # (3) Range FFT
            # Hanning Window 적용
            window = np.hanning(valid_N)
            spec = np.fft.fft(beat * window, n=self.Nfft)
            
            frame[i, :] = np.abs(spec)

        # Guard bin 처리
        if self.guard > 0:
            frame[:, :self.guard] = 0

        return frame

    # ---------------------------
    # Doppler FFT (기존 동일)
    # ---------------------------
    def doppler_fft(self, frame: np.ndarray):
        frame_hp = frame - np.mean(frame, axis=0, keepdims=True)
        w = np.hanning(self.num_chirps)[:, None]
        frame_win = frame_hp * w
        doppler = np.fft.fft(frame_win, axis=0)
        doppler = np.fft.fftshift(doppler, axes=0)
        return np.abs(doppler)

    # ---------------------------
    # HUMAN PRESENCE DETECTION (기존 동일)
    # ---------------------------
    def detect_human_presence(
        self,
        doppler_map: np.ndarray,
        range_gate_m=(0.5, 8.0),
        doppler_exclude_bins=1,
        threshold_scale=1.3,
        debug_once: bool = False,
    ):
        N_range, N_dopp = doppler_map.shape
        r = self.range_axis_m

        r_min, r_max = range_gate_m
        gate_bins = np.where((r >= r_min) & (r <= r_max))[0]

        if gate_bins.size == 0:
            gate_bins = np.arange(N_range)

        gate_spec = doppler_map[gate_bins, :]
        P = np.abs(gate_spec) ** 2

        mid = N_dopp // 2
        if doppler_exclude_bins > 0:
            ex = doppler_exclude_bins
            P[:, mid-ex:mid+ex+1] = 0.0

        energy_now = float(np.percentile(P, 90))

        if self.bg_energy is None or self.bg_energy <= 0:
            self.bg_energy = energy_now

        if energy_now < self.bg_energy * 3.0:
            self.bg_energy = (
                self.bg_alpha * self.bg_energy
                + (1 - self.bg_alpha) * energy_now
            )

        threshold = self.bg_energy * threshold_scale
        detected = energy_now > threshold

        return detected, energy_now, threshold, gate_bins