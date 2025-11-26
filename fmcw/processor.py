import numpy as np

class FMCWProcessor:
    """
    Updated FMCW Processor
    - cfg 기반 초기화
    """

    def __init__(self, cfg):
        self.cfg = cfg

        # ---------------------------
        # 기본 FMCW 설정
        # ---------------------------
        self.fs = float(cfg.sample_rate)
        self.B = float(cfg.B)
        self.T = float(cfg.T)

        # per-chirp sample count
        self.N = int(self.fs * self.T)

        # FFT 크기
        self.Nfft = int(cfg.fft_size)

        # Doppler chirp 수
        self.num_chirps = int(cfg.num_chirps)

        # guard bins
        self.guard = int(cfg.guard_bins)

        # speed of light
        self.c = 3e8

        # slope
        self.k = self.B / self.T

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
    # Beat mixing
    # ---------------------------
    def mix_to_beat(self, rx: np.ndarray, chirp: np.ndarray) -> np.ndarray:
        rx_seg = rx[: self.N]
        beat = rx_seg * np.conj(chirp)
        beat = beat - np.mean(beat)
        return beat

    # ---------------------------
    # Range FFT
    # ---------------------------
    def range_fft(self, beat: np.ndarray) -> np.ndarray:
        window = np.hanning(len(beat))
        spec = np.fft.fft(beat * window, n=self.Nfft)
        mag = np.abs(spec)

        if self.guard > 0:
            mag[: self.guard] = 0

        return mag

    # ---------------------------
    # Collect frame (Nchirps)
    # ---------------------------
    def collect_frame(self, pluto, chirp: np.ndarray):
        frame = np.zeros((self.num_chirps, self.Nfft), dtype=np.float32)

        for i in range(self.num_chirps):
            pluto.tx(chirp)
            rx = pluto.rx()

            beat = self.mix_to_beat(rx, chirp)
            rp = self.range_fft(beat)

            frame[i, :] = rp

        return frame

    # ---------------------------
    # Doppler FFT
    # ---------------------------
    def doppler_fft(self, frame: np.ndarray):
        frame_hp = frame - np.mean(frame, axis=0, keepdims=True)

        w = np.hanning(self.num_chirps)[:, None]
        frame_win = frame_hp * w

        doppler = np.fft.fft(frame_win, axis=0)
        doppler = np.fft.fftshift(doppler, axes=0)

        return np.abs(doppler)

    # ---------------------------
    # HUMAN PRESENCE DETECTION
    # (너가 준 최신 로직 통합)
    # ---------------------------
    def detect_human_presence(
        self,
        doppler_map: np.ndarray,
        range_gate_m=(0.5, 8.0),
        doppler_exclude_bins=1,
        threshold_scale=1.3,
        debug_once: bool = False,
    ):
        """
        return:
          detected(bool), energy(float), threshold(float), gate_bins(array)
        """

        N_range, N_dopp = doppler_map.shape
        r = self.range_axis_m

        r_min, r_max = range_gate_m
        gate_bins = np.where((r >= r_min) & (r <= r_max))[0]

        if gate_bins.size == 0:
            gate_bins = np.arange(N_range)

        if debug_once and not hasattr(self, "_dbg_printed"):
            print(f"[DEBUG] Range axis: {r[0]:.1f} ~ {r[-1]:.1f} m")
            print(f"[DEBUG] First 10 bins (m): {list(r[:10])}")
            print(f"[DEBUG] Gate {r_min}~{r_max} m -> bins {gate_bins[0]} ~ {gate_bins[-1]} (len={len(gate_bins)})")
            self._dbg_printed = True

        gate_spec = doppler_map[gate_bins, :]
        P = np.abs(gate_spec) ** 2

        mid = N_dopp // 2
        if doppler_exclude_bins > 0:
            ex = doppler_exclude_bins
            P[:, mid-ex:mid+ex+1] = 0.0

        # ----- Energy 정의 -----
        energy_now = float(np.percentile(P, 90))

        # ----- Background 갱신 -----
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