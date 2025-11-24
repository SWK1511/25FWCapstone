'''	•	collect_frame(pluto, chirp) → chirp 128번 쏘고, 각 chirp의 range profile을 쌓아서 프레임 생성
	•	doppler_fft(frame) → 프레임을 시간축 FFT 해서 도플러 맵 생성
(사람이 움직이면 도플러 성분이 강하게 나옴) •	detect_human_presence(doppler_map, range_gate_m=(0,8)) → 0~8m 구간에서 DC(정적 성분) 빼고
도플러 에너지 증가하면 사람 존재 True'''
import numpy as np
from .config import FMCWConfig

class FMCWProcessor:
    """
    FMCW Processor for HUMAN PRESENCE (motion) detection.

    Pipeline:
      - beat mixing (fast-time)
      - range FFT -> range profile per chirp
      - stack profiles over num_chirps (slow-time)
      - doppler FFT -> doppler map
      - detect presence by doppler energy in near-range gate
    """

    def __init__(self, cfg: FMCWConfig):
        self.cfg = cfg

        self.fs = float(cfg.sample_rate)
        self.B = float(cfg.B)
        self.T = float(cfg.T)
        self.N = int(self.T * self.fs)  # samples per chirp

        self.num_chirps = int(cfg.num_chirps)
        self.Nfft = int(cfg.fft_size)
        self.guard = int(cfg.guard_bins)

        self.c = 3e8  # speed of light

        # frequency bin spacing for range FFT
        self.bin_spacing = self.fs / self.Nfft

        # sweep slope
        self.k = self.B / self.T

    # ---------------------------
    # 1) Beat mixing
    # ---------------------------
    def mix_to_beat(self, rx: np.ndarray, chirp: np.ndarray) -> np.ndarray:
        """
        beat[n] = rx[n] * conj(chirp[n])
        """
        rx_seg = rx[: self.N]
        beat = rx_seg * np.conj(chirp)
        beat = beat - np.mean(beat)  # DC removal
        return beat

    # ---------------------------
    # 2) Range FFT per chirp
    # ---------------------------
    def range_fft(self, beat: np.ndarray) -> np.ndarray:
        """
        FFT magnitude of beat signal.
        """
        window = np.hanning(len(beat))
        spec = np.fft.fft(beat * window, n=self.Nfft)
        mag = np.abs(spec)

        # guard DC bins if needed
        if self.guard > 0:
            mag[: self.guard] = 0

        return mag

    def estimate_range_axis(self):
        """
        Returns range value (meters) for each FFT bin.
        R = (c*T/(2B)) * f_b
        f_b = bin * bin_spacing
        """
        bins = np.arange(self.Nfft)
        fb = bins * self.bin_spacing
        R = (self.c * self.T * fb) / (2 * self.B)
        return R

    # ---------------------------
    # 3) Frame collection helper
    # ---------------------------
    def collect_frame(self, pluto, chirp: np.ndarray):
        """
        Collect num_chirps range profiles into a frame.

        Returns:
          frame: shape (num_chirps, Nfft)
        """
        frame = np.zeros((self.num_chirps, self.Nfft), dtype=np.float32)

        for i in range(self.num_chirps):
            pluto.tx(chirp)
            rx = pluto.rx()

            beat = self.mix_to_beat(rx, chirp)
            rp = self.range_fft(beat)

            frame[i, :] = rp

        return frame

    # ---------------------------
    # 4) Doppler processing
    # ---------------------------
    def doppler_fft(self, frame: np.ndarray):
        """
        Slow-time FFT over chirps to get Doppler map.

        Input:
          frame: (num_chirps, Nfft)

        Output:
          doppler_map: (num_chirps, Nfft) (magnitude, FFT-shifted)
        """
        # remove static clutter (mean over time)
        frame_hp = frame - np.mean(frame, axis=0, keepdims=True)

        # slow-time window
        w = np.hanning(self.num_chirps)[:, None]
        frame_win = frame_hp * w

        doppler = np.fft.fft(frame_win, axis=0)
        doppler = np.fft.fftshift(doppler, axes=0)

        doppler_map = np.abs(doppler)
        return doppler_map

    # ---------------------------
    # 5) Human presence detection
    # ---------------------------
    def detect_human_presence(
        self,
        doppler_map: np.ndarray,
        range_gate_m=(0.0, 8.0),
        doppler_exclude_bins=3,
        threshold_scale=6.0,
    ):
        """
        Detect motion-like human presence by Doppler energy in near-range gate.

        Args:
          doppler_map: (num_chirps, Nfft)
          range_gate_m: meters range window to inspect (e.g., 0~8m)
          doppler_exclude_bins: center bins to ignore (static/near-DC Doppler)
          threshold_scale: how strict detection is (higher = fewer false alarms)

        Returns:
          detected (bool)
          energy (float)
          threshold (float)
          gate_bins (tuple)
        """
        R_axis = self.estimate_range_axis()

        # find bin indices for range gate
        r0, r1 = range_gate_m
        gate = np.where((R_axis >= r0) & (R_axis <= r1))[0]
        if len(gate) < 2:
            return False, 0.0, 0.0, (0, 0)

        g_start, g_end = int(gate[0]), int(gate[-1])

        # select gate region
        sub = doppler_map[:, g_start : g_end + 1]

        # exclude near-zero Doppler bins (static)
        mid = self.num_chirps // 2
        sub_dc_removed = sub.copy()
        sub_dc_removed[mid - doppler_exclude_bins : mid + doppler_exclude_bins + 1, :] = 0

        # energy in gate
        energy = float(np.sum(sub_dc_removed))

        # adaptive threshold from outside-gate noise floor
        # use a wider band excluding gate to estimate noise
        noise_part = np.concatenate(
            [doppler_map[:, :g_start], doppler_map[:, g_end + 1 :]], axis=1
        )
        noise_floor = float(np.mean(noise_part)) + 1e-9

        threshold = threshold_scale * noise_floor * sub_dc_removed.size

        detected = energy > threshold
        return detected, energy, threshold, (g_start, g_end)