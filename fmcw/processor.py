import numpy as np
from .config import FMCWConfig


class FMCWProcessor:
    """
    Signal processing core for FMCW radar:
    - beat mixing
    - range FFT
    - range estimation
    """

    def __init__(self, cfg: FMCWConfig):
        self.cfg = cfg
        self.fs = cfg.sample_rate
        self.N = int(cfg.T * self.fs)
        self.c = 3e8

    def mix_to_beat(self, rx: np.ndarray, chirp: np.ndarray) -> np.ndarray:
        """
        beat[n] = rx[n] * conj(chirp[n])
        """
        rx_seg = rx[:self.N]
        beat = rx_seg * np.conj(chirp)
        beat = beat - np.mean(beat)  # DC removal
        return beat

    def range_fft(self, beat: np.ndarray) -> np.ndarray:
        """
        FFT magnitude of beat.
        """
        Nfft = self.cfg.fft_size
        window = np.hanning(len(beat))
        spec = np.fft.fft(beat * window, n=Nfft)
        mag = np.abs(spec)
        return mag

    def estimate_range(self, mag: np.ndarray) -> tuple[float, float, int]:
        """
        Peak bin -> beat freq -> range.
        """
        guard = self.cfg.guard_bins
        mag2 = mag.copy()
        mag2[:guard] = 0

        peak_bin = int(np.argmax(mag2))
        fb = peak_bin * (self.fs / self.cfg.fft_size)

        R = (self.c * self.cfg.T * fb) / (2 * self.cfg.B)
        return R, fb, peak_bin