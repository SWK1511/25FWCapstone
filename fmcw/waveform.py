import numpy as np
from .config import FMCWConfig

def make_chirp(cfg: FMCWConfig) -> np.ndarray:
    """
    Create baseband complex up-chirp
    """
    fs = cfg.sample_rate
    N = int(cfg.T * fs)
    t = np.arange(N) / fs
    k = cfg.B / cfg.T  # sweep slope (Hz/s)
    phase = np.pi * k * t**2
    chirp = np.exp(1j * phase)

    chirp = chirp / np.max(np.abs(chirp))  # normalize
    return chirp.astype(np.complex64)

def make_frame(cfg: FMCWConfig, num_chirps: int) -> np.ndarray:
    chirp = make_chirp(cfg)
    frame = np.tile(chirp, (num_chirps, 1))
    return frame