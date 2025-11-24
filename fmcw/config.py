from dataclasses import dataclass

@dataclass
class FMCWConfig:
    """
    FMCW Radar Configuration
    All parameters live here so other modules can import one object.
    """

    # Pluto 연결
    uri: str = "ip:pluto.local"

    # RF 중심 주파수
    fc: float = 2.4e9

    # 샘플레이트
    sample_rate: float = 2e6

    # FMCW sweep bandwidth
    B: float = 2.39e6

    # Chirp duration (0.5 ms)
    T: float = 5e-4

    # 프레임 당 chirp 수 (Doppler/존재탐지용)
    num_chirps: int = 128

    # TX 출력 (dB)
    tx_gain: int = -45

    # DC/누설 제거용 guard bins
    guard_bins: int = 0

    # RX 버퍼/FFT 크기 (안 적어도 되지만 기본값 추천)
    rx_buffer_size: int = 4096
    fft_size: int = 4096