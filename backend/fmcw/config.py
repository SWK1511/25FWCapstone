from dataclasses import dataclass

@dataclass
class FMCWConfig:
    """
    FMCW Radar Configuration (Jetson Nano Optimized)
    """
    # Pluto 연결 IP
    uri: str = "ip:pluto.local"

    # RF 중심 주파수 (2.4GHz)
    fc: float = 2.4e9

    # 샘플레이트 (2 MHz)
    sample_rate: float = 2e6

    # 대역폭 (50MHz) - 거리 해상도 확보를 위해 필수
    B: float = 50e6 

    # Chirp 지속 시간
    T: float = 5e-4

    # 프레임 당 Chirp 수 (속도 최적화: 64)
    num_chirps: int = 64

    # TX/RX Gain
    tx_gain: int = 0
    rx_gain: int = 50
    
    # Guard bins
    guard_bins: int = 0

    # FFT 크기 (속도 최적화: 512)
    fft_size: int = 512
    
    # 버퍼 크기
    rx_buffer_size: int = 64 * 512