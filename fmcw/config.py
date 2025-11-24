from dataclasses import dataclass

@dataclass
class FMCWConfig:
    # Pluto 연결
    uri: str = "ip:pluto.local"

    # RF 설정
    fc: float = 2.40e9            # 2.4GHz 대역
    sample_rate: float = 2e6      # Pluto 안정 MAX

    # FMCW 파라미터
    B: float = 2.39e6             # 주어진 bandwidth
    T: float = 1.5e-3             # 1.5ms (50m 커버 안정적)

    # 송수신 설정
    tx_gain: int = -45            # 누설 최소 수준
    rx_buffer_size: int = 4096    # 1.5ms * 2MHz = 3000 sample → 충분

    # FFT/신호처리
    fft_size: int = 4096
    guard_bins: int = 4          # DC 성분 제거용