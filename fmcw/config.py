from dataclasses import dataclass

@dataclass
class FMCWConfig:
    """
    FMCW Radar Configuration
    젯슨 나노 실시간 처리를 위한 최적화 설정
    """

    # Pluto 연결
    uri: str = "ip:192.168.2.1" #바꾸지 마시오 

    # RF 중심 주파수 (2.4GHz 안테나 사용)
    fc: float = 2.4e9

    # 샘플레이트 (2 MHz - USB 전송 안정성 최적)
    sample_rate: float = 2e6

    # [중요] FMCW sweep bandwidth (대역폭)
    # 샘플레이트(2MHz)보다 크면 신호가 깨집니다(Aliasing).
    # 1 MHz로 설정하여 안정적인 신호를 생성합니다.
    # (참고: 대역폭이 좁아 거리 해상도는 낮지만, 도플러(속도) 분석에는 문제없습니다.)
    B: float = 50e6

    # Chirp duration (0.5 ms)
    T: float = 5e-4

    # [중요] 프레임 당 chirp 수 (속도 해상도)
    # 128 -> 64로 줄여서 처리 속도를 2배 높입니다.
    num_chirps: int = 128

    # TX 출력 (dB) - 0이면 최대 출력
    tx_gain: int = 0

    # RX 게인 (dB) - 50 정도면 실내에서 충분합니다.
    rx_gain: int = 64
    
    # DC/누설 제거용 guard bins
    guard_bins: int = 0

    # [중요] RX 버퍼/FFT 크기 (데이터 크기 축소)
    # FFT 크기를 1024 -> 512로 줄여 연산량을 줄입니다.
    fft_size: int = 1024
    
    # 버퍼 크기는 반드시 (num_chirps * fft_size)와 같아야 합니다.
    rx_buffer_size: int = 128 * 1024