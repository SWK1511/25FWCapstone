import numpy as np

class FMCWProcessor:
    """
    FMCW Radar Signal Processor (Optimized for Jetson Nano)
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.num_chirps = cfg.num_chirps
        self.fft_size = cfg.fft_size
        # 한 Chirp당 샘플 수 (버퍼 크기 / 첩 개수)
        self.samples_per_chirp = int(cfg.rx_buffer_size / cfg.num_chirps)

    def collect_frame(self, pluto, chirp):
        """
        SDR에서 데이터를 받아 프레임 단위로 변환
        """
        # 1. 데이터 수신 (1차원 배열)
        rx_data = pluto.rx()
        
        # 2. 형태 변환 (Chirp 개수 x 샘플 수)
        # 데이터 개수가 안 맞을 경우를 대비해 슬라이싱
        expected_len = self.num_chirps * self.samples_per_chirp
        if len(rx_data) != expected_len:
            rx_data = rx_data[:expected_len]
            
        frame = rx_data.reshape(self.num_chirps, self.samples_per_chirp)
        return frame

    def doppler_fft(self, frame):
        """
        2D FFT 수행 (Range-Doppler Map 생성)
        """
        # 1. Range FFT (거리) - 가로 방향
        # Hanning Window를 적용하여 사이드로브 억제
        win_range = np.hanning(frame.shape[1])
        range_profile = np.fft.fft(frame * win_range, n=self.fft_size, axis=1)
        
        # 2. Doppler FFT (속도) - 세로 방향
        win_doppler = np.hanning(self.num_chirps)
        # Broadcasting을 위해 차원 맞춤
        win_doppler = win_doppler.reshape(-1, 1)
        
        doppler_map = np.fft.fft(range_profile * win_doppler, axis=0)
        doppler_map = np.fft.fftshift(doppler_map, axes=0)
        
        return np.abs(doppler_map)
