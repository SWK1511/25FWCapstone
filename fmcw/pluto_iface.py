# 오로지 Pluto SDR의 장비제어만 하는 모듈임
import numpy as np
import adi
import time


class PlutoInterface:
    """
    Hardware interface for Pluto SDR.
    Keeps all adi.Pluto calls in one place.
    """

    def __init__(self, uri: str):
        self.uri = uri
        self.sdr = None

    def connect(self):
        print(f"[Pluto] Connecting to {self.uri} ...")
        self.sdr = adi.Pluto(self.uri)
        print("[Pluto] Connected.")

    def configure_common(self, sample_rate: float): #Pluto의 ADC/DAC 샘플레이트 설정
        self.sdr.sample_rate = int(sample_rate)

    def configure_tx(self, fc: float, rf_bw: float, tx_gain: int): #Pluto의 송신 주파수/대역폭/이득 설정
        self.sdr.tx_lo = int(fc)
        self.sdr.tx_rf_bandwidth = int(rf_bw)
        self.sdr.tx_hardwaregain_chan0 = int(tx_gain)

    def configure_rx(self, fc: float, rf_bw: float, rx_buffer_size: int, gain_mode: str = "slow_attack"): # Pluto의 수신 주파수/대역폭/버퍼사이즈/이득모드 설정
        self.sdr.rx_lo = int(fc)
        self.sdr.rx_rf_bandwidth = int(rf_bw)
        self.sdr.rx_buffer_size = int(rx_buffer_size)
        self.sdr.gain_control_mode_chan0 = gain_mode

    def tx(self, samples: np.ndarray): # Baseband complex 샘플 송신
        """Transmit complex baseband samples."""
        self.sdr.tx(samples)

    def rx(self) -> np.ndarray: # Baseband complex 샘플 수신
        """Receive complex baseband samples."""
        return np.array(self.sdr.rx(), dtype=np.complex64)

    def close(self):
        try:
            self.sdr.rx_destroy_buffer()
        except Exception:
            pass
        self.sdr = None
        time.sleep(0.05)