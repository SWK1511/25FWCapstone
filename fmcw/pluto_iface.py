import numpy as np
import adi
import time

class PlutoInterface:
    """
    Hardware interface for Pluto SDR.
    """
    def __init__(self, uri: str):
        self.uri = uri
        self.sdr = None

    def connect(self):
        print(f"[Pluto] Connecting to {self.uri} ...")
        self.sdr = adi.Pluto(self.uri)
        print("[Pluto] Connected.")

    def configure_common(self, sample_rate: float):
        self.sdr.sample_rate = int(sample_rate)

    def configure_tx(self, fc: float, rf_bw: float, tx_gain: int):
        self.sdr.tx_lo = int(fc)
        self.sdr.tx_rf_bandwidth = int(rf_bw)
        self.sdr.tx_hardwaregain_chan0 = int(tx_gain)

    def configure_rx(self, fc: float, rf_bw: float, rx_buffer_size: int, gain_mode: str = "manual"):
        self.sdr.rx_lo = int(fc)
        self.sdr.rx_rf_bandwidth = int(rf_bw)
        self.sdr.rx_buffer_size = int(rx_buffer_size)
        self.sdr.gain_control_mode_chan0 = gain_mode
        self.sdr.rx_hardwaregain_chan0 = 50 # RX Gain 고정

    def tx(self, samples: np.ndarray):
        self.sdr.tx(samples)

    def rx(self) -> np.ndarray:
        return np.array(self.sdr.rx(), dtype=np.complex64)

    def close(self):
        try:
            self.sdr.rx_destroy_buffer()
        except Exception:
            pass
        self.sdr = None
        time.sleep(0.05)