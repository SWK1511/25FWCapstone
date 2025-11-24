# Pluto SDR Hardware Interface
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

    def configure_common(self, sample_rate: float):
        """Pluto ADC/DAC sample rate"""
        self.sdr.sample_rate = int(sample_rate)

    def configure_tx(self, fc: float, rf_bw: float, tx_gain: int):
        """Pluto TX: carrier, bandwidth, gain"""
        self.sdr.tx_lo = int(fc)
        self.sdr.tx_rf_bandwidth = int(rf_bw)
        self.sdr.tx_hardwaregain_chan0 = int(tx_gain)

    def configure_rx(self, fc: float, rf_bw: float, rx_buffer_size: int, gain_mode: str = "slow_attack"):
        """Pluto RX: carrier, bandwidth, buffer size, gain mode"""
        self.sdr.rx_lo = int(fc)
        self.sdr.rx_rf_bandwidth = int(rf_bw)
        self.sdr.rx_buffer_size = int(rx_buffer_size)
        self.sdr.gain_control_mode_chan0 = gain_mode

    def tx(self, samples: np.ndarray):
        """Transmit complex baseband samples."""

        print(
            f"[DEBUG][TX] mean={np.mean(np.abs(samples)):.4f}, "
            f"max={np.max(np.abs(samples)):.4f}, shape={samples.shape}"
        )

        self.sdr.tx(samples)

    def rx(self) -> np.ndarray:
        """Receive complex baseband samples."""
        return np.array(self.sdr.rx(), dtype=np.complex64)

    def close(self):
        try:
            self.sdr.rx_destroy_buffer()
        except Exception:
            pass
        self.sdr = None
        time.sleep(0.05)