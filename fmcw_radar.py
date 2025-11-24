import numpy as np
import adi
import time


class FMCWRadar:
    """
    Pluto SDR 기반 FMCW 레이더 (초기 버전)
    - baseband chirp 생성
    - chirp 송신(TX)
    - 반사 신호 수신(RX)
    - RX와 reference chirp를 믹싱하여 beat signal 생성
    - beat FFT → beat freq → 거리 추정

    주의:
    - 처음엔 TX gain을 낮게(-25dB 근처) 시작해야 RX 포화가 덜함
    - 송신/수신 안테나는 최대한 떨어뜨리거나 방향을 틀어줘야 반사파가 보임
    """

    def __init__(
        self,
        uri="ip:pluto.local",
        fc=2.4e9,          # carrier center frequency
        sample_rate=2e6,   # Hz
        B=0.5e6,           # sweep bandwidth (start small!)
        T=2e-3,            # chirp duration (sec)
        tx_gain=-25,       # dB (low TX to avoid saturation)
        rx_buffer_size=8192,
        gain_mode="slow_attack",
    ):
        self.fc = float(fc)
        self.fs = float(sample_rate)
        self.B = float(B)
        self.T = float(T)
        self.tx_gain = int(tx_gain)
        self.rx_buffer_size = int(rx_buffer_size)

        # speed of light
        self.c = 3e8

        # Pluto init
        self.sdr = adi.Pluto(uri)

        # sample rate
        self.sdr.sample_rate = int(self.fs)

        # chirp length in samples
        self.N = int(self.T * self.fs)
        if self.N <= 0:
            raise ValueError("Chirp length N must be positive.")

        # RX buffer must hold at least one chirp
        if self.N > self.rx_buffer_size:
            self.rx_buffer_size = self.N
        self.sdr.rx_buffer_size = int(self.rx_buffer_size)

        # TX config
        self.sdr.tx_lo = int(self.fc)
        self.sdr.tx_rf_bandwidth = int(self.B * 2)
        self.sdr.tx_hardwaregain_chan0 = self.tx_gain

        # RX config
        self.sdr.rx_lo = int(self.fc)
        self.sdr.rx_rf_bandwidth = int(self.B * 2)
        self.sdr.gain_control_mode_chan0 = gain_mode

        # reference chirp pre-generate
        self.chirp = self._make_chirp()

    def _make_chirp(self):
        """
        Baseband complex chirp:
        f(t) = (B/T)*t  (0→B)
        phase(t) = pi*(B/T)*t^2
        """
        t = np.arange(self.N) / self.fs
        k = self.B / self.T  # sweep slope (Hz/s)
        phase = np.pi * k * t**2
        chirp = np.exp(1j * phase)

        # normalize to avoid TX clipping
        chirp = chirp / np.max(np.abs(chirp))
        return chirp.astype(np.complex64)

    def transmit_once(self):
        """Transmit one chirp."""
        self.sdr.tx(self.chirp)

    def receive_once(self):
        """Receive IQ samples."""
        rx = self.sdr.rx()
        return np.array(rx, dtype=np.complex64)

    def compute_beat(self, rx):
        """
        Mix RX with conjugate of TX reference chirp
        beat[n] = rx[n] * conj(chirp[n])
        """
        rx_seg = rx[:self.N]
        beat = rx_seg * np.conj(self.chirp)

        # DC removal
        beat = beat - np.mean(beat)
        return beat

    def range_fft(self, beat):
        """FFT magnitude of beat signal."""
        window = np.hanning(len(beat))
        spec = np.fft.fft(beat * window)
        mag = np.abs(spec)
        return mag

    def estimate_range(self, mag, guard_bins=3):
        """
        beat FFT peak -> beat freq -> range
        guard_bins: DC 근처 bin 무시
        """
        mag2 = mag.copy()
        mag2[:guard_bins] = 0

        peak_bin = int(np.argmax(mag2))
        fb = peak_bin * (self.fs / self.N)

        # R = (c*T*fb)/(2B)
        R = (self.c * self.T * fb) / (2 * self.B)
        return R, fb, peak_bin

    def close(self):
        """Safe cleanup."""
        try:
            self.sdr.rx_destroy_buffer()
        except Exception:
            pass
        self.sdr = None
        time.sleep(0.05)