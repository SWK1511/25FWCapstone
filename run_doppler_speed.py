#!/usr/bin/env python3
import adi
import numpy as np
import time
import signal
import sys
from scipy.signal import windows

# --------------------------
# 설정값
# --------------------------
fc = 2.4e9             # 송신/수신 주파수 2.4GHz
fs = 1_000_000         # 샘플링 주파수
N = 4096               # FFT 크기
c = 3e8                # 빛의 속도

# 도플러 필터링
MIN_DOPPLER_HZ = 5       # 잡음 제거
MAX_DOPPLER_HZ = 1000    # 사람 속도 범위 내
WIN = windows.hann(N)

# --------------------------
# Pluto 연결
# --------------------------
print(">>> Connecting to Pluto...")
sdr = adi.Pluto("ip:192.168.2.1")

# RX 설정
sdr.rx_rf_bandwidth = 2_000_000
sdr.rx_lo = int(fc)
sdr.sample_rate = int(fs)
sdr.rx_buffer_size = N

# TX 설정 (CW)
sdr.tx_rf_bandwidth = 2_000_000
sdr.tx_lo = int(fc)
sdr.tx_hardwaregain_chan0 = -10
sdr.tx_cyclic_buffer = True

# CW 신호 생성 (순수한 cosine)
t = np.arange(1024) / fs
cw = 0.5 * np.cos(2 * np.pi * 10e3 * t)   # 작은 10kHz offset tone
sdr.tx(cw)

print("Stabilize 1s...")
time.sleep(1)

print("Start CW Doppler Measurement (Ctrl-C to stop)")
print("-" * 60)

# --------------------------
# Ctrl + C 핸들러
# --------------------------
def stop_handler(sig, frame):
    print("\nStopped.")
    sdr.tx_destroy_buffer()
    sys.exit(0)

signal.signal(signal.SIGINT, stop_handler)


# --------------------------
# 메인 루프
# --------------------------
while True:
    raw = sdr.rx()
    iq = raw * WIN
    fft = np.fft.fftshift(np.fft.fft(iq))
    mag = 20 * np.log10(np.abs(fft) + 1e-6)

    center = N // 2
    search = mag[center - 800 : center + 800]
    rel_idx = np.argmax(search)
    peak_bin = rel_idx + (center - 800)

    # 도플러 주파수
    fd = (peak_bin - center) * (fs / N)
    f_doppler = abs(fd)

    # 필터링
    if f_doppler < MIN_DOPPLER_HZ or f_doppler > MAX_DOPPLER_HZ:
        speed = 0
    else:
        speed = (c * f_doppler) / (2 * fc)

    print(f"Doppler: {f_doppler:7.1f} Hz | Speed: {speed:5.3f} m/s", end="\r")
