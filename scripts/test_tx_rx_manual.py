# scripts/test_tx_rx_loopback_manual_gain.py

import time
import numpy as np
import adi

def main():
    uri = "ip:pluto.local"

    print("[TEST] Connect to Pluto...")
    sdr = adi.Pluto(uri)

    fs = int(2e6)
    fc = int(2.4e9)

    # 공통 샘플레이트
    sdr.sample_rate = fs

    # RX 설정 (★ 수동 게인)
    sdr.rx_lo = fc
    sdr.rx_rf_bandwidth = int(2e6)
    sdr.gain_control_mode_chan0 = "manual"   # ← AGC 끄고
    sdr.rx_hardwaregain_chan0 = 10           # ← 예: 10 dB 고정 (너무 크면 줄이고, 너무 작으면 올리고)
    sdr.rx_buffer_size = 4096

    # TX 설정
    sdr.tx_lo = fc
    sdr.tx_rf_bandwidth = int(2e6)
    sdr.tx_hardwaregain_chan0 = -10   # 너무 세지 않게

    # 100 kHz 톤 생성
    N = 4096
    t = np.arange(N) / fs
    f_tone = 100e3
    tone = 0.5 * np.exp(1j * 2 * np.pi * f_tone * t)

    # cyclic TX 켜기
    sdr.tx_cyclic_buffer = True
    print("[TEST] Start TX 100 kHz tone...")
    sdr.tx(tone)

    print("[TEST] Reading RX level. TX/RX 안테나를 '멀리/가까이' 하면서 변화 확인.")
    try:
        while True:
            rx = sdr.rx()
            mag = np.abs(rx)

            mean_abs = float(np.mean(mag))
            max_abs  = float(np.max(mag))

            print(f"[LOOPBACK-MANUAL] RX mean={mean_abs:.2f}, max={max_abs:.2f}")
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n[TEST] stopped")
    finally:
        try:
            sdr.tx_destroy_buffer()
        except Exception:
            pass
        try:
            sdr.rx_destroy_buffer()
        except Exception:
            pass
        sdr = None
        time.sleep(0.05)

if __name__ == "__main__":
    main()