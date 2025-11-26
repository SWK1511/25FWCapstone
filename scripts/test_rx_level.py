# scripts/test_rx_level.py

import time
import numpy as np
import adi

def main():
    uri = "ip:pluto.local"  # 필요시 변경

    print("[TEST] Connect to Pluto...")
    sdr = adi.Pluto(uri)

    # 기본 세팅 (2.4 GHz 대역, 2 MHz 대역폭)
    sdr.sample_rate = int(2e6)
    sdr.rx_lo = int(2.4e9)
    sdr.rx_rf_bandwidth = int(2e6)
    sdr.gain_control_mode_chan0 = "slow_attack"
    sdr.rx_buffer_size = 4096

    print("[TEST] Start RX level test")
    try:
        while True:
            rx = sdr.rx()  # complex64 배열
            mag = np.abs(rx)

            mean_abs = float(np.mean(mag))
            std_abs  = float(np.std(mag))
            max_abs  = float(np.max(mag))

            print(f"RX level -> mean={mean_abs:.2f}, std={std_abs:.2f}, max={max_abs:.2f}")
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n[TEST] stopped")
    finally:
        try:
            sdr.rx_destroy_buffer()
        except Exception:
            pass
        sdr = None
        time.sleep(0.05)

if __name__ == "__main__":
    main()