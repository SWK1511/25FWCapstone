#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Pluto SDR RX gain / bandwidth 테스트 스크립트

- 서로 다른 RX RF Bandwidth
- 서로 다른 Gain Mode (slow_attack, fast_attack, manual)
- manual 모드에서는 여러 가지 수동 gain 값

에서 RX 레벨(mean, std, max)이 어떻게 변하는지 확인하기 위함.
"""

import time
import numpy as np
import adi


def main():
    uri = "ip:pluto.local"   # 필요하면 여기만 바꿔서 사용

    # 테스트할 파라미터들
    SAMPLE_RATE = 2e6
    RX_BUF_SIZE = 4096

    BW_LIST = [2e6, 10e6, 20e6]  # 2 MHz, 10 MHz, 20 MHz
    GAIN_MODES = ["slow_attack", "fast_attack", "manual"]
    MANUAL_GAINS = [0, 10, 20, 30, 40, 50, 60]  # dB

    print("[TEST] Connect to Pluto...")
    sdr = adi.Pluto(uri)

    # 공통 설정
    sdr.sample_rate = int(SAMPLE_RATE)
    sdr.rx_lo = int(2.4e9)             # 2.4 GHz 근처, 필요하면 변경 가능
    sdr.rx_buffer_size = int(RX_BUF_SIZE)

    print("[TEST] Start RX gain/bandwidth sweep")
    print("  sample_rate = %.1f MHz" % (SAMPLE_RATE / 1e6))
    print("  rx_buffer_size =", RX_BUF_SIZE)
    print("-" * 60)

    try:
        for bw in BW_LIST:
            sdr.rx_rf_bandwidth = int(bw)

            print("\n==============================")
            print("=== RX RF BW = %.1f MHz ===" % (bw / 1e6))
            print("==============================")

            for mode in GAIN_MODES:
                # gain mode 설정
                sdr.gain_control_mode_chan0 = mode

                if mode != "manual":
                    print(f"\n[MODE] gain_mode = {mode} (AGC)")
                    print("  -> 10번 정도 샘플링하면서 RX level 확인\n")

                    # 버퍼 몇 번 비워주고
                    for _ in range(3):
                        _ = sdr.rx()

                    for i in range(10):
                        rx = np.array(sdr.rx(), dtype=np.complex64)
                        mag = np.abs(rx)
                        mean = float(np.mean(mag))
                        std = float(np.std(mag))
                        mx = float(np.max(mag))
                        print(f"[{mode:11s}] #{i:02d}  mean={mean:6.2f}, std={std:6.2f}, max={mx:7.2f}")
                        time.sleep(0.2)

                else:
                    # manual 모드
                    print(f"\n[MODE] gain_mode = manual")
                    print("  -> 다양한 rx_hardwaregain_chan0 dB 값에서 RX level 확인\n")

                    for g in MANUAL_GAINS:
                        # 수동 gain 설정
                        sdr.rx_hardwaregain_chan0 = g
                        print(f"\n  >>> manual gain = {g} dB <<<")

                        # 버퍼 몇 번 비워주고
                        for _ in range(3):
                            _ = sdr.rx()

                        for i in range(5):
                            rx = np.array(sdr.rx(), dtype=np.complex64)
                            mag = np.abs(rx)
                            mean = float(np.mean(mag))
                            std = float(np.std(mag))
                            mx = float(np.max(mag))
                            print(f"[gain={g:2d} dB] #{i:02d}  mean={mean:6.2f}, std={std:6.2f}, max={mx:7.2f}")
                            time.sleep(0.2)

    except KeyboardInterrupt:
        print("\n[TEST] stopped by user")

    finally:
        del sdr
        time.sleep(0.1)


if __name__ == "__main__":
    main()