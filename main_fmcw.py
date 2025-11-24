import time
from fmcw_radar import FMCWRadar

def main():
    radar = FMCWRadar(
        uri="ip:pluto.local",
        fc=2.4e9,
        sample_rate=2e6,
        B=0.5e6,      # 처음은 작게 시작
        T=2e-3,       # 2ms chirp
        tx_gain=-25,  # TX 낮게
        rx_buffer_size=8192
    )

    print("[FMCW] running... Ctrl+C to stop")
    try:
        while True:
            radar.transmit_once()
            rx = radar.receive_once()

            beat = radar.compute_beat(rx)
            mag = radar.range_fft(beat)

            R, fb, peak_bin = radar.estimate_range(mag)

            print(f"Peak bin={peak_bin:4d}  fb={fb:8.1f} Hz  Range≈{R:6.2f} m")
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n[FMCW] stopped")
    finally:
        radar.close()

if __name__ == "__main__":
    main()