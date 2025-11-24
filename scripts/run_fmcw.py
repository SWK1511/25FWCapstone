import time
from fmcw.config import FMCWConfig
from fmcw.pluto_iface import PlutoInterface
from fmcw.waveform import make_chirp
from fmcw.processor import FMCWProcessor


def main():
    cfg = FMCWConfig()   #완성한 config 불러옴

    # 1) Pluto 연결/설정
    pluto = PlutoInterface(cfg.uri)
    pluto.connect()
    pluto.configure_common(cfg.sample_rate)
    pluto.configure_tx(cfg.fc, cfg.B * 2, cfg.tx_gain)
    pluto.configure_rx(cfg.fc, cfg.B * 2, cfg.rx_buffer_size)

    # 2) Chirp / Processor 준비
    chirp = make_chirp(cfg)
    proc = FMCWProcessor(cfg)

    print("[FMCW] running (library pipeline)... Ctrl+C to stop")
    try:
        while True:
            pluto.tx(chirp)
            rx = pluto.rx()

            beat = proc.mix_to_beat(rx, chirp)
            mag = proc.range_fft(beat)
            R, fb, peak_bin = proc.estimate_range(mag)

            print(f"Peak bin={peak_bin:4d}  fb={fb:8.1f} Hz  Range≈{R:6.2f} m")
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n[FMCW] stopped")
    finally:
        pluto.close()


if __name__ == "__main__":
    main()