import time, builtins

_ORIG_PRINT = builtins.print
_last_print_time = 0.0

def rate_limited_print(*args, **kwargs):
    global _last_print_time
    now = time.time()
    if now - _last_print_time >= 0.5:
        _ORIG_PRINT(*args, **kwargs)
        _last_print_time = now

builtins.print = rate_limited_print

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from fmcw.config import FMCWConfig
from fmcw.pluto_iface import PlutoInterface
from fmcw.waveform import make_chirp
from fmcw.processor import FMCWProcessor


def main():
    cfg = FMCWConfig()

    pluto = PlutoInterface(cfg.uri)
    pluto.connect()
    pluto.configure_common(cfg.sample_rate)
    pluto.configure_tx(cfg.fc, cfg.B * 2, cfg.tx_gain)
    pluto.configure_rx(cfg.fc, cfg.B * 2, cfg.rx_buffer_size)

    chirp = make_chirp(cfg)
    proc = FMCWProcessor(cfg)

    print("[FMCW-HUMAN] running... Ctrl+C to stop")
    print("-" * 60)

    frame_idx = 0

    try:
        while True:
            t0 = time.time()

            rx_test = pluto.rx()
            print("[DEBUG] RX mean abs:", np.mean(np.abs(rx_test)))

            frame = proc.collect_frame(pluto, chirp)
            doppler_map = proc.doppler_fft(frame)

            detected, energy, threshold, gate_bins = proc.detect_human_presence(
                doppler_map,
                range_gate_m=(0.0, 300.0),
                doppler_exclude_bins=3,
                threshold_scale=6.0
            )

            dt = time.time() - t0
            if detected:
                print(f"[DETECT] HUMAN MOTION!  E={energy:.2e}  TH={threshold:.2e}  frame={frame_idx}  ({dt*1000:.1f} ms)")
            else:
                print(f"[.....] no motion     E={energy:.2e}  TH={threshold:.2e}  frame={frame_idx}  ({dt*1000:.1f} ms)")

            frame_idx += 1
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n[FMCW-HUMAN] stopped")
    finally:
        pluto.close()


if __name__ == "__main__":
    main()