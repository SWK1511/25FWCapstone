import os
import sys
import time
import numpy as np

# ---- fmcw 모듈 import를 위한 경로 추가 ----
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from fmcw.config import FMCWConfig
from fmcw.pluto_iface import PlutoInterface
from fmcw.waveform import make_chirp
from fmcw.processor import FMCWProcessor


def main():
    NUM_FRAMES = 10

    cfg = FMCWConfig()
    print(f"[INFO] 사용 URI : {cfg.uri}")

    # ---- Pluto 연결 ----
    pluto = PlutoInterface(cfg.uri)
    try:
        pluto.connect()
        print("[INFO] Pluto 연결 성공")
    except Exception as e:
        print(f"[ERROR] Pluto 연결 실패: {e}")
        return

    pluto.configure_common(cfg.sample_rate)
    pluto.configure_tx(cfg.fc, cfg.B * 2, cfg.tx_gain)
    pluto.configure_rx(cfg.fc, cfg.B * 2, cfg.rx_buffer_size)

    chirp = make_chirp(cfg)
    proc = FMCWProcessor(cfg)

    print("\n[INFO] 10프레임 도플러 에너지 디버그 시작")
    print("-------------------------------------------------------")
    print(" idx | total_E      max_bin_E    ratio(모션/기준)  dt(ms)")
    print("-------------------------------------------------------")

    baseline_E = None

    try:
        for i in range(NUM_FRAMES):
            t0 = time.time()

            # 1) 프레임 수집 (복소수 FMCW frame)
            frame = proc.collect_frame(pluto, chirp)

            # 2) 도플러 맵 계산
            doppler_map = proc.doppler_fft(frame)  # shape: [range_bins, doppler_bins]
            mag2 = np.abs(doppler_map) ** 2

            # 3) 전체 에너지 & 최대 에너지
            total_E = float(mag2.sum())
            max_bin_E = float(mag2.max())

            # 4) 기존 detect_human_presence 로 에너지 기준도 같이 보기
            detected, energy_gate, th, _ = proc.detect_human_presence(
                doppler_map,
                range_gate_m=(0.5, 8.0),
                doppler_exclude_bins=1,
                threshold_scale=1.3,
            )

            # baseline 업데이트 (no-motion 기준 비슷한 느낌용)
            if baseline_E is None:
                baseline_E = energy_gate + 1e-9
            else:
                alpha = 0.1
                baseline_E = (1 - alpha) * baseline_E + alpha * energy_gate

            ratio = energy_gate / (baseline_E + 1e-9)

            dt = (time.time() - t0) * 1000.0

            print(
                f"{i:3d} | "
                f"{total_E:10.3e}  "
                f"{max_bin_E:10.3e}  "
                f"{ratio:8.3f}         "
                f"{dt:6.1f}"
            )

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n[INTERRUPT] 사용자 중단")
    finally:
        pluto.close()
        print("[INFO] Pluto 연결 종료")


if __name__ == "__main__":
    main()
