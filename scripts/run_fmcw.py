import time
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from fmcw.config import FMCWConfig
from fmcw.pluto_iface import PlutoInterface
from fmcw.waveform import make_chirp
from fmcw.processor import FMCWProcessor


def main():
    cfg = FMCWConfig()

    # 1) Pluto 연결 & 설정
    pluto = PlutoInterface(cfg.uri)
    pluto.connect()
    pluto.configure_common(cfg.sample_rate)
    pluto.configure_tx(cfg.fc, cfg.B * 2, cfg.tx_gain)
    pluto.configure_rx(cfg.fc, cfg.B * 2, cfg.rx_buffer_size)

    # 2) FMCW 처리기 & chirp
    chirp = make_chirp(cfg)
    proc = FMCWProcessor(cfg)

    # ---- 레인지 축 디버그 ----
    R_axis = proc.range_axis_m
    print(f"[DEBUG] Range axis: 0 ~ {R_axis.max():.1f} m")
    print(f"[DEBUG] First 10 bins (m): {[round(r,2) for r in R_axis[:10]]}")

    # 디버그용 range gate
    r0, r1 = 2.0, 6.0
    gate = np.where((R_axis >= r0) & (R_axis <= r1))[0]
    if len(gate) > 0:
        print(f"[DEBUG] Gate {r0}~{r1} m -> bins {gate[0]} ~ {gate[-1]} (len={len(gate)})")
    else:
        print(f"[DEBUG] Gate {r0}~{r1} m 안에 들어오는 bin이 없음!")

    print("[FMCW-HUMAN] running... Ctrl+C to stop")
    print("------------------------------------------------------------")

    frame_idx = 0
    baseline_E = None

    # ---------------------------------------------------------
    #  디바운싱 & 판정 파라미터
    # ---------------------------------------------------------
    detect_cnt = 0        # 연속 “true” 카운트
    release_cnt = 0       # 연속 “false” 카운트
    final_detect = False  # 최종 판단

    ON_FRAMES = 2        # 3프레임 연속 강한 모션이어야 ON
    OFF_FRAMES = 3        # 8프레임 연속 no-motion이면 OFF

    RATIO_TH = 1.3        # ratio > 1.3 이상일 때만 모션 후보
    TH_SCALE = 1.3        # detect_human_presence threshold_scale
    # ---------------------------------------------------------

    try:
        while True:
            t0 = time.time()

            # 프레임 수집
            try:
                frame = proc.collect_frame(pluto, chirp)
            except OSError as e:
                print(f"[ERROR] RX failed: {e}. Stop and power-cycle Pluto.")
                break

            doppler_map = proc.doppler_fft(frame)

            # 기존 함수로 energy / threshold 계산
            detected_raw, energy, threshold, _ = proc.detect_human_presence(
                doppler_map,
                range_gate_m=(0.5, 8.0),   # 넓게 (필요하면 2.0~6.0으로 줄여도 됨)
                doppler_exclude_bins=1,    # 중앙 도플러 제외
                threshold_scale=TH_SCALE
            )

# === Baseline update ===
            if baseline_E is None:
                baseline_E = energy + 1e-9
            else:
                alpha = 0.03    # baseline 상승을 훨씬 느리게
                baseline_E = (1 - alpha) * baseline_E + alpha * energy

            ratio = energy / (baseline_E + 1e-9)

            # === Strong motion 판단 ===
            RATIO_TH = 1.3      # 더 낮춰 탐지 민감도 증가
            strong_motion = (ratio > RATIO_TH)

            # === Debounce ===
            if strong_motion:
                detect_cnt += 1
                release_cnt = 0
            else:
                release_cnt += 1
                detect_cnt = 0

            if detect_cnt >= ON_FRAMES:
                final_detect = True
            if release_cnt >= OFF_FRAMES:
                final_detect = False
            # ---------------------------------------------------------

            dt = time.time() - t0

            # ★ 3프레임마다 결과 출력
            if True:
                if final_detect:
                    print(
                        f"[DETECT] motion      "
                        f"E={energy:.2e}  TH={threshold:.2e}  "
                        f"R={ratio:.2f}  frame={frame_idx}  ({dt*1000:.1f} ms)"
                    )
                else:
                    print(
                        f"[.....] no motion     "
                        f"E={energy:.2e}  TH={threshold:.2e}  "
                        f"R={ratio:.2f}  frame={frame_idx}  ({dt*1000:.1f} ms)"
                    )

            frame_idx += 1
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n[FMCW-HUMAN] stopped")
    finally:
        pluto.close()


if __name__ == "__main__":
    main()