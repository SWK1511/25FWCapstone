import time
import os
import sys
import argparse
from pathlib import Path

import numpy as np

# FWCapstone 최상위 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from fmcw.config import FMCWConfig
from fmcw.pluto_iface import PlutoInterface
from fmcw.waveform import make_chirp
from fmcw.processor import FMCWProcessor


# -----------------------------
# 저장 함수
# -----------------------------
DATA_ROOT = Path(os.path.expanduser("~/FWCapstone/data_iq"))

def save_iq_frame(frame: np.ndarray, label: str, frame_idx: int):
    """
    frame : proc.collect_frame(...) 에서 나온 한 프레임 (복소 배열)
    label : WALL / HUMAN / HUMAN_MOVE / HUMAN_BEHIND / HUMAN_BEHIND_MOVE
    """
    label_dir = DATA_ROOT / label
    label_dir.mkdir(parents=True, exist_ok=True)

    ts = int(time.time() * 1000)  # ms 단위 타임스탬프
    fname = f"{label}_f{frame_idx:06d}_{ts}.npy"
    np.save(label_dir / fname, frame.astype(np.complex64))

    print(f"[SAVE] {label} -> {label_dir / fname}")


def main():
    # --------- CLI 옵션 ---------
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--label",
        type=str,
        default="HUMAN",
        choices=[
            "WALL",
            "HUMAN",
            "HUMAN_MOVE",
            "HUMAN_BEHIND",
            "HUMAN_BEHIND_MOVE",
        ],
        help="수집할 라벨",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=100,
        help="저장할 프레임 개수(정확히 이 개수만큼 저장 후 종료)",
    )
    args = parser.parse_args()

    label = args.label
    max_frames = args.frames

    print(f"[INFO] label={label}, max_frames={max_frames}")
    print("------------------------------------------------------------")

    # ------------------------------------------------------------------
    # Pluto & FMCW 초기화 (run_fmcw.py 코드와 동일 구조)
    # ------------------------------------------------------------------
    cfg = FMCWConfig()

    pluto = PlutoInterface(cfg.uri)
    pluto.connect()
    pluto.configure_common(cfg.sample_rate)
    pluto.configure_tx(cfg.fc, cfg.B * 2, cfg.tx_gain)
    pluto.configure_rx(cfg.fc, cfg.B * 2, cfg.rx_buffer_size)

    chirp = make_chirp(cfg)
    proc = FMCWProcessor(cfg)

    # ------------------------------------------------------------------
    # Range 축 출력
    # ------------------------------------------------------------------
    R_axis = proc.range_axis_m
    print(f"[DEBUG] Range axis: 0 ~ {R_axis.max():.1f} m")
    print(f"[DEBUG] First 10 bins (m): {[round(r, 2) for r in R_axis[:10]]}")

    # 디버그용 range gate
    r0, r1 = 2.0, 6.0
    gate = np.where((R_axis >= r0) & (R_axis <= r1))[0]
    if len(gate) > 0:
        print(f"[DEBUG] Gate {r0}~{r1} m -> {gate[0]} ~ {gate[-1]}")
    else:
        print("[DEBUG] 해당 범위 gate 없음")

    print("[FMCW-COLLECT] 시작 (Ctrl+C 종료)")
    print("------------------------------------------------------------")

    frame_idx = 0

    # ------------------------------------------------------------------
    # 모션 판정 파라미터
    # ------------------------------------------------------------------
    detect_cnt = 0
    release_cnt = 0
    final_detect = False

    ON_FRAMES = 2        # 연속 2프레임 모션이면 ON
    OFF_FRAMES = 4       # 연속 4프레임 조용하면 OFF

    # ★ 문턱값: 비교적 많이 낮춘 값
    RATIO_TH = 1.05      # baseline보다 5%만 커져도 strong_motion 후보
    TH_SCALE = 1.10      # detect_human_presence threshold_scale

    # 라벨 그룹
    STATIC_LABELS = {"WALL", "HUMAN", "HUMAN_BEHIND"}
    MOTION_LABELS = {"HUMAN_MOVE", "HUMAN_BEHIND_MOVE"}

    # 저장된 개수 카운트
    saved_count = 0

    # ------------------------------------------------------------------
    # baseline(정지 기준) 잡기 위한 워밍업
    # ------------------------------------------------------------------
    WARMUP_FRAMES = 60           # 처음 60프레임은 "정지 상태"라고 가정
    warmup_count = 0
    baseline_sum = 0.0
    baseline_E = None
    warmup_done = False

    print(f"[INFO] 워밍업 시작: 처음 {WARMUP_FRAMES} 프레임 동안은 '정지 기준'만 측정합니다.")
    print("[INFO] ★ 이때는 사람/물체 모두 최대한 가만히 있어야 합니다! ★")

    # ------------------------------------------------------------------
    # 메인 루프
    # ------------------------------------------------------------------
    try:
        while True:
            if saved_count >= max_frames:
                print(f"[INFO] 원하는 저장 개수({max_frames})에 도달. 종료.")
                break

            t0 = time.time()

            # 프레임 수집
            try:
                frame = proc.collect_frame(pluto, chirp)
            except OSError as e:
                print(f"[ERROR] RX failed: {e}")
                break

            doppler_map = proc.doppler_fft(frame)

            # 에너지 계산 (detect_human_presence 활용)
            detected_raw, energy, threshold, _ = proc.detect_human_presence(
                doppler_map,
                range_gate_m=(0.5, 8.0),
                doppler_exclude_bins=1,
                threshold_scale=TH_SCALE,
            )

            # -------------------------------
            #        1) 워밍업 단계
            # -------------------------------
            if not warmup_done:
                warmup_count += 1
                baseline_sum += energy

                avg_now = baseline_sum / warmup_count

                print(
                    f"[WARMUP] frame={frame_idx} E={energy:.2e} "
                    f"avg={avg_now:.2e} ({warmup_count}/{WARMUP_FRAMES})"
                )

                if warmup_count >= WARMUP_FRAMES:
                    baseline_E = baseline_sum / warmup_count + 1e-9
                    warmup_done = True
                    print("------------------------------------------------------------")
                    print(f"[INFO] 워밍업 완료. baseline_E = {baseline_E:.2e}")
                    print("[INFO] 이제부터 움직임을 시작해 주세요.")
                    print("------------------------------------------------------------")

                frame_idx += 1
                time.sleep(0.05)
                # 워밍업 동안에는 모션 판정/저장 안 함
                continue

            # -------------------------------
            #        2) 본격 모션 판정
            # -------------------------------
            ratio = energy / (baseline_E + 1e-9)
            strong_motion = ratio > RATIO_TH

            # 디바운싱
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

            dt = time.time() - t0

            # 디버그용 로그
            print(
                f"[DBG] E={energy:.2e} base={baseline_E:.2e} "
                f"ratio={ratio:.3f} strong={strong_motion}"
            )

            # 상태 로그
            if final_detect:
                print(
                    f"[DETECT] motion  frame={frame_idx} "
                    f"E={energy:.2e} R={ratio:.3f} ({dt*1000:.1f} ms)"
                )
            else:
                print(
                    f"[.....] no motion frame={frame_idx} "
                    f"E={energy:.2e} R={ratio:.3f} ({dt*1000:.1f} ms)"
                )

            # -------------------------------------------------------
            #                 I/Q 저장 로직
            # -------------------------------------------------------
            # 정지 라벨: 진짜 no motion일 때만 저장
            if label in STATIC_LABELS:
                if not final_detect:
                    save_iq_frame(frame, label, frame_idx)
                    saved_count += 1
                    print(f"[INFO] Saved {saved_count}/{max_frames} frames (STATIC)")

            # 모션 라벨: final_detect가 True일 때만 저장
            if label in MOTION_LABELS:
                if final_detect:
                    save_iq_frame(frame, label, frame_idx)
                    saved_count += 1
                    print(f"[INFO] Saved {saved_count}/{max_frames} frames (MOTION)")

            frame_idx += 1
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n[FMCW-COLLECT] 사용자 종료")

    finally:
        pluto.close()


if __name__ == "__main__":
    main()
