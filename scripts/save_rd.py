import os
import sys
import time
import argparse

import numpy as np

# FWCapstone 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from fmcw.config import FMCWConfig
from fmcw.pluto_iface import PlutoInterface
from fmcw.waveform import make_chirp
from fmcw.processor import FMCWProcessor


def main():
    parser = argparse.ArgumentParser(
        description="FMCW RD map(.npy) 수집 스크립트"
    )
    parser.add_argument(
        "--label",
        type=str,
        required=True,
        help="저장할 라벨 이름 (예: HUMAN, NO_MOTION, NON_HUMAN)",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=300,
        help="저장할 프레임 개수",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data_rd",
        help="루트 출력 디렉토리 (기본: data_rd)",
    )
    parser.add_argument(
        "--range-min",
        type=float,
        default=0.5,
        help="RD gate 최소 거리 [m] (기본: 0.5)",
    )
    parser.add_argument(
        "--range-max",
        type=float,
        default=8.0,
        help="RD gate 최대 거리 [m] (기본: 8.0)",
    )

    args = parser.parse_args()

    # -------------------------
    # 1) 출력 폴더 준비
    # -------------------------
    label = args.label.upper()
    save_dir = os.path.join(args.out_dir, label)
    os.makedirs(save_dir, exist_ok=True)

    print(f"[INFO] Save RD maps to: {save_dir}")
    print(f"[INFO] Label = {label}, num_frames = {args.num_frames}")
    print(f"[INFO] Range gate = {args.range_min} ~ {args.range_max} m")

    # -------------------------
    # 2) FMCW 초기화
    # -------------------------
    cfg = FMCWConfig()

    pluto = PlutoInterface(cfg.uri)
    print("[Pluto] Connecting...")
    pluto.connect()
    print("[Pluto] Connected.")

    pluto.configure_common(cfg.sample_rate)
    pluto.configure_tx(cfg.fc, cfg.B * 2, cfg.tx_gain)
    pluto.configure_rx(cfg.fc, cfg.B * 2, cfg.rx_buffer_size)

    chirp = make_chirp(cfg)
    proc = FMCWProcessor(cfg)

    # range 축에서 gate 미리 계산
    R_axis = proc.range_axis_m
    gate = np.where(
        (R_axis >= args.range_min) & (R_axis <= args.range_max)
    )[0]

    if gate.size > 0:
        print(
            f"[DEBUG] Range axis: {R_axis[0]:.2f} ~ {R_axis[-1]:.2f} m, "
            f"gate bins = {gate[0]} ~ {gate[-1]} (len={len(gate)})"
        )
    else:
        print("[WARN] 주어진 range gate 안에 bin이 없음 -> 전체 range 사용")
        gate = np.arange(len(R_axis))

    print("------------------------------------------------------------")
    print("[SAVE_RD] Start capturing RD maps. Ctrl+C to stop.")
    print("  >> 이 상태에서 사람/배경 상황을 맞춰서 움직여 주세요.")
    print("------------------------------------------------------------")

    try:
        for idx in range(args.num-frames):
            t0 = time.time()

            # 1) 프레임 수집
            frame = proc.collect_frame(pluto, chirp)

            # 2) Doppler FFT -> RD 맵 (range x doppler)
            doppler_map = proc.doppler_fft(frame)  # shape (N_range, N_dopp)

            # 3) 관심 거리 구간만 슬라이스
            rd = doppler_map[gate, :]  # (Ngate, N_dopp)

            # 4) magnitude로 변환 후 float32로 저장
            rd_mag = np.abs(rd).astype(np.float32)

            # (원하면 로그 스케일도 가능하지만 일단 raw magnitude로 저장)
            # rd_mag = 20.0 * np.log10(np.abs(rd) + 1e-6).astype(np.float32)

            # 5) 파일 이름 생성 & 저장
            ts_ms = int(time.time() * 1000)
            fname = f"{label.lower()}_{ts_ms}_{idx:04d}.npy"
            fpath = os.path.join(save_dir, fname)

            np.save(fpath, rd_mag)

            dt = (time.time() - t0) * 1000.0
            print(
                f"[SAVE] {idx+1:04d}/{args.num_frames:04d} "
                f"-> {fname}  shape={rd_mag.shape}  ({dt:.1f} ms)"
            )

            # 너무 빠르게 찍히지 않도록 약간 쉼
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n[SAVE_RD] Interrupted by user.")

    finally:
        pluto.close()
        print("[Pluto] Closed.")
        print("[SAVE_RD] Done.")


if __name__ == "__main__":
    main()
