"""
실행 방법 (FWCapstone 루트에서):
cd ~/FWCapstone
source .venv/bin/activate
python -m scripts.run_motion_detector
"""

import sys
import time
import numpy as np

from fmcw.config import MotionConfig
from fmcw.pluto_iface import PlutoInterface
from fmcw.waveform import make_motion_tone


def main() -> None:
    cfg = MotionConfig()

    # 1) Pluto 연결 + 설정
    print(f">>> PlutoSDR({cfg.uri}) 연결 중...")
    try:
        pluto = PlutoInterface(cfg.uri)
        pluto.connect(
            sample_rate=cfg.sample_rate,
            center_freq=cfg.center_freq,
            rx_buffer_size=cfg.rx_buffer_size,
            rx_gain=cfg.rx_gain,
            tx_gain=cfg.tx_gain,
        )
    except Exception as e:
        print(f"❌ PlutoSDR 연결 실패: {e}")
        return

    print(">>> PlutoSDR 설정 완료")

    # 2) 송신 톤 생성 및 시작
    tx_signal = make_motion_tone(cfg)
    try:
        pluto.start_tx(tx_signal)
    except Exception as e:
        print(f"❌ TX 시작 실패: {e}")
        pluto.close()
        return

    # 3) 초기 캘리브레이션 (0점 잡기)
    print("\n" + "=" * 60)
    print("   [ 스마트 모션 감지기 (노이즈 필터링 적용됨) ]")
    print("=" * 60)
    print(f">>> 안정화 대기 중... ({cfg.settle_time:.1f}초)")
    time.sleep(cfg.settle_time)

    print(">>> 기준값 측정 중... (사람은 가만히 계세요!)")
    baseline_list = []
    for _ in range(cfg.calib_frames):
        data = pluto.sdr.rx()
        energy = np.mean(np.abs(data))
        baseline_list.append(energy)
        time.sleep(cfg.calib_sleep)

    baseline = float(np.mean(baseline_list))
    print(f">>> 기준값 설정 완료: {baseline:.2f}")
    print(">>> 감지 시작! (Ctrl+C로 종료)\n")

    # 4) 메인 루프 (원래 단일 파일 코드의 로직)
    current_score = 0.0

    try:
        while True:
            # 1. 데이터 수신 및 에너지 계산
            data = pluto.sdr.rx()
            current_energy = np.mean(np.abs(data))

            # 2. 변화량 계산 (현재값 - 기준값)
            diff = abs(current_energy - baseline)

            # 3. 점수 판정 로직
            if diff > cfg.threshold:
                # 변화량이 threshold를 넘으면 점수 추가 (사람 움직임)
                current_score += 2.0
            else:
                # 변화량이 적으면 점수 감소
                current_score -= 1.0

                # [적응형 기준값]
                # 사람이 없다고 판단될 때(점수 0 이하), 기준값을 현재 환경에 맞게 미세 조정
                if current_score <= 0:
                    baseline = (
                        baseline * (1.0 - cfg.adaptation_rate)
                        + current_energy * cfg.adaptation_rate
                    )

            # 점수 범위 제한
            if current_score < 0:
                current_score = 0
            if current_score > cfg.max_score:
                current_score = cfg.max_score

            # 4. 시각화
            is_detected = current_score > cfg.detect_limit

            # 게이지 바 길이 설정
            bar_len = int(current_score * 2.0)
            if bar_len > 40:
                bar_len = 40

            bar = "█" * bar_len
            space = " " * (40 - bar_len)

            if is_detected:
                status = "🚨 DETECTED!"
                color = "\033[91m"     # 빨강
                bar_color = "\033[91m"
            else:
                status = "   Secure   "
                color = "\033[92m"     # 녹색
                bar_color = "\033[90m" # 회색

            reset = "\033[0m"

            info = f"기준:{baseline:5.1f} | 변화:{diff:4.1f} | 점수:{current_score:4.1f}"
            print(
                f"\r{color}[{status}]{reset} "
                f"{info} |{bar_color}{bar}{space}{reset}|",
                end="",
            )

            # 반응 속도 빠르게 유지 (원본 코드처럼 sleep 없음)

    except KeyboardInterrupt:
        print("\n\n>>> 시스템을 종료합니다.")
    finally:
        pluto.close()


if __name__ == "__main__":
    main()