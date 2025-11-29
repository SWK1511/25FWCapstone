# scripts/run_data_capture.py

"""
실행 방법 (FWCapstone 루트에서):

cd ~/FWCapstone
source .venv/bin/activate
python -m scripts.run_data_capture

"""

from fmcw.config import DataCaptureConfig
from fmcw.waveform import make_data_capture_tone
from fmcw.pluto_iface import PlutoInterface
from fmcw.processor import BaselineTracker, collect_data_batch


def main() -> None:
    cfg = DataCaptureConfig()
    cfg.ensure_dirs()

    # 1) Pluto 연결 + 설정
    pluto = PlutoInterface(cfg.sdr_uri)
    try:
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

    # 2) 송신 톤 생성 및 시작
    tx_signal = make_data_capture_tone(cfg)
    try:
        pluto.start_tx(tx_signal)
    except Exception as e:
        print(f"❌ TX 시작 실패: {e}")
        pluto.close()
        return

    # 3) 기준값 측정
    baseline = BaselineTracker.measure(pluto, cfg)

    # 4) 메뉴 루프 (팀원 data.py 의 main 부분)
    try:
        while True:
            print("\n" + "=" * 50)
            print(f"   [ 📡 데이터 수집기 | 현재 기준값: {baseline.value:.2f} ]")
            print("=" * 50)
            print("  1. '빈 방' (Empty) 데이터 수집")
            print("  2. '사람' (Human) 데이터 수집")
            print("  r. 기준값(0점) 다시 잡기")
            print("  q. 종료")
            print("-" * 50)

            cmd = input("선택 >> ").strip().lower()

            if cmd == "1":
                input(">>> 방을 비우고 Enter를 누르세요...")
                collect_data_batch(
                    pluto=pluto,
                    cfg=cfg,
                    baseline=baseline,
                    label_name="empty",
                    save_dir=cfg.dir_empty,
                    count=cfg.save_batch_size,
                )

            elif cmd == "2":
                print(">>> ⚠️ 그래프(신호강도)가 잘 올라가는지 확인하면서 수집하세요!")
                input(">>> 가벽 뒤에 서서 Enter를 누르세요...")
                collect_data_batch(
                    pluto=pluto,
                    cfg=cfg,
                    baseline=baseline,
                    label_name="human",
                    save_dir=cfg.dir_human,
                    count=cfg.save_batch_size,
                )

            elif cmd == "r":
                baseline.recalibrate(pluto, cfg)

            elif cmd == "q":
                print("종료합니다.")
                break

            else:
                print("잘못된 입력입니다.")

    except KeyboardInterrupt:
        print("\n프로그램 종료 (Ctrl+C).")

    finally:
        pluto.close()


if __name__ == "__main__":
    main()