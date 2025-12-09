# 파일명: main_head.py
# 기능: 1 → backend/scripts/run_motion_cw.py 실행
#      2 → backend/scripts/run_motion_fmcw.py 실행

import os
import sys
import subprocess

# --- 실행 스크립트 경로 설정 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_DIR = os.path.join(BASE_DIR, "backend", "scripts")

def run_script(script_name):
    """지정된 Python 스크립트 실행"""
    script_path = os.path.join(SCRIPT_DIR, script_name)

    if not os.path.exists(script_path):
        print(f"[ERROR] 스크립트 없음: {script_path}")
        return

    print(f"[INFO] 실행 중 → {script_path}")
    
    cmd = [sys.executable, script_path]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] 스크립트 실행 오류: {e}")

def main():
    print("\n===== 레이더 모드 선택 =====")
    print("1) CW 모드 실행 (run_motion_cw.py)")
    print("2) FMCW 모드 실행 (run_motion_fmcw.py)")
    print("q) 종료")

    choice = input("\n번호 입력: ").strip()

    if choice == "1":
        run_script("run_motion_cw.py")

    elif choice == "2":
        run_script("run_motion_fmcw.py")

    elif choice.lower() == "q":
        print("종료합니다.")
        return

    else:
        print("잘못된 입력입니다. (1, 2, q 중 선택)")

if __name__ == "__main__":
    main()
