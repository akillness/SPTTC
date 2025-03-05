import subprocess
import time
import logging
from datetime import datetime
import os

# 설정 값
REPO_PATH = "/path/to/your/github/repository"  # 실제 저장소 경로로 변경
INTERVAL_MINUTES = 10  # 풀 간격 (분)
BRANCH = "main"  # 기본 브랜치

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("git_pull.log"),
        logging.StreamHandler()
    ]
)

def git_pull():
    try:
        # 저장소 존재 여부 확인
        if not os.path.exists(os.path.join(REPO_PATH, ".git")):
            raise FileNotFoundError(f"Git repository not found at {REPO_PATH}")

        # Git pull 명령 실행
        result = subprocess.run(
            ["git", "-C", REPO_PATH, "pull", "origin", BRANCH],
            capture_output=True,
            text=True,
            check=True
        )

        # 결과 로깅
        logging.info(f"Pull successful\n{result.stdout}")
        return True

    except subprocess.CalledProcessError as e:
        logging.error(f"Pull failed\nError: {e.stderr}")
        return False
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        return False

def main():
    logging.info("Git Auto Pull Service Started")
    while True:
        try:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logging.info(f"Attempting pull at {now}")
            
            if git_pull():
                logging.info(f"Next pull in {INTERVAL_MINUTES} minutes")
            else:
                logging.warning("Retrying in 1 minute due to failure")
                time.sleep(60)  # 실패 시 1분 대기
                continue

            time.sleep(INTERVAL_MINUTES * 60)

        except KeyboardInterrupt:
            logging.info("Service stopped by user")
            break

if __name__ == "__main__":
    main()