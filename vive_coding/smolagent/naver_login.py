import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pickle
import os
import time

def save_cookies(driver, cookie_file):
    """쿠키를 파일로 저장"""
    if not os.path.exists('cookies'):
        os.makedirs('cookies')
    pickle.dump(driver.get_cookies(), open(f"cookies/{cookie_file}", "wb"))
    print("쿠키가 저장되었습니다.")

def load_cookies(driver, cookie_file):
    """저장된 쿠키 로드"""
    cookies = pickle.load(open(f"cookies/{cookie_file}", "rb"))
    for cookie in cookies:
        driver.add_cookie(cookie)
    print("쿠키를 로드했습니다.")

def naver_login(username, password, cookie_file='naver_cookies.pkl'):
    driver = None
    try:
        # Chrome 드라이버 설정
        options = uc.ChromeOptions()
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-blink-features=AutomationControlled')
        
        driver = uc.Chrome(
            options=options,
            version_main=134
        )
        
        # 쿠키 파일 존재 여부 확인
        if os.path.exists(f"cookies/{cookie_file}"):
            # 네이버 메인 페이지로 이동
            driver.get('https://www.naver.com')
            
            # 저장된 쿠키 로드
            load_cookies(driver, cookie_file)
            
            # 페이지 새로고침
            driver.refresh()
            time.sleep(3)
            
            # 로그인 상태 확인
            if check_login_status(driver):
                print("저장된 쿠키로 로그인 성공!")
                return True
        
        # 쿠키가 없거나 유효하지 않은 경우 새로 로그인
        print("새로 로그인을 시도합니다...")
        driver.get('https://nid.naver.com/nidlogin.login')
        time.sleep(3)
        
        # 아이디 입력
        id_input = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "id"))
        )
        id_input.send_keys(username)
        time.sleep(1)
        
        # 비밀번호 입력
        pw_input = driver.find_element(By.ID, "pw")
        pw_input.send_keys(password)
        time.sleep(1)
        
        # 로그인 버튼 클릭
        login_button = driver.find_element(By.ID, "log.login")
        driver.execute_script("arguments[0].click();", login_button)
        time.sleep(5)
        
        # 로그인 성공 확인
        if check_login_status(driver):
            print("새로운 로그인 성공!")
            # 쿠키 저장
            save_cookies(driver, cookie_file)
            return True
        else:
            print("로그인 실패!")
            return False
            
    except Exception as e:
        print(f"에러 발생: {str(e)}")
        return False
    
    finally:
        # 사용자 입력 대기
        input("계속하려면 아무 키나 누르세요...")
        
        # 브라우저 종료
        if driver:
            driver.quit()

def check_login_status(driver):
    """로그인 상태 확인"""
    try:
        # 네이버 메인 페이지로 이동
        driver.get('https://www.naver.com')
        time.sleep(3)
        
        # 로그인 버튼이 없으면 로그인된 상태
        login_button = driver.find_elements(By.CLASS_NAME, "link_login")
        if not login_button:
            return True
        return False
    except:
        return False

if __name__ == "__main__":
    # 로그인 정보
    USERNAME = "akillness"
    PASSWORD = "#Dbwls0208!"
    
    # 로그인 실행
    naver_login(USERNAME, PASSWORD) 