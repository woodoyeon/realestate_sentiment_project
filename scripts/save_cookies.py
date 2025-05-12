import json
import os
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

# ✅ 경로 설정
BASE_DIR = os.getcwd()
CHROME_DRIVER_PATH = os.path.join(BASE_DIR, "..", "driver", "chromedriver.exe")
COOKIE_OUTPUT_PATH = os.path.join(BASE_DIR, "..", "data", "kbland_cookies.json")

# ✅ 셀레니움 옵션 설정
options = Options()
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
# ❌ headless 제거 (직접 로그인 필요!)
# options.add_argument("--headless")

# ✅ 웹드라이버 실행
driver = webdriver.Chrome(service=Service(CHROME_DRIVER_PATH), options=options)

# ✅ 수동 로그인
driver.get("https://kbland.kr")
print("👉 로그인 창이 뜨면 구글 로그인을 완료한 뒤 엔터를 누르세요.")
input("⏳ 로그인 후 Enter 키 ▶ 쿠키 저장")

# ✅ 쿠키 저장
cookies = driver.get_cookies()
with open(COOKIE_OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(cookies, f, ensure_ascii=False, indent=2)

print(f"✅ 쿠키 저장 완료! → {COOKIE_OUTPUT_PATH}")
driver.quit()
