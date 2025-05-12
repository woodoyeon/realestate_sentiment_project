# 📊 부동산 뉴스 감정 분석 프로젝트

한국 부동산 뉴스를 크롤링하고 감정 분석 모델을 통해 긍/부정/중립으로 분류한 뒤, Streamlit 대시보드로 시각화하는 프로젝트입니다.

---

## 🧱 폴더 구조

- `data/` : 뉴스 원본 + 라벨링 데이터 + 폰트
- `model/` : 학습된 모델 (.h5) 및 토크나이저 (.pkl)
- `notebooks/` : 전체 워크플로우 분할
- `app/streamlit_app.py` : Streamlit 대시보드 실행 코드
- `train_model.py` : 모델 학습 스크립트
- `requirements.txt` : 설치 필요 패키지 목록

---

## 🚀 실행 방법

```bash
# 가상환경 준비 (선택)
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# 필수 라이브러리 설치
pip install -r requirements.txt

# 모델 학습 (선택)
python train_model.py

# 대시보드 실행
cd app
streamlit run streamlit_app.py
