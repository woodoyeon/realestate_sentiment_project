import os
import pandas as pd
import numpy as np
import re
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
from konlpy.tag import Okt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import streamlit as st
from matplotlib import font_manager, rc

# ✅ 기본 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "preprocessed_labeled.csv")
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "sentiment_model.keras")
TOKENIZER_PATH = os.path.join(BASE_DIR, "..", "model", "tokenizer.pkl")
FONT_PATH = os.path.join(BASE_DIR, "..", "data", "NanumGothic.ttf")

# ✅ 한글 폰트 설정
font_name = font_manager.FontProperties(fname=FONT_PATH).get_name()
rc('font', family=font_name)

# ✅ Streamlit UI 설정
st.set_page_config(page_title="부동산 감정 분석", layout="centered")
st.title("📊 부동산 뉴스 감정 분석 대시보드")

# ✅ 모델/토크나이저 불러오기
@st.cache_resource
def load_resources():
    model = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_resources()

# ✅ 형태소 분석기 및 불용어
okt = Okt()
stopwords = ["의", "가", "이", "은", "들", "는", "좀", "잘", "걍", "과", "도", "를", "으로", "자", "에", "와", "한", "하다"]

def clean_text(text):
    text = re.sub(r"[^가-힣0-9\\s]", "", str(text))
    text = re.sub(r"\\s+", " ", text).strip()
    return text

def preprocess(text):
    tokens = [word for word in okt.morphs(clean_text(text)) if word not in stopwords]
    seq = tokenizer.texts_to_sequences([" ".join(tokens)])
    return pad_sequences(seq, maxlen=30)

# ✅ 데이터 로딩
@st.cache_data
def load_dataframe():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        return df.dropna(subset=["clean_title", "label"])
    return pd.DataFrame(columns=["clean_title", "label"])

if "df" not in st.session_state:
    st.session_state.df = load_dataframe()

df = st.session_state.df

# ✅ 테스트용 긍정/부정 샘플 수동 추가
manual = pd.DataFrame([
    {"clean_title": "서울 아파트 분양 완판, 실수요자 몰려", "label": 1},
    {"clean_title": "정부 부동산 대책 효과 기대", "label": 1},
    {"clean_title": "전세 사기 피해자 속출, 부동산 시장 불신", "label": 0},
    {"clean_title": "집값 하락세 지속, 투자자들 손실 우려", "label": 0}
])
df = pd.concat([df, manual], ignore_index=True)

# ✅ 감정 분포 시각화
st.subheader("📌 감정 분포")
label_counts = Counter(df["label"])
full_counts = [label_counts.get(i, 0) for i in range(3)]
fig, ax = plt.subplots()
ax.pie(full_counts, labels=["부정", "긍정", "중립"], autopct="%.1f%%", startangle=90)
ax.axis("equal")
st.pyplot(fig)

# ✅ 긍정 뉴스 워드클라우드
st.subheader("📌 워드 클라우드 (긍정 뉴스)")
if df[df["label"] == 1].shape[0] > 0:
    positive_text = " ".join(df[df["label"] == 1]["clean_title"])
    wc = WordCloud(font_path=FONT_PATH, background_color="white").generate(positive_text)
    st.image(wc.to_array())
else:
    st.info("긍정 뉴스가 아직 없습니다.")

# ✅ 사용자 입력 감정 예측
st.subheader("🧠 새 뉴스 감정 예측")
user_input = st.text_input("뉴스 제목을 입력하세요:", key="sentiment_input")

if user_input:
    tokens = [word for word in okt.morphs(clean_text(user_input)) if word not in stopwords]
    oov_count = sum(1 for word in tokens if word not in tokenizer.word_index)
    oov_ratio = oov_count / len(tokens) if tokens else 0

    if oov_ratio > 0.5:
        st.warning("⚠️ 입력하신 문장에 학습되지 않은 단어가 많아 예측 정확도가 낮을 수 있습니다.")
    elif oov_ratio > 0:
        st.info(f"ℹ️ 입력에 {oov_count}개의 OOV 단어가 포함되어 있습니다.")

    x_input = preprocess(user_input)
    pred = model.predict(x_input)
    label = np.argmax(pred)
    label_text = {0: "부정", 1: "긍정", 2: "중립"}[label]
    st.success(f"✅ 예측된 감정: {label_text}")
    st.write("📊 예측 확률:", pred[0].tolist())

    new_row = pd.DataFrame([{"clean_title": user_input, "label": label}])
    st.session_state.df = pd.concat([st.session_state.df, new_row], ignore_index=True)
