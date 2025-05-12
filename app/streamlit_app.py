import os
import pandas as pd
import numpy as np
import re
import pickle
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import streamlit as st
from matplotlib import font_manager, rc

# ✅ 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "preprocessed_labeled.csv")
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "sentiment_model.keras")
TOKENIZER_PATH = os.path.join(BASE_DIR, "..", "model", "tokenizer.pkl")
FONT_PATH = os.path.join(BASE_DIR, "..", "data", "NanumGothic.ttf")

# ✅ 폰트 설정
font_name = font_manager.FontProperties(fname=FONT_PATH).get_name()
rc('font', family=font_name)
plt.rcParams['font.family'] = font_name  # WordCloud 한글깨짐 방지

# ✅ Streamlit UI 설정
st.set_page_config(page_title="부동산 뉴스 감정 분석", layout="wide")
st.markdown(f"<h1 style='text-align:center; color:#1A535C;'>🏠 부동산 뉴스 감정 분석 대시보드</h1>", unsafe_allow_html=True)
st.markdown("<hr style='border: 1px solid #eee;'>", unsafe_allow_html=True)

# ✅ 모델/토크나이저 로딩
@st.cache_resource
def load_resources():
    model = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_resources()

# ✅ 전처리 함수
stopwords = ["의", "가", "이", "은", "들", "는", "좀", "잘", "걍", "과", "도", "를", "으로", "자", "에", "와", "한", "하다"]

def clean_text(text):
    text = re.sub(r"[^가-힣0-9\s]", "", str(text))
    return re.sub(r"\s+", " ", text).strip()

def tokenize(text):
    return [word for word in clean_text(text).split() if word not in stopwords]

def preprocess(text):
    tokens = tokenize(text)
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

# ✅ 수동 샘플 추가
manual = pd.DataFrame([
    {"clean_title": "서울 아파트 분양 완판, 실수요자 몰려", "label": 1},
    {"clean_title": "정부 부동산 대책 효과 기대", "label": 1},
    {"clean_title": "전세 사기 피해자 속출, 부동산 시장 불신", "label": 0},
    {"clean_title": "집값 하락세 지속, 투자자들 손실 우려", "label": 0}
])
df = pd.concat([df, manual], ignore_index=True)

# ✅ 레이아웃 2컬럼
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📊 감정 분포")
    label_counts = Counter(df["label"])
    full_counts = [label_counts.get(i, 0) for i in range(3)]
    colors = ['#FF6B6B', '#4ECDC4', '#1A535C']
    fig, ax = plt.subplots()
    ax.pie(full_counts, labels=["부정", "긍정", "중립"], colors=colors, autopct="%.1f%%", startangle=90)
    ax.axis("equal")
    st.pyplot(fig)

with col2:
    st.markdown("### 🌤️ 긍정 뉴스 워드 클라우드")
    if df[df["label"] == 1].shape[0] > 0:
        positive_text = " ".join(df[df["label"] == 1]["clean_title"])
        wc = WordCloud(font_path=FONT_PATH, background_color="white", width=500, height=300).generate(positive_text)
        st.image(wc.to_array(), use_column_width=True)
    else:
        st.info("긍정 뉴스가 아직 없습니다.")

# ✅ 사용자 입력 감정 예측
st.markdown("<hr style='border: 1px solid #eee;'>", unsafe_allow_html=True)
st.markdown("### 🧠 실시간 뉴스 감정 예측")

with st.container():
    user_input = st.text_input("✏️ 분석할 뉴스 제목을 입력하세요:", key="sentiment_input")
    if user_input:
        tokens = tokenize(user_input)
        oov_count = sum(1 for word in tokens if word not in tokenizer.word_index)
        oov_ratio = oov_count / len(tokens) if tokens else 0

        if oov_ratio > 0.5:
            st.warning("⚠️ 입력된 단어 대부분이 학습되지 않아 예측 정확도가 낮을 수 있습니다.")
        elif oov_ratio > 0:
            st.info(f"ℹ️ 입력에 {oov_count}개의 미학습(OOV) 단어가 포함되어 있습니다.")

        x_input = preprocess(user_input)
        pred = model.predict(x_input)
        label = np.argmax(pred)
        label_text = {0: "부정", 1: "긍정", 2: "중립"}[label]

        st.success(f"🎯 예측 감정 결과: **{label_text}**")
        st.markdown(f"예측 확률: `{[round(p, 3) for p in pred[0]]}`")

        new_row = pd.DataFrame([{"clean_title": user_input, "label": label}])
        st.session_state.df = pd.concat([st.session_state.df, new_row], ignore_index=True)
