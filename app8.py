import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

# 세션 상태 변수 초기화
if 'user_input' not in st.session_state:
    st.session_state.user_input = ""
# Set page configuration
st.set_page_config(page_title="포트폴리오 챗봇!", page_icon="🤖", layout="centered")

# Custom CSS for green theme
st.markdown("""
<style>
    .stApp {
        background-color: #e6f3e6;
    }
    .stTextInput > div > div > input {
        background-color: #ffffff;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    .user-message {
        background-color: #a5d6a7;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .bot-message {
        background-color: #c8e6c9;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# 임베딩 모델 로드
@st.cache(allow_output_mutation=True)
def load_model():
    return SentenceTransformer('jhgan/ko-sroberta-multitask')

encoder = load_model()

# 오디오 파일 경로 설정
AUDIO_DIR = "C:\\chat"
AUDIO_FILE = "minji.wav"

# 포트폴리오 관련 질문과 답변 데이터
questions = [
    "포트폴리오 주제가 무엇인가요?",
    "모델은 어떤 걸 썼나요?",
    "프로젝트 기간은 어떻게 되나요?",
    "조장이 누구인가요",
    "데이터는 무엇을 이용했나요?",
    "힘든 점은 없었나요?"
]

answers = [
    "tts를 활용한 심리상담 챗봇 구현하기 입니다.",
    "bert, lstm 주로 자연어처리가 가능한 모델을 사용했습니다.",
    "총 3주로, 기획과 구현, 발표 준비 등으로 구성했습니다.",
    "조장은 유재현 입니다.",
    "연세대 세브란스 정신과 상담 데이터와 facebook 데이터, 직접 녹음한 음성 데이터를 활용했습니다.",
    "텍스트를 원하는 음성으로 구현하는 것과 답변 자료 데이터셋을 확장시키는 것이 다소 어려웠지만 좋은 출력물을 낼 수 있었습니다."
]

# 질문 임베딩과 답변 데이터프레임 생성
@st.cache(allow_output_mutation=True)
def create_dataframe():
    question_embeddings = encoder.encode(questions)
    audio_paths = [os.path.join(AUDIO_DIR, AUDIO_FILE) if q == "힘든 점은 없었나요?" else None for q in questions]
    return pd.DataFrame({
        'question': questions, 
        '챗봇': answers, 
        'embedding': list(question_embeddings),
        'audio': audio_paths
    })

df = create_dataframe()

# 챗봇 함수 정의
def get_response(user_input):
    embedding = encoder.encode(user_input)
    df['distance'] = df['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())
    answer = df.loc[df['distance'].idxmax()]
    audio_path = answer['audio']
    
    if audio_path and os.path.exists(audio_path):
        st.session_state.history.append({"user": user_input, "bot": answer['챗봇'], "audio": audio_path})
    else:
        st.session_state.history.append({"user": user_input, "bot": answer['챗봇'], "audio": None})

# (제출 콜백 함수는 그대로 유지)

# Streamlit 인터페이스
st.title("🤖 포트폴리오 챗봇")
st.write("포트폴리오에 관한 질문을 입력해보세요. 예: 포트폴리오 주제가 무엇인가요? 하하...")

# 이미지 표시
st.image("heart2.png", caption="Welcome to the Portfolio Chatbot", use_column_width=True)

# 폼 생성
with st.form(key='chat_form'):
    st.text_input("질문을 입력하세요:", key='temp_input', value=st.session_state.user_input)
    submit_button = st.form_submit_button(label='제출', on_click=submit_callback)

# 대화 이력 표시 및 오디오 재생
for message in st.session_state.history:
    st.markdown(f"<div class='user-message'><b>사용자</b>: {message['user']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='bot-message'><b>챗봇</b>: {message['bot']}</div>", unsafe_allow_html=True)
    
    # 음성 파일 재생 (힘든 점 질문에 대해서만)
    if message.get('audio') and os.path.exists(message['audio']):
        try:
            st.audio(message['audio'], format="audio/wav")
        except Exception as e:
            st.error(f"오디오 파일을 재생할 수 없습니다: {str(e)}")
