import streamlit as st
from ultralytics import YOLO
import tempfile
import cv2
import os

# 전체 레이아웃을 넓게 설정
st.set_page_config(layout="wide")

# 제목 설정
st.title("비디오 사물 검출 앱")

# 파일 업로드 버튼을 상단으로 이동
uploaded_file = st.file_uploader("비디오 파일을 업로드하세요", type=["mp4", "mov", "avi"])

# 전체 레이아웃을 컨테이너로 감싸기
with st.container():
    col1, col2 = st.columns(2)  # 열을 균등하게 분배하여 넓게 표시

    with col1:
        st.header("원본 영상")
        if uploaded_file is not None:
            st.video(uploaded_file)
        else:
            st.write("원본 영상을 표시하려면 비디오 파일을 업로드하세요.")

    with col2:
        st.header("사물 검출 결과 영상")
        # 사물 검출 결과가 나타날 자리 확보 및 고정 높이 회색 박스 스타일 추가
        result_placeholder = st.empty()
        if "processed_video" in st.session_state and st.session_state["processed_video"] is not None:
            result_placeholder.video(st.session_state["processed_video"])
        else:
            result_placeholder.markdown(
                """
                <div style='width:100%; height:620px; background-color:#d3d3d3; display:flex; align-items:center; justify-content:center; border-radius:5px;'>
                    <p style='color:#888;'>여기에 사물 검출 결과가 표시됩니다.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

# 사물 검출 버튼 클릭 이벤트 처리
if st.button("사물 검출 실행") and uploaded_file and model_file: # 버튼을 눌렀고 영상도 업로드 했고 모델 어쩌구 
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_output:
        output_path = temp_output.name # 임시 비디오 파일을 생성하고 output_path에 저장해라 

    with tempfile.NamedTemporaryFile(delete=False) as temp_input:
        temp_input.write(uploaded_file.read())
        temp_input_path = temp_input.name

    cap = cv2.VideoCapture(temp_input_path) # 원본 비디오를 cap을 생성해서 읽을 준비를 합니다. 
    fourcc = cv2.VideoWriter_fourcc(*'XVID') # XVID 코덱을 사용합니다. 
    fps = cap.get(cv2.CAP_PROP_FPS) # 원본 비디오의 속도를 가져옵니다. 
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # 원본 비디오 해상도 넓이 
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # 원본 비디오 해상도 높이 
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height)) 
    # out을 생성해서 YOLO 모델 결과를 기록할 비디오 파일을 준비합니다. 

    frame_count = 0 # 프레임 수를 기록하기 위한 변수 생성
    while cap.isOpened(): # 비디오가 끝날 때까지
        ret, frame = cap.read() # 프레임을 하나씩 읽어옵니다. 
        if not ret:             # 더 이상 읽을 프레임이 없으면
            break               # 종료해라 

        # YOLO 모델로 예측 수행 및 디버깅
        results = model(frame) # 모델에 frame을 넣어서 객체를 검출합니다. 
        detections = results[0].boxes if len(results) > 0 else []

        if len(detections) > 0: # 만약 detections에 값이 있다면
            for box in detections:  # 박스 바운딩을 수행합니다. 
                x1, y1, x2, y2 = map(int, box.xyxy[0]) # 박스 바운딩 4개의 좌표 
                confidence = box.conf[0] # 바운딩에 해당 물체가 맞을 확률 
                class_id = int(box.cls[0])
                class_name = model.names[class_id] # 클래스 이름 (선수 이름)
                label = f"{class_name} {confidence:.2f}" # 그 선수일 확률 

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # 사각형을 그립니다.
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            # 검출 결과가 없을 때 로그 출력
            st.write(f"Frame {frame_count}: No detections")

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()

    # 결과 비디오를 st.session_state에 저장하여 스트림릿에 표시
    st.session_state["processed_video"] = output_path
    result_placeholder.video(output_path)
    st.success("사물 검출이 완료되어 오른쪽에 표시됩니다.")
