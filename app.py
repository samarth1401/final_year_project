import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
import easyocr
import numpy as np
import pandas as pd
from PIL import Image
import datetime
import os
import asyncio

# ✅ Must be first Streamlit command
st.set_page_config(
    page_title="Helmet & Number Plate Detection",
    page_icon="🚦",
    layout="centered",
    initial_sidebar_state="auto"
)

# ✅ Hide only the Deploy button, keep menu (three dots)
st.markdown("""
    <style>
    button[title="Deploy this app"] {
        display: none !important;
        visibility: hidden !important;
    }
    </style>
""", unsafe_allow_html=True)

# Fix for asyncio event loop (Windows only)
asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Load YOLO model
model = YOLO("runs/detect/train7/weights/best.pt")

# Load EasyOCR
reader = easyocr.Reader(['en'], gpu=False)

# Save path for OCR results
save_file = "detected_numbers/ocr_results.txt"
os.makedirs(os.path.dirname(save_file), exist_ok=True)
if not os.path.exists(save_file):
    with open(save_file, "w") as f:
        f.write("Timestamp\tPlate Number\n")

# UI Input Type
option = st.radio("Choose input type:", ("Image", "Video"))

# Model Accuracy
def get_accuracy_info():
    try:
        df = pd.read_csv("runs/detect/train7/results.csv")
        last_row = df.iloc[-1]
        return last_row['metrics/mAP_0.5'], last_row['metrics/mAP_0.5:0.95']
    except:
        return None, None

map50, map5095 = get_accuracy_info()
if map50:
    st.markdown("### 📊 Model Accuracy")
    st.write(f"**mAP@0.5:** {map50:.2f}")
    st.write(f"**mAP@0.5:0.95:** {map5095:.2f}")

# Detection + OCR
def detect_and_display(frame, timestamp=None):
    if frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

    results = model(frame)
    detected_numbers = []

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            label = result.names[cls]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Expand bounding box
            margin = 15
            h, w, _ = frame.shape
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(w, x2 + margin)
            y2 = min(h, y2 + margin)

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

            if label == 'number plate':
                cropped = frame[y1:y2, x1:x2]
                cropped = cv2.copyMakeBorder(cropped, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])
                gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
                gray = cv2.GaussianBlur(gray, (5, 5), 0)
                dilated = cv2.dilate(gray, np.ones((2, 2), np.uint8), iterations=1)

                ocr_result = reader.readtext(dilated, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", rotation_info=[0])
                print("Raw OCR:", ocr_result)

                combined_plate = ""
                for (_, text, prob) in ocr_result:
                    text = text.strip().upper().replace(" ", "")
                    print(f"Detected OCR Text: {text} with Confidence: {prob:.2f}")
                    if prob > 0.10 and len(text) >= 1:
                        combined_plate += text

                if len(combined_plate) >= 3:
                    detected_numbers.append(combined_plate)
                    cv2.putText(frame, f"Plate: {combined_plate}", (x1, y2 + 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                    with open(save_file, "a") as f:
                        f.write(f"{timestamp or datetime.datetime.now()}\t{combined_plate}\n")

    return frame, detected_numbers

# Handle image input
if option == "Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        frame = np.array(image)
        result_img, numbers = detect_and_display(frame)
        st.image(result_img, caption="Detection Result", use_container_width=True)
        if numbers:
            st.success(f"Detected Plate Number: {', '.join(numbers)}")
        else:
            st.error("❌ No plate number detected.")

# Handle video input
else:
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()
        plate_log = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            frame, numbers = detect_and_display(frame, timestamp)
            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
            plate_log.extend([(timestamp, num) for num in numbers])

        cap.release()

        if plate_log:
            st.markdown("### Detected Plate Numbers")
            for t, num in plate_log:
                st.write(f"{t} — {num}")
        else:
            st.error("❌ No plate numbers detected.")
