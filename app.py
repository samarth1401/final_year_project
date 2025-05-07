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
import difflib
from pymongo import MongoClient
from bson.objectid import ObjectId
from dotenv import load_dotenv

# ✅ Load environment variables
load_dotenv()

# ✅ Set Streamlit page config
st.set_page_config(
    page_title="Helmet & Number Plate Detection",
    page_icon="🚦",
    layout="centered",
    initial_sidebar_state="auto"
)

# ✅ Hide "Deploy" button
st.markdown("""
    <style>
    button[title="Deploy this app"] {
        display: none !important;
        visibility: hidden !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🚨 Helmet Violation & Number Plate Detection System using OCR")

# ✅ Fix asyncio loop for Windows
asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# ✅ Load YOLO model
model = YOLO("runs/detect/train7/weights/best.pt")

# ✅ Load EasyOCR
reader = easyocr.Reader(['en'], gpu=False)

# ✅ OCR result file
os.makedirs("detected_numbers", exist_ok=True)
save_file = os.path.join("detected_numbers", "ocr_results.txt")
if not os.path.exists(save_file):
    with open(save_file, "w") as f:
        f.write("Timestamp\tPlate Number\n")

# ✅ MongoDB setup
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["database"]
users_collection = db["user_details"]
challans_collection = db["challans"]

# ✅ Accuracy info
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

# ✅ Detection + OCR logic
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

            margin = 15
            h, w, _ = frame.shape
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(w, x2 + margin)
            y2 = min(h, y2 + margin)

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

                combined_plate = ""
                for (_, text, prob) in ocr_result:
                    text = text.strip().upper().replace(" ", "")
                    if prob > 0.10 and len(text) >= 1:
                        combined_plate += text

                if len(combined_plate) >= 3:
                    detected_numbers.append(combined_plate)
                    cv2.putText(frame, f"Plate: {combined_plate}", (x1, y2 + 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                    with open(save_file, "a") as f:
                        f.write(f"{timestamp or datetime.datetime.now()}\t{combined_plate}\n")

    return frame, detected_numbers

# ✅ Challan creation logic
def create_challan(plate):
    user = users_collection.find_one({"vehicle_no": plate})
    if not user:
        st.warning(f"⚠️ No user found for vehicle number: {plate}")
        return

    user_id = user["_id"]

    start_of_day = datetime.datetime.combine(datetime.datetime.today(), datetime.time.min)
    end_of_day = datetime.datetime.combine(datetime.datetime.today(), datetime.time.max)
    count_today = challans_collection.count_documents({
        "user": user_id,
        "violation_datetime": {"$gte": start_of_day, "$lte": end_of_day}
    })

    if count_today >= 5:
        st.warning(f"⚠️ Max 5 challans already issued today for {plate}")
        return

    previous_challans = list(challans_collection.find({"user": user_id}))
    previous_fine = sum(ch["fine_amount"] for ch in previous_challans if not ch.get("is_paid", False))

    challan_doc = {
        "user": user_id,
        "vehicle_no": user["vehicle_no"],
        "fine_amount": 500,
        "previous_fine_amount": previous_fine,
        "total_fine_due": previous_fine + 500,
        "violation_type": "No Helmet",
        "violation_datetime": datetime.datetime.now(),
        "location": {
            "latitude": 12.9716,
            "longitude": 77.5946,
            "address": "Detected via CCTV at MG Road, Bengaluru"
        },
        "image_proof_url": "/images/sample-violation.jpg",
        "officer_in_charge": "Automated System",
        "challan_status": "Pending"
    }

    inserted = challans_collection.insert_one(challan_doc)
    created_challan = challans_collection.find_one({"_id": inserted.inserted_id})
    created_challan["user_detail"] = {
        "name": user.get("name", "N/A"),
        "vehicle_no": user.get("vehicle_no")
    }

    st.info(f"✅ Challan created for {plate}")
    st.json(created_challan)

# ✅ UI input option
option = st.radio("Choose input type:", ("Image", "Video"))

if option == "Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        frame = np.array(image)
        result_img, numbers = detect_and_display(frame)
        st.image(result_img, caption="Detection Result", use_container_width=False, width=700)

        if numbers:
            for plate in numbers:
                create_challan(plate)
        else:
            st.error("❌ No plate number detected.")

else:
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()
        plate_log = []
        prev_plate = ""

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            frame, numbers = detect_and_display(frame, timestamp)

            for num in numbers:
                if len(num) >= 5 and difflib.SequenceMatcher(None, num, prev_plate).ratio() < 0.85:
                    plate_log.append((timestamp, num))
                    prev_plate = num

            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", width=700)

        cap.release()

        if plate_log:
            for t, num in plate_log[:1]:  # Just take first for challan creation
                st.write(f"First Detected Plate: {num}")
                create_challan(num)

            st.markdown("### Detected Plate Numbers")
            for t, num in plate_log:
                st.write(f"{t} — {num}")
        else:
            st.error("❌ No plate numbers detected.")
