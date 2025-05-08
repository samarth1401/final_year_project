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
from dotenv import load_dotenv
from twilio.rest import Client
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import logging
import warnings

# Suppress warnings and logs
logging.getLogger('streamlit.runtime.scriptrunner').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)

# ✅ Load environment variables
load_dotenv()

# ✅ Streamlit config
st.set_page_config(page_title="Helmet & Number Plate Detection", page_icon="🚦", layout="centered")
st.title("🚨 Helmet Violation & Number Plate Detection System using OCR")

# ✅ Fix for Windows asyncio
asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# ✅ Load models
model = YOLO("runs/detect/train7/weights/best.pt")
reader = easyocr.Reader(['en'], gpu=False)

# ✅ Result file
os.makedirs("detected_numbers", exist_ok=True)
save_file = os.path.join("detected_numbers", "ocr_results.txt")
if not os.path.exists(save_file):
    with open(save_file, "w") as f:
        f.write("Timestamp\tPlate Number\n")

# ✅ MongoDB config
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["database"]
users_collection = db["user_details"]
challans_collection = db["challans"]

# ✅ Twilio config
TWILIO_SID = os.getenv("TWILIO_SID")
TWILIO_TOKEN = os.getenv("TWILIO_TOKEN")
TWILIO_FROM = os.getenv("TWILIO_FROM_PHONE")
twilio_client = Client(TWILIO_SID, TWILIO_TOKEN)

# ✅ Email config
EMAIL_HOST = os.getenv("EMAIL_HOST")
EMAIL_PORT = int(os.getenv("EMAIL_PORT"))
EMAIL_USER = os.getenv("EMAIL_HOST_USER")
EMAIL_PASS = os.getenv("EMAIL_HOST_PASSWORD")

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

# ✅ Detection function
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
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

            if label == 'number plate':
                cropped = frame[y1:y2, x1:x2]
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
                    print(f"[INFO] Detected Number Plate: {combined_plate}")
                    cv2.putText(frame, f"Plate: {combined_plate}", (x1, y2 + 35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                    with open(save_file, "a") as f:
                        f.write(f"{timestamp or datetime.datetime.now()}\t{combined_plate}\n")

    return frame, detected_numbers

# ✅ Email sender
def send_email(to_email, plate, fine_amount, previous_fine, total_due):
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_USER
        msg['To'] = to_email
        msg['Subject'] = f"🚨 Traffic Violation - {plate}"
        body = f"""
        <h2>Traffic Violation Notice</h2>
        <p><strong>Vehicle Number:</strong> {plate}</p>
        <p><strong>Violation:</strong> No Helmet</p>
        <p><strong>Fine:</strong> ₹{fine_amount}</p>
        <p><strong>Previous Dues:</strong> ₹{previous_fine}</p>
        <p><strong>Total Due:</strong> ₹{total_due}</p>
        <p><strong>Location:</strong> MG Road, Bengaluru</p>
        <p><strong>Date & Time:</strong> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        """
        msg.attach(MIMEText(body, 'html'))

        with smtplib.SMTP(EMAIL_HOST, EMAIL_PORT) as server:
            server.starttls()
            server.login(EMAIL_USER, EMAIL_PASS)
            server.send_message(msg)
            print(f"Email sent to {to_email}")
    except Exception as e:
        print(f"Email error: {e}")

# ✅ SMS sender
def send_sms(to_phone, plate, fine_amount, previous_fine, total_due):
    try:
        message = twilio_client.messages.create(
            body=(
                f"Traffic Violation (No Helmet)\n"
                f"Plate: {plate}\n"
                f"Fine: ₹{fine_amount}, Prev: ₹{previous_fine}, Total: ₹{total_due}"
            ),
            from_=TWILIO_FROM,
            to=to_phone
        )
        print(f"SMS sent to {to_phone}: SID {message.sid}")
    except Exception as e:
        print(f"SMS error: {e}")

# ✅ Challan generator
def create_challan(plate):
    user = users_collection.find_one({"vehicle_no": plate})
    if not user:
        st.warning(f"⚠️ No user found for vehicle number: {plate}")
        return

    user_id = user["_id"]
    now = datetime.datetime.now()
    start_of_day = datetime.datetime.combine(now, datetime.time.min)
    end_of_day = datetime.datetime.combine(now, datetime.time.max)

    count_today = challans_collection.count_documents({
        "user": user_id,
        "violation_datetime": {"$gte": start_of_day, "$lte": end_of_day}
    })
    if count_today >= 9000:
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
        "violation_datetime": now,
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

    send_sms(user["phone_no"], plate, challan_doc["fine_amount"], challan_doc["previous_fine_amount"], challan_doc["total_fine_due"])
    send_email(user["email"], plate, challan_doc["fine_amount"], challan_doc["previous_fine_amount"], challan_doc["total_fine_due"])

    st.success(f"✅ Challan created for {plate}")
    st.markdown(f"""
    ### 🚨 Challan Summary
    - **Name:** {user.get("name", "N/A")}
    - **Phone Number:** {user.get("phone_no", "N/A")}
    - **Vehicle Number:** {created_challan['vehicle_no']}
    - **Fine Amount:** ₹{created_challan['fine_amount']}
    - **Previous Dues:** ₹{created_challan['previous_fine_amount']}
    - **Total Fine Due:** ₹{created_challan['total_fine_due']}
    - **Violation:** {created_challan['violation_type']}
    - **Date & Time:** {created_challan['violation_datetime'].strftime('%Y-%m-%d %H:%M:%S')}
    - **Location:** {created_challan['location']['address']}
    - **Status:** {created_challan['challan_status']}
    """)

# ✅ Streamlit UI
option = st.radio("Choose input type:", ("Image", "Video", "Live Webcam"))

if option == "Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        frame = np.array(image)
        result_img, numbers = detect_and_display(frame)
        st.image(result_img, caption="Detection Result", use_container_width=False, width=700)
        for plate in numbers:
            create_challan(plate)

elif option == "Video":
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
        for t, num in plate_log[:1]:
            st.write(f"First Detected Plate: {num}")
            create_challan(num)

elif option == "Live Webcam":
    st.info("🔴 Using Mobile IP Webcam")
    mobile_ip = "http://192.168.29.141:8080/video"
    cap = cv2.VideoCapture(mobile_ip)

    if not cap.isOpened():
        st.error("❌ Failed to connect to mobile camera. Please check IP and ensure app is running.")
    else:
        stframe = st.empty()
        plate_log = []
        prev_plate = ""
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % 10 == 0:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                processed_frame, numbers = detect_and_display(frame, timestamp)
                for num in numbers:
                    if len(num) >= 5 and difflib.SequenceMatcher(None, num, prev_plate).ratio() < 0.85:
                        st.write(f"[LIVE] {timestamp} - Plate: {num}")
                        create_challan(num)
                        prev_plate = num
                stframe.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), channels="RGB", width=700)
            else:
                stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", width=700)

            frame_count += 1

        cap.release()
