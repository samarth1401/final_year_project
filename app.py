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
from fpdf import FPDF
from email.mime.application import MIMEApplication




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

            margin = 20
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

def generate_challan_pdf(challan_doc, user):
    pdf = FPDF()
    pdf.add_page()

    # Title with font and police color
    pdf.set_font("Arial", 'B', 16)
    pdf.set_text_color(0, 0, 128)
    pdf.cell(200, 10, txt="Davangere Traffic Police", ln=True, align='C')

    pdf.set_font("Arial", 'B', 14)
    pdf.set_text_color(220, 50, 50)
    pdf.cell(200, 10, txt="Traffic Violation Challan", ln=True, align='C')
    pdf.set_text_color(0, 0, 0)
    pdf.ln(10)

    # Basic Information
    pdf.set_font("Arial", '', 12)
    data = [
        ("Challan No.", str(challan_doc['_id'])),
        ("Violation Date & Time", challan_doc['violation_datetime'].strftime('%d-%b-%Y %H:%M')),
        ("Vehicle Number", challan_doc['vehicle_no']),
        ("License Number", "N/A"),
        ("Violation", challan_doc['violation_type']),
        ("Fine Amount (Rs.)", str(challan_doc['fine_amount'])),
        ("Previous Dues (Rs.)", str(challan_doc['previous_fine_amount'])),
        ("Total Fine Due (Rs.)", str(challan_doc['total_fine_due'])),
        ("Location", challan_doc['location']['address']),
        ("Officer In Charge", challan_doc['officer_in_charge']),
        ("Status", challan_doc['challan_status']),
    ]
    for key, value in data:
        pdf.cell(60, 10, txt=f"{key}:", border=0)
        pdf.cell(100, 10, txt=value, border=0, ln=True)

    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Violation Evidence Image:", ln=True)

    # Add Image
    if 'image_path' in challan_doc and os.path.exists(challan_doc['image_path']):
        pdf.image(challan_doc['image_path'], x=60, w=90)
    else:
        pdf.set_font("Arial", '', 10)
        pdf.cell(200, 10, txt="(No image found)", ln=True)

    pdf.ln(10)
    pdf.set_font("Arial", 'I', 9)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(200, 10, txt="Note: This is a system-generated challan. No signature required.", ln=True, align='C')

    # Reset color for future documents
    pdf.set_text_color(0, 0, 0)

    filepath = os.path.join("detected_numbers", f"challan_{challan_doc['_id']}.pdf")
    pdf.output(filepath)
    return filepath


# ✅ Email sender

def send_email(to_email, plate, fine_amount, previous_fine, total_due, challan_doc, user):
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_USER
        msg['To'] = to_email
        msg['Subject'] = f"🚨 Traffic Violation - {plate}"

        body = f"""
        <h2>Traffic Violation Notice</h2>
        <p><strong>Vehicle Number:</strong> {plate}</p>
        <p><strong>Violation:</strong> No Helmet</p>
        <p><strong>Fine:</strong> Rs.{fine_amount}</p>
        <p><strong>Previous Dues:</strong> Rs.{previous_fine}</p>
        <p><strong>Total Due:</strong> Rs.{total_due}</p>
        <p><strong>Location:</strong> MG Road, Bengaluru</p>
        <p><strong>Date & Time:</strong> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        """
        msg.attach(MIMEText(body, 'html'))

        # attach pdf
        pdf_path = generate_challan_pdf(challan_doc, user)
        with open(pdf_path, "rb") as f:
            part = MIMEApplication(f.read(), _subtype="pdf")
            part.add_header('Content-Disposition', 'attachment', filename=os.path.basename(pdf_path))
            msg.attach(part)

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
        if not to_phone.startswith('+91'):
            to_phone = '+91' + to_phone

        message = twilio_client.messages.create(
            body=(
                f"Traffic Violation (No Helmet)\n"
                f"Plate: {plate}\n"
                f"Fine: Rs.{fine_amount}, Prev: Rs.{previous_fine}, Total: Rs.{total_due}"
            ),
            from_=TWILIO_FROM,
            to=to_phone
        )
        print(f"SMS sent to {to_phone}: SID {message.sid}")
    except Exception as e:
        print(f"SMS error: {e}")

# ✅ Challan generator
def create_challan(plate, violation_frame=None):
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
    image_path = f"detected_numbers/violation_{plate}_{now.strftime('%Y%m%d%H%M%S')}.jpg"
    if violation_frame is not None:
        cv2.imwrite(image_path, violation_frame)

    challan_doc = {
        "user": user_id,
        "vehicle_no": user["vehicle_no"],
        "fine_amount": 500,
        "previous_fine_amount": previous_fine,
        "total_fine_due": previous_fine + 500,
        "violation_type": "No Helmet",
        "violation_datetime": now,
        "image_path": image_path,
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

    # send_sms(user["phone_no"], plate, challan_doc["fine_amount"], challan_doc["previous_fine_amount"], challan_doc["total_fine_due"])
    send_email(user["email"], plate, challan_doc["fine_amount"], challan_doc["previous_fine_amount"], challan_doc["total_fine_due"], challan_doc, user)

    st.success(f"✅ Challan created for {plate}")
    st.markdown(f"""
    ### 🚨 Challan Summary
    - **Name:** {user.get("name", "N/A")}
    - **Phone Number:** {user.get("phone_no", "N/A")}
    - **Vehicle Number:** {created_challan['vehicle_no']}
    - **Fine Amount:** Rs.{created_challan['fine_amount']}
    - **Previous Dues:** Rs.{created_challan['previous_fine_amount']}
    - **Total Fine Due:** Rs.{created_challan['total_fine_due']}
    - **Violation:** {created_challan['violation_type']}
    - **Date & Time:** {created_challan['violation_datetime'].strftime('%Y-%m-%d %H:%M:%S')}
    - **Location:** {created_challan['location']['address']}
    - **Status:** {created_challan['challan_status']}
    """)

    pdf_path = generate_challan_pdf(created_challan, user)
    with open(pdf_path, "rb") as f:
        st.download_button(
            label="📄 Download Challan PDF",
            data=f.read(),
            file_name=os.path.basename(pdf_path),
            mime="application/pdf"
        )

    try:
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
        if 'image_path' in created_challan and os.path.exists(created_challan['image_path']):
            os.remove(created_challan['image_path'])
    except Exception as e:
         st.warning(f"⚠️ Could not delete temporary files: {e}")

# ✅ Streamlit UI
option = st.radio("Choose input type:", ("Image", "Video", "Live Webcam"))

if option == "Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        frame = np.array(image)
        result_img, numbers = detect_and_display(frame)
        st.image(result_img, caption="Detection Result", use_container_width=False, width=300)
        for plate in numbers:
            create_challan(plate,frame)

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
            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", width=200)
        cap.release()
        for t, num in plate_log[:1]:
            st.write(f"First Detected Plate: {num}")
            create_challan(num,frame)

elif option == "Live Webcam":
    st.info("🔴 Using Mobile IP Webcam")
    mobile_ip = os.getenv("MOBILE_IP", "http://192.168.1.11:8080/video")
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
                        create_challan(num,frame)
                        prev_plate = num
                stframe.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), channels="RGB", width=700)
            else:
                stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", width=700)

            frame_count += 1

        cap.release()
