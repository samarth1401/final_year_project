
# 🚨 Helmet Violation & Number Plate Detection System

This project is an automated traffic law enforcement system that detects helmet violations and extracts number plate text using **YOLOv8** and **EasyOCR**. It also logs the information in a MongoDB database and sends e-challans to violators via **Twilio (SMS)** and **email with PDF**.

GitHub Repository: [https://github.com/samarth1401/final_year_project.git](https://github.com/samarth1401/final_year_project.git)

---

## 📌 Features

- 🎥 Real-time video processing with OpenCV
- 🧠 Helmet detection using YOLOv8 (`ultralytics`)
- 🔍 Number plate recognition using EasyOCR
- 📤 Streamlit-based user interface
- 🗃️ Data storage in MongoDB
- ✉️ E-challan generation in PDF format
- 📲 SMS notifications via Twilio
- 📧 Email alerts with PDF attachments

---

## 🧰 Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python (OpenCV, YOLO, EasyOCR, MongoDB)
- **Database**: MongoDB (via pymongo)
- **Notifications**: Twilio (SMS), SMTP (Email)

---

## ⚙️ Installation Guide

### ✅ Step 1: Clone the Repository

```bash
git clone https://github.com/samarth1401/final_year_project.git
cd final_year_project
```

### ✅ Step 2: Create a Virtual Environment

```bash
python -m venv .venv
```

#### Activate the Virtual Environment

- **For Windows**:
```bash
.venv\Scripts\activate
```

- **For Linux/Mac**:
```bash
source .venv/bin/activate
```

### ✅ Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

Make sure your `requirements.txt` includes the following:

```
streamlit
opencv-python
ultralytics
easyocr
numpy
pandas
Pillow
pymongo
python-dotenv
twilio
fpdf
```

---

## 🛠️ Environment Configuration

### ✅ Step 4: Create `.env` File

Create a `.env` file in the root directory and add the following credentials:

```
MONGO_URI=your_mongodb_connection_uri
TWILIO_SID=your_twilio_account_sid
TWILIO_AUTH_TOKEN=your_twilio_auth_token
TWILIO_PHONE=your_twilio_registered_number
EMAIL_ID=your_email@gmail.com
EMAIL_PASS=your_email_password_or_app_password
```

> ⚠️ Use app-specific password if 2FA is enabled for your email.

---

## 🚀 Run the Application

```bash
streamlit run app.py
```

> Streamlit will automatically open your browser. If not, go to [http://localhost:8501](http://localhost:8501)

---

## 📂 Project Structure

```
final_year_project/
├── app.py
├── requirements.txt
├── .env
├── detected_numbers/
│   └── (saved plates and images)
├── pdfs/
│   └── (generated challans)
```

---

## 📸 Output Snapshots

- 🚨 Helmet violation detection with bounding boxes
- 🧾 Auto-generated challan in PDF format
- 📧 Email sent with attached PDF
- 📲 SMS alert using Twilio

---

## 🧠 How It Works

1. Capture video input using OpenCV.
2. Detect helmet presence using YOLOv8.
3. Extract number plate using EasyOCR.
4. Log the data into MongoDB.
5. Generate PDF challan and send it via Email and SMS.


