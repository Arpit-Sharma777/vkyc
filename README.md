# V-KYC System (AI-Powered Video Verification)

A full-stack Video KYC system that performs real-time liveness detection, smart ID card scanning (Aadhaar, PAN, etc.), OCR extraction, and Face Matching using Computer Vision and Deep Learning.

## 🚀 Key Features

* **Liveness Detection:** Anti-spoofing checks using head gestures (Turn Left/Right) and blink detection.
* **Smart ID Scanning:**
    * **Auto-Detection:** Automatically identifies Aadhaar, PAN, Passport, or DL.
    * **CLAHE Enhancement:** "Scanner Mode" preprocessing to read text from low-light or blurry mobile photos.
    * **Smart Crop:** Removes background noise and hands for cleaner data.
* **Intelligent OCR:**
    * **Error Correction:** Fixes common OCR typos (e.g., `0` vs `O`, `1` vs `I`).
    * **Context Logic:** Cross-references data boxes with full-text scans to ensure 100% accuracy on Names and DOBs.
* **Face Verification:** Matches the user's live selfie with the photo extracted from their ID card.

## 🛠️ Tech Stack

* **Backend:** Python (Flask)
* **Computer Vision:** OpenCV, Ultralytics YOLOv8
* **OCR Engine:** EasyOCR
* **Face Analysis:** DeepFace (VGG-Face model)
* **Frontend:** HTML5, CSS3, JavaScript (Canvas API)

## 📂 Project Structure

```text
vkyc-project/
├── app.py                 # Main Flask API Server
├── utils.py               # Core Logic (OCR, Face Match, Image Processing)
├── requirements.txt       # Python Dependencies
├── best.pt                # Trained YOLO Model
├── .gitignore             # Git ignore rules
├── templates/
│   └── index.html         # Frontend Interface
├── captures/              # (Auto-generated) Stores temp images
└── data/                  # (Auto-generated) Stores JSON results

Installation Guide
1. Clone the Repository

git clone [https://github.com/your-username/vkyc-system.git](https://github.com/your-username/vkyc-system.git)
cd vkyc-system

2. Create a Virtual Environment

# Windows
python -m venv .venv
.venv\Scripts\activate

# Mac/Linux
python3 -m venv .venv
source .venv/bin/activate

3. Install Dependencies
pip install -r requirements.txt

How to Run
1. Start the Backend Server Open your terminal in the project folder and run:
python app.py
You should see: Running on http://0.0.0.0:5001

2. Launch the Frontend Open your web browser and go to:

Local PC: http://localhost:5001

Mobile Testing: Connect your phone to the same Wi-Fi and visit http://YOUR_PC_IP:5001 (e.g., http://192.168.1.5:5001).

User Verification Flow
1.Permission: Allow camera access (requests 4K resolution if available).
2.Liveness Check: Follow on-screen instructions (Turn Right -> Turn Left -> Blink).
3.Selfie: The system captures a high-quality selfie automatically.
4.ID Scan:
        -Align your ID card within the green box.
        -Mobile: The system automatically switches to the back camera.
        -Stability Check: The app waits for the camera to hold steady and focus before auto-capturing.
5. Result: The system extracts data and verifies the face match.        