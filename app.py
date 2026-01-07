import cv2
import numpy as np
import base64
import time
import os
import signal
import threading
import uuid
from functools import wraps
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

from utils import create_directories, save_capture, extract_id_data, save_to_json, verify_faces

app = Flask(__name__)
CORS(app)
create_directories()

# --- CONFIGURATION ---
API_KEY = "vkyc_secure_key_2026"

# --- MODELS ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

sessions = {}

def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if request.headers.get('X-API-KEY') == API_KEY:
            return f(*args, **kwargs)
        return jsonify({"action": "error", "message": "Unauthorized"}), 401
    return decorated

def base64_to_image(base64_string):
    if "," in base64_string:
        header, encoded = base64_string.split(",", 1)
    else:
        encoded = base64_string
    data = base64.b64decode(encoded)
    np_arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

def check_lighting(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    avg = np.mean(gray)
    if avg < 50: return "Too Dark! Turn on lights."
    if avg > 230: return "Too Bright! Avoid glare."
    return "Good"

def get_image_quality(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    glare = cv2.countNonZero(thresh) / (roi.shape[0] * roi.shape[1]) * 100
    return blur, glare

def get_document_tilt(roi):
    """Calculates the angle of the document to ensure it's straight"""
    try:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours: return 0
        
        # Find the largest contour (the card)
        c = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(c)
        angle = rect[-1]
        
        # Normalize angle to be deviation from horizontal (0 degrees)
        if angle < -45: angle = 90 + angle
        if angle > 45: angle = angle - 90
        
        return angle
    except: return 0

def is_valid_id_card(roi):
    try:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > 5000: return True
        return False
    except: return False

def check_pose_cv(gray):
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60,60))
    if len(faces) > 0: return "center", faces[0]
    profiles = profile_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60,60))
    if len(profiles) > 0: return "left", profiles[0]
    flipped = cv2.flip(gray, 1)
    profiles_flip = profile_cascade.detectMultiScale(flipped, 1.1, 5, minSize=(60,60))
    if len(profiles_flip) > 0: return "right", profiles_flip[0]
    return "none", None

@app.route('/')
def index():
    return "VKYC API Running."

@app.route('/api/start', methods=['POST'])
@require_api_key
def start_session():
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "step": 0, "message": "Initializing...", "face_start_time": 0, "transition_start": 0,
        "captured_face": None, "id_stability": 0, "is_processing": False, 
        "liveness_stage": 0, "blink_timer": 0, "camera_switched": False,
        "final_result": None
    }
    return jsonify({"status": "success", "session_id": session_id})

@app.route('/api/process', methods=['POST'])
@require_api_key
def process_frame():
    data = request.json
    sid = data.get('session_id')
    if not sid or sid not in sessions:
        return jsonify({"action": "error", "message": "Session Invalid", "box": None}), 400

    state = sessions[sid]
    if state["step"] == 4:
        return jsonify({"action": "stop", "message": "Done", "box": None})
    
    try: frame = base64_to_image(data['image'])
    except: return jsonify({"action": "wait", "message": "Image Error", "box": None})

    h, w, _ = frame.shape
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    response = {"action": "scan", "message": "", "box": None}

    # STEP 0: LIGHTING
    if state["step"] == 0:
        light = check_lighting(frame)
        if light == "Good":
            state["step"] = 1; state["liveness_stage"] = 0
            response["message"] = "Light Good. Starting Liveness..."
        else: response["message"] = light

    # STEP 1: LIVENESS
    elif state["step"] == 1:
        pose, _ = check_pose_cv(gray)
        if state["liveness_stage"] == 0:
            response["message"] = "Turn Head RIGHT >>"
            if pose == "right": state["liveness_stage"] = 1; response["message"] = "Good!"
        elif state["liveness_stage"] == 1:
            response["message"] = "Turn Head LEFT <<"
            if pose == "left": state["liveness_stage"] = 2; state["blink_timer"] = time.time(); response["message"] = "Good!"
        elif state["liveness_stage"] == 2:
            response["message"] = "Look Center and BLINK"
            if pose == "center":
                if time.time() - state["blink_timer"] > 2.0:
                    state["step"] = 2; state["face_start_time"] = time.time(); response["message"] = "Verified!"
            else: state["blink_timer"] = time.time()

    # STEP 2: PASSPORT PHOTO
    elif state["step"] == 2:
        box_h = int(h * 0.6); box_w = int(box_h * 0.75)
        cx, cy = w // 2, h // 2
        x1, y1 = cx - box_w // 2, cy - box_h // 2
        response["box"] = [x1, y1, box_w, box_h]
        
        pose, _ = check_pose_cv(gray)
        elapsed = time.time() - state["face_start_time"]
        
        if pose == "center":
            countdown = 3.0 - elapsed
            if countdown <= 0:
                state["captured_face"] = frame[y1:y1+box_h, x1:x1+box_w].copy()
                state["step"] = 3; state["transition_start"] = time.time() + 4.0
                state["camera_switched"] = False
                response["action"] = "flash"; response["message"] = "Photo Taken!"
            else: response["message"] = f"Hold Still... {int(countdown)+1}"
        else: state["face_start_time"] = time.time(); response["message"] = "Look at Camera"

    # STEP 3: ID SCANNING (STRICTER CHECKS)
    elif state["step"] == 3:
        if state["is_processing"]: 
            return jsonify({"action": "wait", "message": "Reading Text...", "box": None})

        wait = state["transition_start"] - time.time()
        
        if wait > 0:
            msg = f"Switching Camera... {int(wait)}"
            if not state["camera_switched"]:
                state["camera_switched"] = True
                return jsonify({"action": "switch_to_back_camera", "message": msg, "box": None})
            return jsonify({"action": "wait", "message": msg, "box": None})

       # --- BIGGER BOX LOGIC ---
        # Allows user to bring camera CLOSER to card
        if w > h: 
            # PC: 70% Height (was 45%)
            box_h = int(h * 0.70)
            box_w = int(box_h * 1.58)
        else:
            # Mobile: 90% Width (was 70%)
            box_w = int(w * 0.90)
            box_h = int(box_w / 1.58)

        x = (w // 2) - (box_w // 2)
        y = (h // 2) - (box_h // 2)
        response["box"] = [x, y, box_w, box_h]  
        
        roi = frame[y:y+box_h, x:x+box_w]
        
        blur_val, glare_val = get_image_quality(roi)
        # We don't check tilt anymore because it was blocking legitimate captures too often

        if not is_valid_id_card(roi):
            state["id_stability"] = 0; response["message"] = "Align ID in Box"
        elif glare_val > 5:
            state["id_stability"] = 0; response["message"] = "Glare Detected! Tilt Card."
            
        # LOWER BLUR THRESHOLD (Real-world fix)
        elif blur_val < 100: 
            state["id_stability"] = 0; response["message"] = f"Too Blurry ({int(blur_val)}) - Hold Still"
            
        else:
            state["id_stability"] += 1
            response["message"] = f"Scanning... {int(state['id_stability']*10)}%"
            
            # INCREASE PATIENCE: Wait 15 frames (~5 seconds) to ensure autofocus
            if state["id_stability"] > 15:
                state["is_processing"] = True
                threading.Thread(target=process_backend_async, args=(sid, roi)).start()

    sessions[sid] = state
    return jsonify(response)

def process_backend_async(sid, roi):
    try:
        if sid not in sessions: return
        state = sessions[sid]
        
        extracted = extract_id_data(roi)
        
        if extracted["doc_type"] == "Unknown Document" and not extracted["doc_id_number"]:
            state["is_processing"] = False
            state["id_stability"] = 0
            sessions[sid] = state
            return

        match, score = verify_faces(state["captured_face"], roi)
        f_path, i_path, ts = save_capture(state["captured_face"], roi)
        
        state["final_result"] = {
            "timestamp": ts, 
            "ocr_data": extracted,
            "verification": { "face_match": bool(match), "confidence": float(score) },
            "images": {"face": f_path, "id": i_path}
        }
        save_to_json(state["final_result"])
        state["step"] = 4
        sessions[sid] = state
        
    except Exception as e:
        print(f"Backend Error: {e}")
        if sid in sessions:
            sessions[sid]["is_processing"] = False
            sessions[sid]["id_stability"] = 0

@app.route('/api/result', methods=['GET'])
@require_api_key
def get_result():
    sid = request.args.get('session_id')
    if sid in sessions and sessions[sid].get("final_result"):
        return jsonify(sessions[sid]["final_result"])
    return jsonify({"status": "processing"})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)