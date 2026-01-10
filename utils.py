import os
import cv2
import easyocr
import json
import re
import torch
import numpy as np
from datetime import datetime
from ultralytics import YOLO
from ultralytics.nn import tasks, modules
from deepface import DeepFace

print("Loading AI Models...")
# --- FULL ALLOWLIST FOR YOLO CHECKPOINT ---
from torch.serialization import add_safe_globals

# --- ALLOWLIST YOLO GLOBALS ---
torch.serialization.add_safe_globals([
    tasks.DetectionModel,
    modules.conv.Conv,
    modules.common.Bottleneck,
    modules.common.Concat,
    modules.common.SPPF,
    modules.common.SPP,
    modules.common.Focus,
    modules.common.DWConv,
    modules.common.C3,
    modules.common.CBL,
    modules.common.ConvBNAct,
    torch.nn.Sequential,
    torch.nn.Conv2d,
    torch.nn.BatchNorm2d,
    torch.nn.Linear,
    torch.nn.ReLU,
    torch.nn.SiLU,
    torch.nn.Sigmoid,
])
model = YOLO('best.pt') 
reader = easyocr.Reader(['en'], gpu=False)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def create_directories():
    os.makedirs("captures/faces", exist_ok=True)
    os.makedirs("captures/ids", exist_ok=True)
    os.makedirs("captures/debug", exist_ok=True)
    os.makedirs("data", exist_ok=True)

def save_debug_data(img, name, text_content=None):
    if img is None or img.size == 0: return
    timestamp = datetime.now().strftime("%H%M%S")
    cv2.imwrite(f"captures/debug/{timestamp}_{name}.jpg", img)
    if text_content:
        with open(f"captures/debug/{timestamp}_{name}.txt", "w", encoding="utf-8") as f:
            f.write(text_content)

def save_capture(face_img, id_img):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if face_img is not None:
        try: face_passport = cv2.resize(face_img, (600, 800), interpolation=cv2.INTER_LANCZOS4)
        except: face_passport = face_img
    else: face_passport = np.zeros((100,100,3), np.uint8)

    f_path = f"captures/faces/face_{timestamp}.jpg"
    i_path = f"captures/ids/id_{timestamp}.jpg"
    cv2.imwrite(f_path, face_passport)
    cv2.imwrite(i_path, id_img)
    return f_path, i_path, timestamp

# --- 1. IMAGE PROCESSING (CLAHE) ---
def preprocess_for_ocr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
    
    gaussian = cv2.GaussianBlur(denoised, (0, 0), 3.0)
    sharpened = cv2.addWeighted(denoised, 1.5, gaussian, -0.5, 0)
    return sharpened

def get_perspective_crop(img):
    try:
        original = img.copy()
        ratio = img.shape[0] / 500.0
        small = cv2.resize(img, (int(img.shape[1]/ratio), int(img.shape[0]/ratio)))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blur, 50, 200)
        cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts[0] if len(cnts) == 2 else cnts[1], key=cv2.contourArea, reverse=True)[:5]
        screenCnt = None
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                screenCnt = approx
                break
        if screenCnt is None: return None
        return img 
    except: return None

# --- 2. INTELLIGENT PARSING ---
def find_best_aadhaar(text_blob):
    space_match = re.search(r'\b\d{4}\s\d{4}\s\d{4}\b', text_blob)
    if space_match: return space_match.group().replace(" ", "")
    
    clean_blob = re.sub(r'[^0-9]', '', text_blob)
    cont_match = re.search(r'\d{12}', clean_blob)
    return cont_match.group() if cont_match else ""

def fix_dob_typo(dob_text):
    match_year = re.search(r'(\d{2})[-/.](\d{2})[-/.](\d{3})\b', dob_text)
    if match_year:
        dob_text = f"{match_year.group(1)}/{match_year.group(2)}/2{match_year.group(3)}"
    parts = re.split(r'[-/.]', dob_text)
    if len(parts) == 3:
        d, m, y = parts
        if m == '40': m = '10' 
        if m == 'L0' or m == 'LO': m = '10'
        return f"{d}/{m}/{y}"
    return dob_text

def clean_person_name(name):
    """
    Cleans up name strings.
    1. Removes digits/symbols.
    2. Fixes common OCR typos in names (1->I, 0->O).
    """
    # Filter only letters and spaces
    clean = re.sub(r'[^A-Za-z ]', '', name).strip()
    
    # Common OCR Corrections for Uppercase Names
    clean = clean.replace('1', 'I').replace('0', 'O').replace('5', 'S')
    
    # "Arpil" -> "Arpit" (Common 'l' vs 't' error at end of name)
    if clean.endswith('l'):
        clean = clean[:-1] + 't'
        
    return clean

def find_name_and_dob_from_full_text(text_lines):
    dob = ""
    name = ""
    dob_index = -1
    
    for i, line in enumerate(text_lines):
        clean_line = line.upper().replace('D08', 'DOB').replace('D0B', 'DOB')
        date_match = re.search(r'\d{1,2}[-/.]\d{1,2}[-/.]\d{3,4}', clean_line)
        
        if date_match:
            dob = fix_dob_typo(date_match.group())
            dob_index = i
            break
        elif "DOB" in clean_line:
            nums = re.findall(r'\d+', clean_line)
            for n in nums:
                if len(n) == 8: 
                    dob = f"{n[:2]}/{n[2:4]}/{n[4:]}"
                    dob_index = i
                    break
    
    if dob_index > 0:
        for offset in [1, 2, 3]:
            if dob_index - offset < 0: break
            candidate = text_lines[dob_index - offset]
            # Must NOT contain digits (rejects "826 23")
            if any(char.isdigit() for char in candidate): continue 
            
            cleaned = clean_person_name(candidate)
            if len(cleaned) > 3:
                name = cleaned
                break 

    return name, dob

def find_face_on_id_card(id_card_img):
    try:
        gray = cv2.cvtColor(id_card_img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
        if len(faces) > 0:
            faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
            x, y, w, h = faces[0]
            return id_card_img[y:y+h, x:x+w]
    except: pass
    return None

def run_specific_box_ocr(roi):
    processed = preprocess_for_ocr(roi)
    try:
        results = reader.readtext(processed, detail=0)
        return " ".join(results)
    except: return ""

# --- 3. MAIN EXTRACTION ---
def extract_id_data(image_input):
    print("\n--- Starting Deep Extraction ---")
    img = cv2.imread(image_input) if isinstance(image_input, str) else image_input

    # 1. Run YOLO
    results = model.predict(source=img, conf=0.15, save=False, verbose=False)
    
    data = {
        "doc_type": "Analyzing...", "first_name": "", "last_name": "", 
        "dob": "", "doc_id_number": "", "raw_ocr": "", "id_photo": None
    }
    
    yolo_name_raw = ""

    # 2. Process YOLO Boxes
    for box in results[0].boxes:
        cls = int(box.cls[0]); name = model.names[cls].lower()
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        roi = img[y1:y2, x1:x2]

        if "photo" in name or "image" in name or cls in [6, 17]:
            data["id_photo"] = roi
            continue
        
        # Capture raw name from box, but don't finalize it yet
        if "name" in name:
            yolo_name_raw = run_specific_box_ocr(roi)
                
        if "dob" in name:
            raw_dob = run_specific_box_ocr(roi)
            date_match = re.search(r'\d{1,2}[-/.]\d{1,2}[-/.]\d{3,4}', raw_dob)
            if date_match: data["dob"] = fix_dob_typo(date_match.group())

    if data["id_photo"] is None:
        data["id_photo"] = find_face_on_id_card(img)

    # 3. FULL SCAN & MERGE
    processed = preprocess_for_ocr(img)
    save_debug_data(processed, "clahe_scan") 
    
    try:
        text_lines = reader.readtext(processed, detail=0)
        full_blob = " ".join(text_lines).upper()
        save_debug_data(img, "extraction_log", "\n".join(text_lines))

        # A. ID Number
        if not data["doc_id_number"]:
            data["doc_id_number"] = find_best_aadhaar(full_blob)
            if data["doc_id_number"]: data["doc_type"] = "Aadhaar Card"
            else:
                pan_match = re.search(r'[A-Z]{5}[0-9]{4}[A-Z]', full_blob.replace(" ",""))
                if pan_match: data["doc_id_number"] = pan_match.group(); data["doc_type"] = "PAN Card"

        # B. Smart Name Merge Logic
        found_name_fs, found_dob_fs = find_name_and_dob_from_full_text(text_lines)
        
        # If YOLO name is garbage ("Api"), try to use Full Scan Name ("Arpit")
        clean_yolo = clean_person_name(yolo_name_raw)
        
        final_name = ""
        # 1. Prefer Full Scan if it looks longer/more complete
        if found_name_fs and len(found_name_fs) > len(clean_yolo):
            final_name = found_name_fs
        # 2. Else use YOLO name if valid
        elif len(clean_yolo) > 3:
            final_name = clean_yolo
        # 3. Fallback to whatever Full Scan found
        elif found_name_fs:
            final_name = found_name_fs
            
        if final_name:
            parts = final_name.split()
            data["first_name"] = parts[0]
            if len(parts) > 1: data["last_name"] = parts[-1]

        # C. DOB Merge
        if not data["dob"] and found_dob_fs:
            data["dob"] = found_dob_fs

        data["raw_ocr"] = full_blob

    except Exception as e:
        print(f"Extraction Error: {e}")

    return data

def verify_faces(live_img, id_card_img_or_crop):
    try:
        res = DeepFace.verify(live_img, id_card_img_or_crop, model_name="VGG-Face", detector_backend="opencv", enforce_detection=False)
        return res['verified'], (1 - res['distance']) * 100
    except: return False, 0.0

def save_to_json(data):
    clean = data.copy()
    if "ocr_data" in clean and "id_photo" in clean["ocr_data"]:
        del clean["ocr_data"]["id_photo"]
    file_path = "data/extracted_data.json"
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            try: h = json.load(f)
            except: h = []
    else: h = []
    h.append(clean)
    with open(file_path, 'w') as f: json.dump(h, f, indent=4)
