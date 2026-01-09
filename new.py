# lifeline_full_runnable_updated.py
# Full runnable LifeLine ID app (single-file)
# Updated: DOB calendar auto-fills age and strict fingerprint-only capture & verification tweaks.
# Includes: webcam capture + file upload, strict fingerprint detector, DB storage, ORB duplicate check.

import os
import cv2
import datetime
import hashlib
import mysql.connector
from mysql.connector import Error
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import numpy as np
import traceback
import base64
import io

# Optional tkcalendar
try:
    from tkcalendar import Calendar
    TKCAL_AVAILABLE = True
except Exception:
    TKCAL_AVAILABLE = False

# ---------------- CONFIG ----------------
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "root",
    "database": "lifelineid_new"
}

FINGER_DIR = "fingerprints"
REGISTER_DIR = os.path.join(FINGER_DIR, "registered")
VERIFY_DIR = os.path.join(FINGER_DIR, "verified")
ENHANCED_DIR = os.path.join(FINGER_DIR, "enhanced")
ROI_DIR = os.path.join(FINGER_DIR, "roi")
os.makedirs(REGISTER_DIR, exist_ok=True)
os.makedirs(VERIFY_DIR, exist_ok=True)
os.makedirs(ENHANCED_DIR, exist_ok=True)
os.makedirs(ROI_DIR, exist_ok=True)

# Matching thresholds (conservative)
MIN_GOOD_MATCHES = 20
MIN_MATCH_RATIO = 0.12
DUPLICATE_MATCH_RATIO = 0.22
ROI_SIZE = 300

# Fingerprint detection thresholds (tunable)
THRESH_EDGE_LOW = 0.02
THRESH_EDGE_HIGH = 0.5
THRESH_TEXTURE_VAR = 60.0
THRESH_KP_COUNT = 60
THRESH_BLUR = 8.0
THRESH_COHERENCE = 0.28
THRESH_GABOR_MEAN = 6.5
THRESH_LBP_VAR = 0.35  # normalized LBP variance threshold (fingerprints have characteristic LBP variance)

BLOOD_GROUPS = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]

# ---------------- DB HELPERS ----------------
def get_db_connection():
    return mysql.connector.connect(**DB_CONFIG)

def ensure_db_schema():
    """Create minimal tables if they don't exist."""
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        # users_new stores fingerprint paths and hash and vector (base64)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users_new (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(200),
                age INT,
                dob DATE,
                blood_group_id INT,
                phone VARCHAR(20),
                address TEXT,
                aadhar VARCHAR(20),
                doctor_id INT,
                emergency_contact_id INT,
                fingerprint_path VARCHAR(500),
                fingerprint_hash VARCHAR(128),
                enhanced_path VARCHAR(500),
                roi_path VARCHAR(500),
                fingerprint_vector LONGBLOB,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            ) ENGINE=InnoDB
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS blood_groups (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(10) UNIQUE
            ) ENGINE=InnoDB
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS doctors (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(200),
                phone VARCHAR(20) UNIQUE
            ) ENGINE=InnoDB
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS emergency_contacts (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(200),
                relation VARCHAR(50),
                phone VARCHAR(20) UNIQUE
            ) ENGINE=InnoDB
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS verification_logs (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT,
                verified_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users_new(id) ON DELETE CASCADE
            ) ENGINE=InnoDB
        """)
        conn.commit()
        cur.close()
    except Exception as e:
        print("Error ensuring DB schema:", e)
    finally:
        if conn:
            conn.close()

def get_or_create_blood_group(cur, conn, name):
    cur.execute("SELECT id FROM blood_groups WHERE name=%s", (name,))
    row = cur.fetchone()
    if row:
        return row[0]
    cur.execute("INSERT INTO blood_groups (name) VALUES (%s)", (name,))
    conn.commit()
    return cur.lastrowid

def get_or_create_doctor(cur, conn, name, phone):
    if not phone:
        phone = None
    cur.execute("SELECT id FROM doctors WHERE phone=%s", (phone,))
    row = cur.fetchone()
    if row:
        return row[0]
    cur.execute("INSERT INTO doctors (name, phone) VALUES (%s, %s)", (name, phone))
    conn.commit()
    return cur.lastrowid

def get_or_create_emergency(cur, conn, name, relation, phone):
    if not phone:
        phone = None
    cur.execute("SELECT id FROM emergency_contacts WHERE phone=%s", (phone,))
    row = cur.fetchone()
    if row:
        return row[0]
    cur.execute("INSERT INTO emergency_contacts (name, relation, phone) VALUES (%s, %s, %s)", (name, relation, phone))
    conn.commit()
    return cur.lastrowid

def get_fingerprint_hash(img_path):
    try:
        with open(img_path, "rb") as f:
            data = f.read()
        return hashlib.sha256(data).hexdigest()
    except Exception:
        return None

# ---------------- Image helpers & fingerprint detectors ----------------
def auto_contrast(img_gray):
    if img_gray is None or img_gray.size == 0:
        return img_gray
    p_low, p_high = np.percentile(img_gray, (2, 98))
    if p_high - p_low <= 0:
        return img_gray
    stretched = np.clip((img_gray - p_low) * 255.0 / (p_high - p_low), 0, 255).astype(np.uint8)
    return stretched

def lbp_image(gray):
    # simple LBP (8,1) uniform-less implementation
    h, w = gray.shape
    lbp = np.zeros_like(gray, dtype=np.uint8)
    for y in range(1, h-1):
        for x in range(1, w-1):
            center = gray[y, x]
            code = 0
            code |= (1 << 0) if gray[y-1, x-1] >= center else 0
            code |= (1 << 1) if gray[y-1, x] >= center else 0
            code |= (1 << 2) if gray[y-1, x+1] >= center else 0
            code |= (1 << 3) if gray[y, x+1] >= center else 0
            code |= (1 << 4) if gray[y+1, x+1] >= center else 0
            code |= (1 << 5) if gray[y+1, x] >= center else 0
            code |= (1 << 6) if gray[y+1, x-1] >= center else 0
            code |= (1 << 7) if gray[y, x-1] >= center else 0
            lbp[y, x] = code
    return lbp

def compute_metrics(roi_bgr):
    try:
        if roi_bgr is None or roi_bgr.size == 0:
            return None
        if len(roi_bgr.shape) == 3:
            gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi_bgr.copy()
        gray = cv2.resize(gray, (300, 300))
        gray = auto_contrast(gray)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

        median_val = np.median(blurred)
        lower = int(max(5, 0.66 * median_val))
        upper = int(min(220, 1.33 * median_val))

        edges = cv2.Canny(blurred, lower, upper)
        edge_ratio = np.sum(edges > 0) / edges.size

        texture_var = float(np.var(enhanced))

        orb = cv2.ORB_create(nfeatures=1500)
        kps = orb.detect(enhanced, None)
        kp_count = len(kps) if kps is not None else 0

        blur_metric = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        coherence = compute_structure_coherence(enhanced)
        gabor_mean = compute_gabor_mean_response(enhanced)

        # LBP variance: compute normalized variance of LBP histogram
        lbp = lbp_image(enhanced)
        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0,255))
        hist = hist.astype(float) / (hist.sum() + 1e-9)
        lbp_var = float(np.var(hist))

        return {
            "edge_ratio": edge_ratio,
            "texture_var": texture_var,
            "kp_count": kp_count,
            "blur_metric": blur_metric,
            "canny_low": lower,
            "canny_high": upper,
            "coherence": coherence,
            "gabor_mean": gabor_mean,
            "lbp_var": lbp_var,
            "enhanced": enhanced
        }
    except Exception:
        return None

def compute_structure_coherence(gray):
    try:
        gray_f = gray.astype(np.float32) / 255.0
        Ix = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
        Iy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
        Ixx = Ix * Ix
        Iyy = Iy * Iy
        Ixy = Ix * Iy
        ksize = (7, 7)
        Sxx = cv2.GaussianBlur(Ixx, ksize, 2)
        Syy = cv2.GaussianBlur(Iyy, ksize, 2)
        Sxy = cv2.GaussianBlur(Ixy, ksize, 2)
        tmp = np.sqrt(np.maximum(((Sxx - Syy) * 0.5) ** 2 + Sxy ** 2, 0))
        l1 = (Sxx + Syy) * 0.5 + tmp
        l2 = (Sxx + Syy) * 0.5 - tmp
        eps = 1e-6
        coherence_map = (l1 - l2) / (l1 + l2 + eps)
        coherence_map = np.nan_to_num(coherence_map)
        return float(np.mean(coherence_map))
    except Exception:
        return 0.0

def compute_gabor_mean_response(gray):
    try:
        gray_f = gray.astype(np.float32) / 255.0
        responses = []
        thetas = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        for theta in thetas:
            kernel = cv2.getGaborKernel(ksize=(21, 21), sigma=4.0, theta=theta, lambd=10.0, gamma=0.5, psi=0)
            filtered = cv2.filter2D(gray_f, cv2.CV_32F, kernel)
            responses.append(np.mean(np.abs(filtered)))
        return float(np.mean(responses))
    except Exception:
        return 0.0

def is_fingerprint(roi_bgr):
    """Return True only when ROI strongly resembles fingerprint ridges/texture."""
    m = compute_metrics(roi_bgr)
    if m is None:
        return False
    cond_edges = (THRESH_EDGE_LOW < m["edge_ratio"] < THRESH_EDGE_HIGH)
    cond_texture = (m["texture_var"] > THRESH_TEXTURE_VAR)
    cond_kp = (m["kp_count"] > THRESH_KP_COUNT)
    cond_blur = (m["blur_metric"] > THRESH_BLUR)
    cond_coh = (m["coherence"] >= THRESH_COHERENCE)
    cond_gab = (m["gabor_mean"] >= THRESH_GABOR_MEAN)
    cond_lbp = (m["lbp_var"] >= THRESH_LBP_VAR)

    true_count = sum([cond_edges, cond_texture, cond_kp, cond_blur, cond_coh, cond_gab, cond_lbp])
    # require at least 6 of 7 checks pass to be strict
    return true_count >= 6

# ---------------- ORB matching & duplicate ----------------
def orb_match_score(img1_gray, img2_gray):
    try:
        orb = cv2.ORB_create(nfeatures=1200)
        kp1, des1 = orb.detectAndCompute(img1_gray, None)
        kp2, des2 = orb.detectAndCompute(img2_gray, None)
        if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
            return 0, 0.0
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.knnMatch(des1, des2, k=2)
        good = []
        for m_n in matches:
            if len(m_n) != 2:
                continue
            m, n = m_n
            if m.distance < 0.75 * n.distance:
                good.append(m)
        good_count = len(good)
        denom = max(1, min(len(kp1), len(kp2)))
        ratio = good_count / denom
        return good_count, ratio
    except Exception:
        return 0, 0.0

def is_duplicate_fingerprint(new_fp_path, debug_preview=False):
    if not os.path.exists(new_fp_path):
        return None, 0.0
    new_hash = get_fingerprint_hash(new_fp_path)
    if new_hash is None:
        return None, 0.0
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT id, name, fingerprint_hash, fingerprint_path FROM users_new WHERE fingerprint_hash=%s", (new_hash,))
        row = cur.fetchone()
        if row:
            cur.close()
            conn.close()
            return row[1], 1.0
        cur.execute("SELECT id, name, fingerprint_path FROM users_new WHERE fingerprint_path IS NOT NULL")
        rows = cur.fetchall()
        cur.close()
        conn.close()
        new_img = cv2.imread(new_fp_path, cv2.IMREAD_GRAYSCALE)
        if new_img is None:
            return None, 0.0
        new_img = cv2.resize(new_img, (300, 300))
        new_img = auto_contrast(new_img)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        new_img = clahe.apply(new_img)
        best_name = None
        best_ratio = 0.0
        for uid, uname, fp_path in rows:
            try:
                if not fp_path or not os.path.exists(fp_path):
                    continue
                existing_img = cv2.imread(fp_path, cv2.IMREAD_GRAYSCALE)
                if existing_img is None:
                    continue
                existing_img = cv2.resize(existing_img, (300, 300))
                existing_img = auto_contrast(existing_img)
                existing_img = clahe.apply(existing_img)
                good_count, ratio = orb_match_score(new_img, existing_img)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_name = uname
                if ratio >= DUPLICATE_MATCH_RATIO:
                    return uname, ratio
            except Exception:
                continue
        return (best_name, best_ratio) if best_ratio >= DUPLICATE_MATCH_RATIO else (None, 0.0)
    except Exception:
        if conn:
            conn.close()
        return None, 0.0

# ---------------- STRICT Webcam capture (no bypass) ----------------
def capture_fingerprint_preview(save_dir,
                                window_title="Place finger in box and press 's' to capture, 'q' to cancel",
                                auto_save=True):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Webcam Error", "Cannot open webcam. Check device/permissions and close other apps using camera.")
        return None

    saved_path = None
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            cx, cy = w // 2, h // 2
            half = ROI_SIZE // 2
            x1, y1 = max(0, cx - half), max(0, cy - half)
            x2, y2 = min(w, cx + half), min(h, cy + half)

            frame_display = frame.copy()
            overlay = frame_display.copy()
            alpha = 0.20
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 128), -1)
            cv2.addWeighted(overlay, alpha, frame_display, 1 - alpha, 0, frame_display)
            cv2.rectangle(frame_display, (x1, y1), (x2, y2), (0, 255, 128), 2)

            cv2.putText(frame_display, "Align fingertip inside the box.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            cv2.putText(frame_display, "Press 's' to capture, 'q' to cancel", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            roi = frame[y1:y2, x1:x2].copy()
            metrics = compute_metrics(roi)
            if metrics is not None:
                info_y = 90
                cv2.putText(frame_display, f"edge_ratio: {metrics['edge_ratio']:.4f}", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1); info_y += 18
                cv2.putText(frame_display, f"texture_var: {metrics['texture_var']:.1f}", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1); info_y += 18
                cv2.putText(frame_display, f"kp_count: {metrics['kp_count']}", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1); info_y += 18
                cv2.putText(frame_display, f"blur: {metrics['blur_metric']:.1f}", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1); info_y += 18
                cv2.putText(frame_display, f"coherence: {metrics['coherence']:.3f}", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230,230,230), 1); info_y += 18
                cv2.putText(frame_display, f"gabor: {metrics['gabor_mean']:.3f}", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230,230,230), 1); info_y += 18
                cv2.putText(frame_display, f"lbp_var: {metrics['lbp_var']:.4f}", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230,230,230), 1)

            cv2.imshow(window_title, frame_display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):
                # strict check
                if not is_fingerprint(roi):
                    cv2.putText(frame_display, "NOT A FINGERPRINT — TRY AGAIN", (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                    cv2.imshow(window_title, frame_display)
                    cv2.waitKey(800)
                    continue

                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (300, 300))
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"fp_{ts}.jpg"
                saved_path = os.path.join(save_dir := save_dir_if_exists(save_dir=REGISTER_DIR, default_dir=REGISTER_DIR), filename)
                cv2.imwrite(saved_path, gray)
                # also save enhanced and roi
                metrics = compute_metrics(gray)
                enhanced = metrics["enhanced"] if metrics and "enhanced" in metrics else gray
                enhanced_path = os.path.join(ENHANCED_DIR, f"enh_{ts}.jpg")
                roi_path = os.path.join(ROI_DIR, f"roi_{ts}.jpg")
                cv2.imwrite(enhanced_path, enhanced)
                cv2.imwrite(roi_path, gray)
                if auto_save:
                    messagebox.showinfo("Captured", f"Fingerprint captured and saved:\n{saved_path}")
                break

            elif key == ord('q'):
                saved_path = None
                break

    except Exception as e:
        print("Error during strict capture:", e)
        traceback.print_exc()
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return saved_path

def save_dir_if_exists(save_dir, default_dir):
    return save_dir if save_dir and os.path.exists(save_dir) else default_dir

# ---------------- Verification & matching ----------------
def fetch_all_users():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT u.id, u.name, u.age, u.dob, bg.name, u.phone, u.address, u.aadhar,
               d.name, d.phone, ec.name, ec.relation, ec.phone, u.fingerprint_path
        FROM users_new u
        LEFT JOIN blood_groups bg ON u.blood_group_id = bg.id
        LEFT JOIN doctors d ON u.doctor_id = d.id
        LEFT JOIN emergency_contacts ec ON u.emergency_contact_id = ec.id
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows

def find_best_match(captured_path):
    captured = cv2.imread(captured_path, cv2.IMREAD_GRAYSCALE)
    if captured is None:
        return None, 0, 0.0
    captured = cv2.resize(captured, (300, 300))
    captured = auto_contrast(captured)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    captured = clahe.apply(captured)

    users = fetch_all_users()
    best_user = None
    best_score = 0
    best_ratio = 0.0
    for row in users:
        fp_path = row[-1]
        if not fp_path or not os.path.exists(fp_path):
            continue
        stored = cv2.imread(fp_path, cv2.IMREAD_GRAYSCALE)
        if stored is None:
            continue
        stored = cv2.resize(stored, (300, 300))
        stored = auto_contrast(stored)
        stored = clahe.apply(stored)
        good_count, ratio = orb_match_score(captured, stored)
        if (good_count > best_score) or (good_count == best_score and ratio > best_ratio):
            best_score = good_count
            best_ratio = ratio
            best_user = row
    return best_user, best_score, best_ratio

def log_verification(user_id):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("INSERT INTO verification_logs (user_id) VALUES (%s)", (user_id,))
    conn.commit()
    cur.close()
    conn.close()

def get_monthly_verification_count(user_id):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT COUNT(*) FROM verification_logs
        WHERE user_id = %s AND MONTH(verified_at) = MONTH(CURRENT_DATE())
        AND YEAR(verified_at) = YEAR(CURRENT_DATE())
    """, (user_id,))
    count = cur.fetchone()[0]
    cur.close()
    conn.close()
    return count

# ---------------- Insert user with duplicate protections ----------------
def insert_user_normalized(details: dict, fp_path: str, enhanced_path: str = None, roi_path: str = None, debug_preview=False):
    if not fp_path or not os.path.exists(fp_path):
        raise Error("Fingerprint image not found for insertion.")

    fp_hash = get_fingerprint_hash(fp_path)
    if not fp_hash:
        raise Error("Could not compute fingerprint hash.")
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, name FROM users_new WHERE fingerprint_hash=%s", (fp_hash,))
    existing = cur.fetchone()
    if existing:
        cur.close(); conn.close()
        raise Error(f"Duplicate fingerprint hash detected! Already registered to: {existing[1]} (User ID: {existing[0]})")

    dup_name, dup_ratio = is_duplicate_fingerprint(fp_path, debug_preview)
    if dup_name:
        cur.close(); conn.close()
        raise Error(f"Duplicate fingerprint pattern detected (match ratio {dup_ratio*100:.2f}%). Already registered to: {dup_name}")

    bg_id = get_or_create_blood_group(cur, conn, details.get("blood_group") or "Unknown")
    doctor_id = get_or_create_doctor(cur, conn, details.get("doctor_name"), details.get("doctor_phone"))
    ec_id = get_or_create_emergency(cur, conn, details.get("emergency_name"), details.get("emergency_relation"), details.get("emergency_phone"))

    # compute a compact fingerprint vector (ORB descriptors average hashed -> base64)
    fp_vec_b64 = None
    try:
        img = cv2.imread(fp_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (300,300))
            img = auto_contrast(img)
            orb = cv2.ORB_create(nfeatures=800)
            kp, des = orb.detectAndCompute(img, None)
            if des is not None:
                # reduce via mean descriptor (not ideal but compact)
                vec = np.mean(des.astype(np.float32), axis=0)
                fp_vec_b64 = base64.b64encode(vec.tobytes())
    except Exception:
        fp_vec_b64 = None

    sql = ("""
        INSERT INTO users_new
        (name, age, dob, blood_group_id, phone, address, aadhar,
         doctor_id, emergency_contact_id, fingerprint_path, fingerprint_hash, enhanced_path, roi_path, fingerprint_vector, created_at)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,NOW())
    """)
    params = (
        details.get("name"),
        int(details.get("age")) if details.get("age") and str(details.get("age")).isdigit() else None,
        details.get("dob"),
        bg_id,
        details.get("phone"),
        details.get("address"),
        details.get("aadhar"),
        doctor_id,
        ec_id,
        fp_path,
        fp_hash,
        enhanced_path,
        roi_path,
        fp_vec_b64
    )
    cur.execute(sql, params)
    conn.commit()
    uid = cur.lastrowid
    cur.close()
    conn.close()
    return uid

# ---------------- UI: Splash ----------------
class SplashScreen(tk.Toplevel):
    def __init__(self, root, delay=900):
        super().__init__(root)
        root.withdraw()
        self.overrideredirect(True)
        self.configure(bg="#17252a")
        w, h = 420, 220
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        x, y = (sw - w) // 2, (sh - h) // 2
        self.geometry(f"{w}x{h}+{x}+{y}")

        tk.Label(self, text="🩺 LifeLine ID", font=("Segoe UI", 20, "bold"), fg="white", bg="#17252a").pack(pady=(16, 6))
        tk.Label(self, text="Secure Fingerprint Healthcare Identity", font=("Segoe UI", 10), fg="#def2f1", bg="#17252a").pack()
        self.progress = ttk.Progressbar(self, mode="indeterminate", length=300)
        self.progress.pack(pady=18)
        self.progress.start()
        self.after(delay, self.close_splash)

    def close_splash(self):
        self.progress.stop()
        self.destroy()
        self.master.deiconify()

# ---------------- UI: Calendar helper ----------------
def open_calendar(entry_widget, age_var):
    win = tk.Toplevel()
    win.title("Select DOB")
    win.geometry("320x340")
    if not TKCAL_AVAILABLE:
        lbl = tk.Label(win, text="tkcalendar not installed.\nInstall: pip install tkcalendar", fg="red")
        lbl.pack(pady=30)
        return

    cal = Calendar(win, selectmode="day", date_pattern="yyyy-mm-dd")
    cal.pack(pady=10, expand=True, fill="both")

    def select_date():
        date_str = cal.get_date()
        entry_widget.delete(0, tk.END)
        entry_widget.insert(0, date_str)
        try:
            dob = datetime.datetime.strptime(date_str, "%Y-%m-%d")
            today = datetime.datetime.today()
            age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
            try:
                age_var.set(str(age))
            except Exception:
                try:
                    age_var.delete(0, tk.END)
                    age_var.insert(0, str(age))
                except Exception:
                    pass
        except Exception:
            pass
        win.destroy()

    btn = ttk.Button(win, text="Select", command=select_date)
    btn.pack(pady=8)

# ---------------- UI: Main App ----------------
class LifeLineApp:
    def __init__(self, root):
        self.root = root
        root.title("LifeLine ID - Polished")
        root.geometry("1000x650")
        root.configure(bg="#ECF0F1")

        header = tk.Frame(root, bg="#2C3E50", height=70)
        header.pack(fill="x")
        tk.Label(header, text="🩺 LifeLine ID", font=("Segoe UI", 22, "bold"), fg="white", bg="#2C3E50").pack(pady=10)

        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TNotebook.Tab", font=("Segoe UI", 11, "bold"), padding=[18, 8])

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True, padx=15, pady=15)

        self.reg_frame = tk.Frame(self.notebook, bg="white")
        self.ver_frame = tk.Frame(self.notebook, bg="white")
        self.tracker_frame = tk.Frame(self.notebook, bg="white")

        self.notebook.add(self.reg_frame, text="Register")
        self.notebook.add(self.ver_frame, text="Verify")
        self.notebook.add(self.tracker_frame, text="Tracker")

        self._build_register_tab()
        self._build_verify_tab()
        self._build_tracker_tab()

        self.last_register_fp = None
        self.last_verify_fp = None

    # Register tab
    def _build_register_tab(self):
        container = ttk.Frame(self.reg_frame)
        container.pack(fill="both", expand=True, padx=12, pady=12)

        form_canvas = tk.Canvas(container, bg="white", highlightthickness=0, width=640)
        vscroll = ttk.Scrollbar(container, orient="vertical", command=form_canvas.yview)
        form_frame = ttk.Frame(form_canvas)
        form_frame.bind("<Configure>", lambda e: form_canvas.configure(scrollregion=form_canvas.bbox("all")))
        form_canvas.create_window((0, 0), window=form_frame, anchor="nw")
        form_canvas.configure(yscrollcommand=vscroll.set)
        form_canvas.pack(side="left", fill="both", expand=True)
        vscroll.pack(side="left", fill="y")

        labels = [
            ("Name", "name"), ("Age", "age"), ("DOB", "dob"),
            ("Blood Group", "blood_group"), ("Phone", "phone"), ("Address", "address"),
            ("Aadhar", "aadhar"), ("Doctor Name", "doctor_name"), ("Doctor Phone", "doctor_phone"),
            ("Emergency Name", "emergency_name"), ("Emergency Relation", "emergency_relation"),
            ("Emergency Phone", "emergency_phone"),
        ]
        self.entries = {}
        for i, (lab, key) in enumerate(labels):
            ttk.Label(form_frame, text=lab).grid(row=i, column=0, sticky="w", pady=6, padx=6)
            if key == "address":
                txt = tk.Text(form_frame, width=36, height=4, relief="solid", bd=1)
                txt.grid(row=i, column=1, pady=6, padx=6, sticky="w")
                self.entries[key] = txt
            elif key == "blood_group":
                cb = ttk.Combobox(form_frame, values=BLOOD_GROUPS, state="readonly", width=36)
                cb.grid(row=i, column=1, pady=6, padx=6)
                cb.set(BLOOD_GROUPS[0])
                self.entries[key] = cb
            elif key == "dob":
                sv = tk.StringVar()
                dob_entry = ttk.Entry(form_frame, textvariable=sv, width=30)
                dob_entry.grid(row=i, column=1, pady=6, padx=6, sticky="w")
                self.entries[key] = sv
                age_var = self.entries.get("age")
                if age_var is None:
                    age_var = tk.StringVar()
                    self.entries["age"] = age_var
                btn = ttk.Button(form_frame, text="📅", width=4, command=lambda e=dob_entry, a=age_var: open_calendar(e, a))
                btn.grid(row=i, column=1, sticky="e", padx=6)
            else:
                sv = tk.StringVar()
                ttk.Entry(form_frame, textvariable=sv, width=40).grid(row=i, column=1, pady=6, padx=6, sticky="w")
                self.entries[key] = sv

        # age display
        age_row_index = None
        for idx, (_, key) in enumerate(labels):
            if key == "age":
                age_row_index = idx
                break
        if age_row_index is not None:
            age_var = self.entries.get("age", tk.StringVar())
            if isinstance(age_var, tk.StringVar):
                age_display = ttk.Entry(form_frame, textvariable=age_var, width=40, state="readonly")
                age_display.grid(row=age_row_index, column=1, pady=6, padx=6, sticky="w")
                self.entries["age"] = age_var

        # capture/upload buttons
        ttk.Button(form_frame, text="📸 Capture Fingerprint (Webcam)", command=self._capture_register_button).grid(row=len(labels), column=0, pady=12)
        ttk.Button(form_frame, text="📁 Upload Fingerprint Image", command=self._upload_register_fp).grid(row=len(labels), column=1, pady=12)
        ttk.Button(form_frame, text="✅ Register", command=self._register_button).grid(row=len(labels)+1, column=1, pady=12)

        self.fp_preview_label = ttk.Label(form_frame, text="No fingerprint captured", relief="sunken", width=50)
        self.fp_preview_label.grid(row=len(labels)+2, column=0, columnspan=2, pady=10)

    def _upload_register_fp(self):
        file_path = filedialog.askopenfilename(title="Select fingerprint image", filetypes=[("Images","*.png;*.jpg;*.jpeg")])
        if not file_path:
            return
        # Validate using the strict detector
        img = cv2.imread(file_path)
        if img is None:
            messagebox.showerror("Invalid", "Could not read image.")
            return
        # center-crop to ROI size
        h,w = img.shape[:2]
        cx, cy = w//2, h//2
        half = ROI_SIZE//2
        roi = img[max(0, cy-half):min(h, cy+half), max(0, cx-half):min(w, cx+half)].copy()
        if not is_fingerprint(roi):
            messagebox.showerror("Invalid Image", "This does not appear to be a fingerprint. Upload a clean fingerprint only.")
            return
        # Save standardized versions
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_name = f"fp_upload_{ts}.jpg"
        save_path = os.path.join(REGISTER_DIR, save_name)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (300,300))
        cv2.imwrite(save_path, gray)
        # enhanced & roi paths
        metrics = compute_metrics(gray)
        enhanced = metrics["enhanced"] if metrics and "enhanced" in metrics else gray
        enhanced_path = os.path.join(ENHANCED_DIR, f"enh_upload_{ts}.jpg")
        roi_path = os.path.join(ROI_DIR, f"roi_upload_{ts}.jpg")
        cv2.imwrite(enhanced_path, enhanced)
        cv2.imwrite(roi_path, gray)
        self.last_register_fp = save_path
        try:
            img_pil = Image.open(save_path).resize((150,150))
            tk_img = ImageTk.PhotoImage(img_pil)
            self.fp_preview_label.configure(image=tk_img, text="")
            self.fp_preview_label.image = tk_img
        except Exception:
            self.fp_preview_label.configure(text=f"Fingerprint ready: {save_path}")
        messagebox.showinfo("Upload", f"Fingerprint accepted and saved:\n{save_path}")

    def _capture_register_button(self):
        saved = capture_fingerprint_preview(REGISTER_DIR)
        if saved:
            self.last_register_fp = saved
            try:
                img = Image.open(saved).resize((150,150))
                tk_img = ImageTk.PhotoImage(img)
                self.fp_preview_label.configure(image=tk_img, text="")
                self.fp_preview_label.image = tk_img
            except Exception:
                self.fp_preview_label.configure(text=f"Fingerprint saved: {saved}")
            messagebox.showinfo("Captured", f"Fingerprint saved: {saved}")

    def _register_button(self):
        details = {}
        for k, v in self.entries.items():
            if isinstance(v, ttk.Combobox):
                details[k] = v.get().strip()
            elif isinstance(v, tk.Text):
                details[k] = v.get("1.0", "end-1c").strip()
            else:
                try:
                    details[k] = v.get().strip()
                except Exception:
                    details[k] = ""

        required = ["name", "dob", "phone", "blood_group", "doctor_name", "doctor_phone",
                    "emergency_name", "emergency_relation", "emergency_phone", "aadhar", "age"]
        for f in required:
            if not details.get(f):
                return messagebox.showwarning("Missing", f"Field '{f}' is required!")

        if not details["phone"].isdigit() or len(details["phone"]) != 10:
            return messagebox.showwarning("Invalid", "Phone must be 10 digits.")
        if not details["doctor_phone"].isdigit() or len(details["doctor_phone"]) != 10:
            return messagebox.showwarning("Invalid", "Doctor phone must be 10 digits.")
        if not details["emergency_phone"].isdigit() or len(details["emergency_phone"]) != 10:
            return messagebox.showwarning("Invalid", "Emergency phone must be 10 digits.")
        if not details["aadhar"].isdigit() or len(details["aadhar"]) != 12:
            return messagebox.showwarning("Invalid", "Aadhar must be 12 digits.")
        if not details["age"].isdigit() or int(details["age"]) <= 0:
            return messagebox.showwarning("Invalid", "Age must be a positive integer.")
        try:
            datetime.datetime.strptime(details["dob"], "%Y-%m-%d")
        except Exception:
            return messagebox.showwarning("Invalid", "DOB must be YYYY-MM-DD")

        if not self.last_register_fp or not os.path.exists(self.last_register_fp):
            return messagebox.showwarning("Missing", "Capture or upload fingerprint first.")

        # prepare enhanced & roi paths
        metrics = compute_metrics(cv2.imread(self.last_register_fp, cv2.IMREAD_GRAYSCALE))
        enhanced = metrics["enhanced"] if metrics and "enhanced" in metrics else cv2.imread(self.last_register_fp, cv2.IMREAD_GRAYSCALE)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        enhanced_path = os.path.join(ENHANCED_DIR, f"enh_reg_{ts}.jpg")
        roi_path = os.path.join(ROI_DIR, f"roi_reg_{ts}.jpg")
        try:
            cv2.imwrite(enhanced_path, enhanced)
            cv2.imwrite(roi_path, cv2.imread(self.last_register_fp, cv2.IMREAD_GRAYSCALE))
        except Exception:
            pass

        try:
            uid = insert_user_normalized(details, self.last_register_fp, enhanced_path, roi_path, debug_preview=False)
            messagebox.showinfo("Success", f"User registered with ID: {uid}")
            self.fp_preview_label.configure(image="", text="No fingerprint captured")
            self._clear_register_form()
            self.last_register_fp = None
        except Error as e:
            messagebox.showerror("DB Error", str(e))
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Error", str(e))

    def _clear_register_form(self):
        for k, widget in self.entries.items():
            if isinstance(widget, tk.Text):
                widget.delete("1.0", tk.END)
            elif isinstance(widget, ttk.Combobox):
                widget.set(BLOOD_GROUPS[0])
            else:
                try:
                    widget.set("")
                except Exception:
                    try:
                        widget.delete(0, tk.END)
                    except Exception:
                        pass

    # Verify tab
    def _build_verify_tab(self):
        left = ttk.Frame(self.ver_frame, padding=12)
        left.pack(side="left", fill="y")
        self.verify_preview_label = ttk.Label(left, text="No capture yet", relief="sunken", width=40)
        self.verify_preview_label.pack(pady=6)
        ttk.Button(left, text="📸 Capture For Verification", command=self._capture_verify_button).pack(pady=6)
        ttk.Button(left, text="📁 Upload For Verification", command=self._upload_verify_fp).pack(pady=6)
        ttk.Button(left, text="🔍 Run Match", command=self._run_match_button).pack(pady=6)

        right = ttk.Frame(self.ver_frame, padding=12)
        right.pack(side="left", fill="both", expand=True)
        self.result_text = tk.Text(right, width=70, height=22, relief="solid", bd=1)
        self.result_text.pack(pady=6, fill="both", expand=True)

    def _upload_verify_fp(self):
        file_path = filedialog.askopenfilename(title="Select fingerprint image", filetypes=[("Images","*.png;*.jpg;*.jpeg")])
        if not file_path:
            return
        img = cv2.imread(file_path)
        if img is None:
            messagebox.showerror("Invalid", "Could not read image.")
            return
        h,w = img.shape[:2]
        cx, cy = w//2, h//2
        half = ROI_SIZE//2
        roi = img[max(0, cy-half):min(h, cy+half), max(0, cx-half):min(w, cx+half)].copy()
        if not is_fingerprint(roi):
            messagebox.showerror("Invalid Image", "This does not appear to be a fingerprint. Upload a clean fingerprint only.")
            return
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_name = f"fp_verify_upload_{ts}.jpg"
        save_path = os.path.join(VERIFY_DIR, save_name)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (300,300))
        cv2.imwrite(save_path, gray)
        self.last_verify_fp = save_path
        try:
            img_pil = Image.open(save_path).resize((150,150))
            tk_img = ImageTk.PhotoImage(img_pil)
            self.verify_preview_label.configure(image=tk_img, text="")
            self.verify_preview_label.image = tk_img
        except Exception:
            self.verify_preview_label.configure(text=f"Captured: {save_path}")
        messagebox.showinfo("Upload", f"Verification fingerprint accepted and saved:\n{save_path}")

    def _capture_verify_button(self):
        saved = capture_fingerprint_preview(VERIFY_DIR)
        if saved:
            self.last_verify_fp = saved
            try:
                img = Image.open(saved).resize((150,150))
                tk_img = ImageTk.PhotoImage(img)
                self.verify_preview_label.configure(image=tk_img, text="")
                self.verify_preview_label.image = tk_img
            except Exception:
                self.verify_preview_label.configure(text=f"Captured: {saved}")
            messagebox.showinfo("Captured", f"Verification fingerprint saved: {saved}")

    def _run_match_button(self):
        if not self.last_verify_fp:
            return messagebox.showerror("Error", "Please capture/upload fingerprint first.")
        best_user, good, ratio = find_best_match(self.last_verify_fp)
        self.result_text.delete("1.0", tk.END)
        if best_user and good >= MIN_GOOD_MATCHES and ratio >= MIN_MATCH_RATIO:
            user_id = best_user[0]
            log_verification(user_id)
            count = get_monthly_verification_count(user_id)
            fields = ["ID", "Name", "Age", "DOB", "Blood Group", "Phone", "Address", "Aadhar",
                      "Doctor", "Doctor Phone", "Emergency Name", "Emergency Relation", "Emergency Phone", "Fingerprint Path"]
            result_lines = [f"{fields[i]}: {best_user[i]}" for i in range(len(fields))]
            result_lines.append(f"\nMatch Score: {good} good matches, {ratio*100:.2f}% ratio ✅")
            result_lines.append(f"\nMonthly Verifications: {count}")
            if count > 5:
                result_lines.append("\n⚠️ User shows abnormal verification activity. Possible medical attention advised.")
            self.result_text.insert("1.0", "\n".join(result_lines))
        else:
            self.result_text.insert("1.0", f"❌ No matching user found or poor fingerprint quality.\nBest found: {good} good matches, {ratio*100:.2f}% ratio\n")

    # Tracker tab
    def _build_tracker_tab(self):
        top = ttk.Frame(self.tracker_frame)
        top.pack(fill="x", pady=8)
        ttk.Button(top, text="Refresh Tracker", command=self._load_verification_log).pack(side="right", padx=10)
        self.tracker_tree = ttk.Treeview(self.tracker_frame, columns=("Name", "Count", "Last Verified"), show="headings")
        self.tracker_tree.heading("Name", text="Name")
        self.tracker_tree.heading("Count", text="Verification Count")
        self.tracker_tree.heading("Last Verified", text="Last Verified Time")
        self.tracker_tree.pack(expand=True, fill="both", padx=20, pady=12)

    def _load_verification_log(self):
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT u.name, COUNT(v.id) AS count, MAX(v.verified_at)
            FROM verification_logs v
            JOIN users_new u ON v.user_id = u.id
            GROUP BY v.user_id
            ORDER BY count DESC
        """)
        rows = cur.fetchall()
        cur.close()
        conn.close()
        self.tracker_tree.delete(*self.tracker_tree.get_children())
        for r in rows:
            self.tracker_tree.insert("", "end", values=r)

# ---------------- Run ----------------
if __name__ == "__main__":
    ensure_db_schema()
    root = tk.Tk()
    splash = SplashScreen(root, delay=800)
    app = LifeLineApp(root)
    root.mainloop()
