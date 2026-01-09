"""
lifeline_final_loose_fixed.py
Fixed LifeLine ID - Loose mode with stricter duplicate prevention and improved live-fingerprint-only checks.
Usage: pip install opencv-python pillow mysql-connector-python numpy tkcalendar
Then: python lifeline_final_loose_fixed.py

This is a copy of your provided loose-mode app with these changes:
- Lowered duplicate-match threshold and added transactional duplicate check
- Ensured unique DB indexes on fingerprint_hash and fingerprint_path
- Improved detection messaging and slightly tightened heuristics to avoid non-fingerprint images
"""
import os
import cv2
import datetime
import hashlib
import mysql.connector
from mysql.connector import Error
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import numpy as np
import traceback

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

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
FINGER_DIR = os.path.join(ROOT_DIR, "fingerprints")
REGISTER_DIR = os.path.join(FINGER_DIR, "registered")
VERIFY_DIR = os.path.join(FINGER_DIR, "verified")
os.makedirs(REGISTER_DIR, exist_ok=True)
os.makedirs(VERIFY_DIR, exist_ok=True)

# Matching thresholds
MIN_GOOD_MATCHES = 10
MIN_MATCH_RATIO = 0.08
# Lowered duplicate threshold to be more conservative about duplicates
DUPLICATE_MATCH_RATIO = 0.12
ROI_SIZE = 300

BLOOD_GROUPS = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]

# If True, accept stylized fingerprint images like the one you uploaded (loose mode).
ALLOW_VECTOR_FINGERPRINTS = True

# ---------------- DB HELPERS ----------------
def get_db_connection():
    try:
        return mysql.connector.connect(**DB_CONFIG)
    except mysql.connector.errors.ProgrammingError as e:
        if "Unknown database" in str(e):
            tmp = mysql.connector.connect(host=DB_CONFIG["host"], user=DB_CONFIG["user"], password=DB_CONFIG["password"])
            cur = tmp.cursor()
            cur.execute(f"CREATE DATABASE IF NOT EXISTS {DB_CONFIG['database']}")
            tmp.commit()
            cur.close(); tmp.close()
            return mysql.connector.connect(**DB_CONFIG)
        raise

def ensure_db_schema():
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users_new (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(200), age INT, dob DATE,
                blood_group_id INT, phone VARCHAR(20), address TEXT,
                aadhar VARCHAR(20), doctor_id INT, emergency_contact_id INT,
                fingerprint_path VARCHAR(500), fingerprint_hash VARCHAR(128),
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
                name VARCHAR(200), phone VARCHAR(20) UNIQUE
            ) ENGINE=InnoDB
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS emergency_contacts (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(200), relation VARCHAR(50), phone VARCHAR(20) UNIQUE
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

        # Create unique indexes if they don't already exist to enforce uniqueness at DB level
        cur.execute("SELECT COUNT(*) FROM information_schema.statistics WHERE table_schema=%s AND table_name='users_new' AND index_name='uq_fingerprint_hash'", (DB_CONFIG['database'],))
        if cur.fetchone()[0] == 0:
            try:
                cur.execute("ALTER TABLE users_new ADD UNIQUE INDEX uq_fingerprint_hash (fingerprint_hash)")
            except Exception:
                pass
        cur.execute("SELECT COUNT(*) FROM information_schema.statistics WHERE table_schema=%s AND table_name='users_new' AND index_name='uq_fingerprint_path'", (DB_CONFIG['database'],))
        if cur.fetchone()[0] == 0:
            try:
                cur.execute("ALTER TABLE users_new ADD UNIQUE INDEX uq_fingerprint_path (fingerprint_path(255))")
            except Exception:
                pass
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

# ---------------- Fingerprint metric & detection ----------------
def auto_contrast(img_gray):
    if img_gray is None or img_gray.size == 0:
        return img_gray
    p_low, p_high = np.percentile(img_gray, (2, 98))
    if p_high - p_low <= 0:
        return img_gray
    stretched = np.clip((img_gray - p_low) * 255.0 / (p_high - p_low), 0, 255).astype(np.uint8)
    return stretched

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
        contrast_std = float(np.std(gray))
        
        # Check for skin-like color (helps reject non-fingerprint objects)
        # But mobile fingerprints may have different lighting, so make this more permissive
        skin_score = 0.0
        if len(roi_bgr.shape) == 3:
            b, g, r = cv2.split(roi_bgr)
            # Skin pixels: R > G > B, but allow wider range for mobile photos
            skin_mask = (r > g) & (g >= b) & (r > 70)
            skin_score = np.sum(skin_mask) / skin_mask.size if skin_mask.size > 0 else 0.0
        
        return {
            "edge_ratio": edge_ratio,
            "texture_var": texture_var,
            "kp_count": kp_count,
            "blur_metric": blur_metric,
            "contrast_std": contrast_std,
            "canny_low": lower,
            "canny_high": upper,
            "skin_score": skin_score
        }
    except Exception:
        return None

def is_fingerprint_loose(roi_bgr, debug=False):
    """
    Relaxed fingerprint detector to accept:
    - Real fingerprints from webcam
    - Fingerprints captured from mobile phone
    - Stylized vector fingerprints
    While still rejecting:
    - Cat/animal images (fur has random orientation)
    - Flowers/plants (organic random pattern)
    - Plain backgrounds
    """
    m = compute_metrics(roi_bgr)
    if not m:
        if debug: print("[is_fingerprint_loose] metrics missing")
        return False

    # Check: ridge orientation consistency
    # Fingerprints have organized structure; random textures (fur, leaves) do not
    try:
        if len(roi_bgr.shape) == 3:
            gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi_bgr.copy()
        gray = cv2.resize(gray, (300, 300))
        gray = auto_contrast(gray)
        
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        angle = np.arctan2(sobely, sobelx)
        
        hist, _ = np.histogram(angle, bins=8, range=(-np.pi, np.pi))
        hist_normalized = hist / np.sum(hist) if np.sum(hist) > 0 else hist
        angle_entropy = -np.sum(hist_normalized * np.log(hist_normalized + 1e-10))
        
        if debug:
            print(f"[is_fingerprint_loose] angle_entropy={angle_entropy:.3f}, skin_score={m['skin_score']:.3f}")
    except Exception as e:
        if debug: print(f"[is_fingerprint_loose] angle check failed: {e}")
        angle_entropy = 1.5

    # Photographic fingerprint (real finger, webcam or mobile)
    # Relaxed: accept wider range of edge ratios, lower keypoint requirement for mobile
    cond_photo = (
        0.05 < m["edge_ratio"] < 0.50 and 
        m["texture_var"] > 8 and 
        m["kp_count"] > 20 and 
        m["blur_metric"] > 8 and
        angle_entropy < 1.5  # organized directional structure (rejects random fur/leaves)
    )

    # Stylized/vector fingerprint (high contrast, organized lines)
    cond_vector = (
        m["contrast_std"] > 30 and 
        m["edge_ratio"] > 0.04 and 
        m["edge_ratio"] < 0.55 and
        m["kp_count"] > 10 and
        angle_entropy < 1.5
    )

    if debug:
        print(f"[is_fingerprint_loose] edge_ratio={m['edge_ratio']:.3f}, kp={m['kp_count']}, blur={m['blur_metric']:.1f}, contrast={m['contrast_std']:.1f}")
        print(f"[is_fingerprint_loose] cond_photo={cond_photo}, cond_vector={cond_vector}, angle_entropy={angle_entropy:.3f}")

    if ALLOW_VECTOR_FINGERPRINTS:
        return cond_photo or cond_vector
    return cond_photo

# ---------------- ORB matching & duplicate ----------------
def orb_match_score(img1_gray, img2_gray):
    try:
        orb = cv2.ORB_create(nfeatures=1200)
        kp1, des1 = orb.detectAndCompute(img1_gray, None)
        kp2, des2 = orb.detectAndCompute(img2_gray, None)
        if des1 is None or des2 is None or len(kp1) < 5 or len(kp2) < 5:
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

def is_duplicate_fingerprint(new_fp_path):
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
            cur.close(); conn.close(); return row[1], 1.0
        cur.execute("SELECT id, name, fingerprint_path FROM users_new WHERE fingerprint_path IS NOT NULL")
        rows = cur.fetchall(); cur.close(); conn.close()
        new_img = cv2.imread(new_fp_path, cv2.IMREAD_GRAYSCALE)
        if new_img is None:
            return None, 0.0
        new_img = cv2.resize(new_img, (300, 300))
        best_name = None; best_ratio = 0.0
        for uid, uname, fp_path in rows:
            try:
                if not fp_path or not os.path.exists(fp_path):
                    continue
                existing_img = cv2.imread(fp_path, cv2.IMREAD_GRAYSCALE)
                if existing_img is None:
                    continue
                existing_img = cv2.resize(existing_img, (300, 300))
                _, ratio = orb_match_score(new_img, existing_img)
                if ratio > best_ratio:
                    best_ratio = ratio; best_name = uname
                if ratio >= DUPLICATE_MATCH_RATIO:
                    return uname, ratio
            except Exception:
                continue
        return (best_name, best_ratio) if best_ratio >= DUPLICATE_MATCH_RATIO else (None, 0.0)
    except Exception:
        if conn: conn.close()
        return None, 0.0

# ---------------- Webcam capture (loose acceptance) ----------------
def capture_fingerprint_preview(save_dir,
                                window_title="Place finger (or fingerprint image) in box, press 's' to capture, 'q' to cancel",
                                auto_save=True):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Webcam Error", "Cannot open webcam. Close other apps using camera.")
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
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 200, 120), -1)
            cv2.addWeighted(overlay, 0.18, frame_display, 0.82, 0, frame_display)
            cv2.rectangle(frame_display, (x1, y1), (x2, y2), (0, 200, 120), 2)
            cv2.putText(frame_display, "Align fingerprint inside the box. Press 's' to capture, 'q' to cancel", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            roi = frame[y1:y2, x1:x2].copy()
            M = compute_metrics(roi)
            if M:
                info_y = 70
                cv2.putText(frame_display, f"edge_ratio:{M['edge_ratio']:.4f}", (10,info_y), 0, 0.5, (220,220,220), 1); info_y += 18
                cv2.putText(frame_display, f"texture:{M['texture_var']:.1f}", (10,info_y), 0, 0.5, (220,220,220), 1); info_y += 18
                cv2.putText(frame_display, f"kp:{M['kp_count']}", (10,info_y), 0, 0.5, (220,220,220), 1); info_y += 18
                cv2.putText(frame_display, f"blur:{M['blur_metric']:.1f}", (10,info_y), 0, 0.5, (220,220,220), 1)

            cv2.imshow(window_title, frame_display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                saved_path = None
                break
            if key == ord('s'):
                if roi is None or roi.size == 0:
                    continue
                # check whether ROI is fingerprint-like (loose)
                if not is_fingerprint_loose(roi):
                    cv2.putText(frame_display, "NOT A FINGERPRINT — TRY AGAIN", (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                    cv2.imshow(window_title, frame_display)
                    cv2.waitKey(900)
                    continue
                # convert to grayscale and save normalized
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (300, 300))
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"fp_{ts}.jpg"
                saved_path = os.path.join(save_dir, filename)
                cv2.imwrite(saved_path, gray)
                if auto_save:
                    messagebox.showinfo("Captured", f"Fingerprint saved: {saved_path}")
                break
    except Exception as e:
        print("Capture error:", e)
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return saved_path

# ---------------- Verification and DB utilities ----------------
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
        good_count, ratio = orb_match_score(captured, stored)
        if good_count > best_score or (good_count == best_score and ratio > best_ratio):
            best_score = good_count
            best_ratio = ratio
            best_user = row
    return best_user, best_score, best_ratio

def log_verification(user_id):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("INSERT INTO verification_logs (user_id) VALUES (%s)", (user_id,))
    conn.commit(); cur.close(); conn.close()

def get_monthly_verification_count(user_id):
    conn = get_db_connection(); cur = conn.cursor()
    cur.execute("""
        SELECT COUNT(*) FROM verification_logs
        WHERE user_id = %s AND MONTH(verified_at) = MONTH(CURRENT_DATE())
        AND YEAR(verified_at) = YEAR(CURRENT_DATE())
    """, (user_id,))
    count = cur.fetchone()[0]
    cur.close(); conn.close()
    return count

# ---------------- Insert user with duplicate protections (transactional) ----------------
def insert_user_normalized(details: dict, fp_path: str):
    if not fp_path or not os.path.exists(fp_path):
        raise Error("Fingerprint image not found for insertion.")
    fp_hash = get_fingerprint_hash(fp_path)
    if not fp_hash:
        raise Error("Could not compute fingerprint hash.")

    conn = get_db_connection()
    try:
        # start transaction to avoid race where two inserts slip through at same time
        conn.start_transaction()
        cur = conn.cursor()

        # DB-level hash duplicate - recheck inside transaction
        cur.execute("SELECT id, name FROM users_new WHERE fingerprint_hash=%s FOR SHARE", (fp_hash,))
        existing = cur.fetchone()
        if existing:
            cur.close(); conn.rollback(); conn.close()
            raise Error(f"Duplicate fingerprint hash detected! Already registered: {existing[1]}")

        # Pattern duplicate check
        dup_name, dup_ratio = is_duplicate_fingerprint(fp_path)
        if dup_name:
            cur.close(); conn.rollback(); conn.close()
            raise Error(f"Duplicate fingerprint (pattern) detected: {dup_name} ({dup_ratio*100:.2f}%)")

        bg_id = get_or_create_blood_group(cur, conn, details.get("blood_group") or "Unknown")
        doctor_id = get_or_create_doctor(cur, conn, details.get("doctor_name"), details.get("doctor_phone"))
        ec_id = get_or_create_emergency(cur, conn, details.get("emergency_name"), details.get("emergency_relation"), details.get("emergency_phone"))

        sql = ("""
            INSERT INTO users_new
            (name, age, dob, blood_group_id, phone, address, aadhar,
             doctor_id, emergency_contact_id, fingerprint_path, fingerprint_hash, created_at)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,NOW())
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
            fp_hash
        )
        cur.execute(sql, params)
        conn.commit()
        uid = cur.lastrowid
        cur.close(); conn.close()
        return uid
    except mysql.connector.IntegrityError as ie:
        # unique constraint triggered
        try:
            conn.rollback()
            conn.close()
        except Exception:
            pass
        raise Error("Duplicate fingerprint detected (database constraint).")
    except Exception as e:
        try:
            conn.rollback(); conn.close()
        except Exception:
            pass
        raise

# ---------------- UI / remaining helpers (kept mostly unchanged) ----------------
def open_calendar(dob_var, age_var):
    win = tk.Toplevel()
    win.title("Select DOB")
    win.geometry("320x360")
    if not TKCAL_AVAILABLE:
        msg = "tkcalendar not installed.\nInstall with:\n\npip install tkcalendar"
        tk.Label(win, text=msg, fg="red", padx=10, pady=20).pack()
        return
    cal = Calendar(win, selectmode="day", date_pattern="yyyy-mm-dd")
    cal.pack(pady=10, expand=True, fill="both")
    def select_date():
        date_str = cal.get_date()
        dob_var.set(date_str)
        dob = datetime.datetime.strptime(date_str, "%Y-%m-%d")
        today = datetime.datetime.today()
        age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
        age_var.set(str(age))
        win.destroy()
    ttk.Button(win, text="Select", command=select_date).pack(pady=10)

class LifeLineApp:
    def __init__(self, root):
        self.root = root
        root.title("LifeLine ID - Fixed Loose Mode")
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
        style.configure("TNotebook.Tab", font=("Segoe UI", 11, "bold"), padding=[18,8])
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True, padx=15, pady=15)
        self.reg_frame = tk.Frame(self.notebook, bg="white")
        self.ver_frame = tk.Frame(self.notebook, bg="white")
        self.tracker_frame = tk.Frame(self.notebook, bg="white")
        self.notebook.add(self.reg_frame, text="Register")
        self.notebook.add(self.ver_frame, text="Verify")
        self.notebook.add(self.tracker_frame, text="Tracker")
        self._build_register_tab(); self._build_verify_tab(); self._build_tracker_tab()
        self.last_register_fp = None; self.last_verify_fp = None

    def _build_register_tab(self):
        container = ttk.Frame(self.reg_frame); container.pack(fill="both", expand=True, padx=12, pady=12)
        form_canvas = tk.Canvas(container, bg="white", highlightthickness=0, width=640)
        vscroll = ttk.Scrollbar(container, orient="vertical", command=form_canvas.yview)
        form_frame = ttk.Frame(form_canvas)
        form_frame.bind("<Configure>", lambda e: form_canvas.configure(scrollregion=form_canvas.bbox("all")))
        form_canvas.create_window((0,0), window=form_frame, anchor="nw")
        form_canvas.configure(yscrollcommand=vscroll.set)
        form_canvas.pack(side="left", fill="both", expand=True); vscroll.pack(side="left", fill="y")
        labels = [("Name","name"),("Age (auto)","age"),("DOB","dob"),("Blood Group","blood_group"),("Phone","phone"),("Address","address"),("Aadhar","aadhar"),("Doctor Name","doctor_name"),("Doctor Phone","doctor_phone"),("Emergency Name","emergency_name"),("Emergency Relation","emergency_relation"),("Emergency Phone","emergency_phone")]
        self.entries = {}
        for i,(lab,key) in enumerate(labels):
            ttk.Label(form_frame, text=lab).grid(row=i, column=0, sticky="w", pady=6, padx=6)
            if key == "address":
                txt = tk.Text(form_frame, width=36, height=4, relief="solid", bd=1); txt.grid(row=i, column=1, pady=6, padx=6, sticky="w"); self.entries[key]=txt
            elif key == "blood_group":
                cb = ttk.Combobox(form_frame, values=BLOOD_GROUPS, state="readonly", width=36); cb.set(BLOOD_GROUPS[0]); cb.grid(row=i, column=1, pady=6, padx=6); self.entries[key]=cb
            elif key == "dob":
                dob_var = tk.StringVar(); dob_entry = ttk.Entry(form_frame, textvariable=dob_var, width=30); dob_entry.grid(row=i, column=1, pady=6, padx=6, sticky="w"); btn = ttk.Button(form_frame, text="📅", width=4, command=lambda dv=dob_var: open_calendar(dv, self.entries["age"]))
                btn.grid(row=i, column=1, sticky="e", padx=6); self.entries[key]=dob_var
            elif key == "age":
                age_var = tk.StringVar(); age_entry = ttk.Entry(form_frame, textvariable=age_var, width=12); age_entry.grid(row=i, column=1, pady=6, padx=6, sticky="w"); self.entries[key]=age_var
            else:
                sv = tk.StringVar(); ttk.Entry(form_frame, textvariable=sv, width=40).grid(row=i, column=1, pady=6, padx=6, sticky="w"); self.entries[key]=sv
        ttk.Button(form_frame, text="📸 Capture Fingerprint", command=self._capture_register_button).grid(row=len(labels), column=0, pady=12)
        ttk.Button(form_frame, text="✅ Register", command=self._register_button).grid(row=len(labels), column=1, pady=12)
        self.fp_preview_label = ttk.Label(form_frame, text="No fingerprint captured", relief="sunken", width=50); self.fp_preview_label.grid(row=len(labels)+1, column=0, columnspan=2, pady=10)

    def _capture_register_button(self):
        saved = capture_fingerprint_preview(REGISTER_DIR)
        if saved:
            self.last_register_fp = saved
            try:
                img = Image.open(saved); img = img.resize((150,150)); tk_img = ImageTk.PhotoImage(img); self.fp_preview_label.configure(image=tk_img, text=""); self.fp_preview_label.image = tk_img
            except Exception:
                pass
            messagebox.showinfo("Captured", f"Fingerprint saved: {saved}")

    def _register_button(self):
        details = {}
        for k,v in self.entries.items():
            if isinstance(v, tk.StringVar): details[k] = v.get().strip()
            elif isinstance(v, ttk.Combobox): details[k] = v.get().strip()
            elif isinstance(v, tk.Text): details[k] = v.get("1.0", "end-1c").strip()
            else:
                try: details[k] = v.get().strip()
                except Exception: details[k] = ""
        required = ["name","dob","phone","blood_group","doctor_name","doctor_phone","emergency_name","emergency_relation","emergency_phone","aadhar","age"]
        for f in required:
            if not details.get(f): return messagebox.showwarning("Missing", f"Field '{f}' is required!")
        if not details["phone"].isdigit() or len(details["phone"]) != 10: return messagebox.showwarning("Invalid", "Phone must be 10 digits.")
        if not details["doctor_phone"].isdigit() or len(details["doctor_phone"]) != 10: return messagebox.showwarning("Invalid", "Doctor phone must be 10 digits.")
        if not details["emergency_phone"].isdigit() or len(details["emergency_phone"]) != 10: return messagebox.showwarning("Invalid", "Emergency phone must be 10 digits.")
        if not details["aadhar"].isdigit() or len(details["aadhar"]) != 12: return messagebox.showwarning("Invalid", "Aadhar must be 12 digits.")
        if not details["age"].isdigit() or int(details["age"]) <= 0: return messagebox.showwarning("Invalid", "Age must be a positive integer.")
        try: datetime.datetime.strptime(details["dob"], "%Y-%m-%d")
        except Exception: return messagebox.showwarning("Invalid", "DOB must be YYYY-MM-DD")
        if not self.last_register_fp: return messagebox.showwarning("Missing", "Capture fingerprint first.")
        try:
            uid = insert_user_normalized(details, self.last_register_fp)
            messagebox.showinfo("Success", f"User registered with ID: {uid}")
            self.fp_preview_label.configure(image="", text="No fingerprint captured")
            for k,v in self.entries.items():
                if isinstance(v, tk.StringVar): v.set("")
                elif isinstance(v, ttk.Combobox): v.set(BLOOD_GROUPS[0])
                elif isinstance(v, tk.Text): v.delete("1.0", tk.END)
            self.last_register_fp = None
        except Error as e:
            messagebox.showerror("DB Error", str(e))
        except Exception as e:
            traceback.print_exc(); messagebox.showerror("Error", str(e))

    def _build_verify_tab(self):
        left = ttk.Frame(self.ver_frame, padding=12); left.pack(side="left", fill="y")
        self.verify_preview_label = ttk.Label(left, text="No capture yet", relief="sunken", width=40); self.verify_preview_label.pack(pady=6)
        ttk.Button(left, text="📸 Capture For Verification", command=self._capture_verify_button).pack(pady=6)
        ttk.Button(left, text="🔍 Run Match", command=self._run_match_button).pack(pady=6)
        right = ttk.Frame(self.ver_frame, padding=12); right.pack(side="left", fill="both", expand=True)
        self.result_text = tk.Text(right, width=70, height=22, relief="solid", bd=1); self.result_text.pack(pady=6, fill="both", expand=True)

    def _capture_verify_button(self):
        saved = capture_fingerprint_preview(VERIFY_DIR)
        if saved:
            self.last_verify_fp = saved
            try:
                img = Image.open(saved); img = img.resize((150,150)); tk_img = ImageTk.PhotoImage(img); self.verify_preview_label.configure(image=tk_img, text=""); self.verify_preview_label.image = tk_img
            except Exception: pass
            messagebox.showinfo("Captured", f"Verification fingerprint saved: {saved}")

    def _run_match_button(self):
        if not self.last_verify_fp: return messagebox.showerror("Error", "Capture fingerprint first.")
        best_user, good, ratio = find_best_match(self.last_verify_fp)
        self.result_text.delete("1.0", tk.END)
        if best_user and good >= MIN_GOOD_MATCHES and ratio >= MIN_MATCH_RATIO:
            user_id = best_user[0]; log_verification(user_id); count = get_monthly_verification_count(user_id)
            fields = ["ID","Name","Age","DOB","Blood Group","Phone","Address","Aadhar","Doctor","Doctor Phone","Emergency Name","Emergency Relation","Emergency Phone","Fingerprint Path"]
            result_lines = [f"{fields[i]}: {best_user[i]}" for i in range(len(fields))]
            result_lines.append(f"\nMatch Score: {good} good matches, {ratio*100:.2f}% ratio ✅")
            result_lines.append(f"\nMonthly Verifications: {count}")
            if count > 5: result_lines.append("\n⚠️ User shows abnormal verification activity.")
            self.result_text.insert("1.0", "\n".join(result_lines))
        else:
            msg = "❌ No matching registered user found for this fingerprint.\n"
            msg += f"Best candidate had {good} good matches with ratio {ratio*100:.2f}%.\n"
            msg += "If you are testing stylized fingerprints, ensure the stored fingerprint was registered from similar image types.\n"
            self.result_text.insert("1.0", msg)

    def _build_tracker_tab(self):
        top = ttk.Frame(self.tracker_frame); top.pack(fill="x", pady=8)
        ttk.Button(top, text="Refresh Tracker", command=self._load_verification_log).pack(side="right", padx=10)
        self.tracker_tree = ttk.Treeview(self.tracker_frame, columns=("Name","Count","Last Verified"), show="headings")
        self.tracker_tree.heading("Name", text="Name"); self.tracker_tree.heading("Count", text="Verification Count"); self.tracker_tree.heading("Last Verified", text="Last Verified Time")
        self.tracker_tree.pack(expand=True, fill="both", padx=20, pady=12)

    def _load_verification_log(self):
        conn = get_db_connection(); cur = conn.cursor()
        cur.execute("""
            SELECT u.name, COUNT(v.id) AS count, MAX(v.verified_at)
            FROM verification_logs v
            JOIN users_new u ON v.user_id = u.id
            GROUP BY v.user_id
            ORDER BY count DESC
        """)
        rows = cur.fetchall(); cur.close(); conn.close()
        self.tracker_tree.delete(*self.tracker_tree.get_children())
        for r in rows: self.tracker_tree.insert("", "end", values=r)

if __name__ == "__main__":
    ensure_db_schema()
    root = tk.Tk(); app = LifeLineApp(root); root.mainloop()
