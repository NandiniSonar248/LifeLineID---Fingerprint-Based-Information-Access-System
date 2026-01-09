# ---------------- Part 1 of 3 ----------------
# app18.py (part 1) - Final working version (imports, config, DB helpers, fingerprint metrics)

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

# Try import tkcalendar; show helpful message later if missing
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
os.makedirs(REGISTER_DIR, exist_ok=True)
os.makedirs(VERIFY_DIR, exist_ok=True)

# Stronger thresholds to avoid false matches (tuned for webcam/mobile-screen captures)
MIN_GOOD_MATCHES = 25       # minimum number of good ORB matches
MIN_MATCH_RATIO = 0.20      # minimum match ratio (good_matches / min(keypoints)) e.g. 20%
DUPLICATE_MATCH_RATIO = 0.22
ROI_SIZE = 300

BLOOD_GROUPS = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]

# ---------------- DB HELPERS ----------------

def get_db_connection():
    # Connects to DB; attempts to create database if missing.
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn
    except mysql.connector.errors.ProgrammingError as e:
        # Try to create database if missing
        if "Unknown database" in str(e):
            tmp = mysql.connector.connect(host=DB_CONFIG["host"],
                                          user=DB_CONFIG["user"],
                                          password=DB_CONFIG["password"])
            cur = tmp.cursor()
            cur.execute(f"CREATE DATABASE IF NOT EXISTS {DB_CONFIG['database']}")
            tmp.commit()
            cur.close()
            tmp.close()
            return mysql.connector.connect(**DB_CONFIG)
        else:
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
    cur.execute("INSERT INTO emergency_contacts (name, relation, phone) VALUES (%s, %s, %s)",
                (name, relation, phone))
    conn.commit()
    return cur.lastrowid

def get_fingerprint_hash(img_path):
    try:
        with open(img_path, "rb") as f:
            data = f.read()
        return hashlib.sha256(data).hexdigest()
    except Exception:
        return None

# ---------------- Fingerprint metric & strict check ----------------

def auto_contrast(img_gray):
    """Auto contrast stretch (returns uint8 image)."""
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
        # convert to gray if needed
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
        lower = int(max(10, 0.66 * median_val))
        upper = int(min(200, 1.33 * median_val))
        edges = cv2.Canny(blurred, lower, upper)
        edge_ratio = np.sum(edges > 0) / edges.size
        texture_var = float(np.var(enhanced))
        orb = cv2.ORB_create(nfeatures=1500)
        kps = orb.detect(enhanced, None)
        kp_count = len(kps) if kps is not None else 0
        blur_metric = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        return {
            "edge_ratio": edge_ratio,
            "texture_var": texture_var,
            "kp_count": kp_count,
            "blur_metric": blur_metric,
            "canny_low": lower,
            "canny_high": upper
        }
    except Exception:
        return None

def is_fingerprint_strict(roi_bgr):
    """
    Return True only if ROI looks like a fingerprint (ridges/texture/edges/kps not too low).
    This function is used to strictly allow only fingerprint-like images (webcam or mobile-screen).
    """
    m = compute_metrics(roi_bgr)
    if not m:
        return False

    edge = m["edge_ratio"]
    tex  = m["texture_var"]
    kp   = m["kp_count"]
    blur = m["blur_metric"]

    # Tuned thresholds for webcam/mobile-screen captures
    cond_edge = (edge > 0.02 and edge < 0.55)
    cond_tex  = (tex > 20 and tex < 5000)
    cond_kp   = (kp > 30)
    cond_blur = (blur > 8)  # allow some flexibility for mobile photo blur

    return cond_edge and cond_tex and cond_kp and cond_blur

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
    """
    Check duplicates by hash first, then ORB similarity against stored fingerprints.
    Returns (name, ratio) if a likely duplicate found, else (None, 0.0).
    """
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
        # normalize for matching
        new_img = auto_contrast(new_img)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        new_img = clahe.apply(new_img)
        best_name = None; best_ratio = 0.0
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

# ---------------- End of Part 1 ----------------
# ---------------- Part 2 of 3 ----------------
# app18.py (part 2) - Strict fingerprint capture + verification + DOB calendar

# ---------------- STRICT capture (no bypass) ----------------

def capture_fingerprint_preview(save_dir,
                                window_title="Place finger in box and press 's' to capture, 'q' to cancel",
                                auto_save=True):

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Webcam Error", "Cannot open webcam. Close other apps using the camera.")
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
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 128), -1)
            cv2.addWeighted(overlay, 0.20, frame_display, 0.80, 0, frame_display)
            cv2.rectangle(frame_display, (x1, y1), (x2, y2), (0, 255, 128), 2)

            cv2.putText(frame_display, "Press 's' to capture, 'q' to cancel",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            roi = frame[y1:y2, x1:x2].copy()

            M = compute_metrics(roi)
            if M:
                y0 = 70
                for txt in [
                    f"edge_ratio : {M['edge_ratio']:.4f}",
                    f"texture_var: {M['texture_var']:.1f}",
                    f"kp_count   : {M['kp_count']}",
                    f"blur(var)  : {M['blur_metric']:.1f}"
                ]:
                    cv2.putText(frame_display, txt, (10, y0),
                                0, 0.55, (200, 200, 200), 1)
                    y0 += 20

            cv2.imshow(window_title, frame_display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                saved_path = None
                break

            if key == ord('s'):
                # strict fingerprint detection
                if not is_fingerprint_strict(roi):
                    cv2.putText(frame_display, "NOT A FINGERPRINT — TRY AGAIN",
                                (10, h-10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 0, 255), 2)
                    cv2.imshow(window_title, frame_display)
                    cv2.waitKey(1200)
                    continue

                # save fingerprint
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (300, 300))
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"fp_{ts}.jpg"
                saved_path = os.path.join(save_dir, filename)
                cv2.imwrite(saved_path, gray)

                if auto_save:
                    messagebox.showinfo("Captured", f"Fingerprint saved:\n{saved_path}")
                break

    except Exception as e:
        print("Error during strict capture:", e)
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return saved_path


# ---------------- Verification ----------------

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

        # best ratio wins
        if ratio > best_ratio:
            best_ratio = ratio
            best_score = good_count
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
        WHERE user_id = %s
        AND MONTH(verified_at) = MONTH(CURRENT_DATE())
        AND YEAR(verified_at) = YEAR(CURRENT_DATE())
    """, (user_id,))
    count = cur.fetchone()[0]
    cur.close()
    conn.close()
    return count


# ---------------- Insert user ----------------

def insert_user_normalized(details: dict, fp_path: str):
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
        cur.close()
        conn.close()
        raise Error(f"Duplicate fingerprint detected! Already registered to: {existing[1]}")

    dup_name, dup_ratio = is_duplicate_fingerprint(fp_path)
    if dup_name:
        cur.close()
        conn.close()
        raise Error(f"Duplicate fingerprint pattern detected ({dup_ratio*100:.2f}%). Owner: {dup_name}")

    bg_id = get_or_create_blood_group(cur, conn, details["blood_group"])
    doctor_id = get_or_create_doctor(cur, conn, details["doctor_name"], details["doctor_phone"])
    ec_id = get_or_create_emergency(cur, conn,
                                    details["emergency_name"],
                                    details["emergency_relation"],
                                    details["emergency_phone"])

    sql = """
        INSERT INTO users_new
        (name, age, dob, blood_group_id, phone, address, aadhar,
         doctor_id, emergency_contact_id, fingerprint_path,
         fingerprint_hash, created_at)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,NOW())
    """

    params = (
        details["name"],
        int(details["age"]),
        details["dob"],
        bg_id,
        details["phone"],
        details["address"],
        details["aadhar"],
        doctor_id,
        ec_id,
        fp_path,
        fp_hash
    )

    cur.execute(sql, params)
    conn.commit()

    uid = cur.lastrowid
    cur.close()
    conn.close()
    return uid


# ---------------- DOB Calendar Popup (FIXED) ----------------

def open_calendar(entry_widget, age_entry):

    win = tk.Toplevel()
    win.title("Select DOB")
    win.geometry("300x320")

    if not TKCAL_AVAILABLE:
        label = tk.Label(win, text="ERROR: tkcalendar not installed.\npip install tkcalendar",
                         fg="red")
        label.pack(pady=20)
        return

    cal = Calendar(win, selectmode="day", date_pattern="yyyy-mm-dd")
    cal.pack(pady=10)

    def select_date():
        date_str = cal.get_date()

        # Set DOB
        entry_widget.delete(0, tk.END)
        entry_widget.insert(0, date_str)

        # Auto-age calculation
        dob = datetime.datetime.strptime(date_str, "%Y-%m-%d")
        today = datetime.datetime.today()
        age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))

        age_entry.delete(0, tk.END)
        age_entry.insert(0, str(age))

        win.destroy()

    tk.Button(win, text="Select", command=select_date).pack(pady=10)
# ---------------- Part 3 of 3 ----------------
# app18.py (Final UI + Register + Verify + Tracker + RUN)


# ---------------- MAIN APP UI ----------------

class LifeLineApp:
    def __init__(self, root):
        self.root = root
        root.title("LifeLine ID - Strict Capture")
        root.geometry("1020x680")
        root.configure(bg="#ECF0F1")

        # HEADER
        header = tk.Frame(root, bg="#2C3E50", height=70)
        header.pack(fill="x")
        tk.Label(header, text="🩺 LifeLine ID",
                 font=("Segoe UI", 22, "bold"),
                 fg="white", bg="#2C3E50").pack(pady=10)

        # Notebook tabs
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TNotebook.Tab",
                        font=("Segoe UI", 11, "bold"),
                        padding=[18, 8])

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True, padx=15, pady=15)

        self.reg_frame = tk.Frame(self.notebook, bg="white")
        self.ver_frame = tk.Frame(self.notebook, bg="white")
        self.tracker_frame = tk.Frame(self.notebook, bg="white")

        self.notebook.add(self.reg_frame, text="Register")
        self.notebook.add(self.ver_frame, text="Verify")
        self.notebook.add(self.tracker_frame, text="Tracker")

        self.last_register_fp = None
        self.last_verify_fp = None

        self._build_register_tab()
        self._build_verify_tab()
        self._build_tracker_tab()


    # ---------------- REGISTER TAB ----------------
    def _build_register_tab(self):

        container = ttk.Frame(self.reg_frame)
        container.pack(fill="both", expand=True, padx=12, pady=12)

        canvas = tk.Canvas(container, bg="white", highlightthickness=0)
        vscroll = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)

        form_frame = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=form_frame, anchor="nw")

        canvas.configure(yscrollcommand=vscroll.set)

        canvas.pack(side="left", fill="both", expand=True)
        vscroll.pack(side="left", fill="y")

        form_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        labels = [
            ("Name", "name"),
            ("Age (auto)", "age"),
            ("DOB", "dob"),
            ("Blood Group", "blood_group"),
            ("Phone", "phone"),
            ("Address", "address"),
            ("Aadhar", "aadhar"),
            ("Doctor Name", "doctor_name"),
            ("Doctor Phone", "doctor_phone"),
            ("Emergency Name", "emergency_name"),
            ("Emergency Relation", "emergency_relation"),
            ("Emergency Phone", "emergency_phone")
        ]

        self.entries = {}

        for i, (label, key) in enumerate(labels):
            ttk.Label(form_frame, text=label).grid(row=i, column=0, sticky="w", padx=6, pady=6)

            # ADDRESS MULTILINE
            if key == "address":
                txt = tk.Text(form_frame, width=36, height=4, relief="solid", bd=1)
                txt.grid(row=i, column=1, padx=6, pady=6)
                self.entries[key] = txt
                continue

            # BLOOD GROUP
            if key == "blood_group":
                cb = ttk.Combobox(form_frame, values=BLOOD_GROUPS,
                                  state="readonly", width=36)
                cb.set(BLOOD_GROUPS[0])
                cb.grid(row=i, column=1, padx=6, pady=6)
                self.entries[key] = cb
                continue

            # DOB FIELD (with calendar)
            if key == "dob":
                dob_entry = ttk.Entry(form_frame, width=30)
                dob_entry.grid(row=i, column=1, padx=6, pady=6, sticky="w")
                self.entries[key] = dob_entry

                btn = ttk.Button(form_frame, text="📅", width=4,
                                 command=lambda e=dob_entry,
                                        a=self.entries["age"]: open_calendar(e, a))
                btn.grid(row=i, column=1, padx=6, sticky="e")
                continue

            # AGE FIELD — must be Entry, NOT StringVar
            if key == "age":
                age_entry = ttk.Entry(form_frame, width=40)
                age_entry.grid(row=i, column=1, padx=6, pady=6)
                self.entries[key] = age_entry
                continue

            # DEFAULT: normal entry
            entry = ttk.Entry(form_frame, width=40)
            entry.grid(row=i, column=1, padx=6, pady=6)
            self.entries[key] = entry

        # Capture + Register Buttons
        ttk.Button(form_frame,
                   text="📸 Capture Fingerprint",
                   command=self._capture_register_button
                   ).grid(row=len(labels), column=0, pady=12)

        ttk.Button(form_frame,
                   text="✅ Register",
                   command=self._register_button
                   ).grid(row=len(labels), column=1, pady=12)

        self.fp_preview_label = ttk.Label(
            form_frame,
            text="No fingerprint captured",
            relief="sunken",
            width=50
        )
        self.fp_preview_label.grid(row=len(labels)+1, column=0,
                                   columnspan=2, pady=10)


    def _capture_register_button(self):
        saved = capture_fingerprint_preview(REGISTER_DIR)
        if saved:
            self.last_register_fp = saved
            img = Image.open(saved)
            img = img.resize((150, 150))
            tk_img = ImageTk.PhotoImage(img)
            self.fp_preview_label.configure(image=tk_img, text="")
            self.fp_preview_label.image = tk_img
            messagebox.showinfo("Captured", f"Fingerprint saved: {saved}")


    def _register_button(self):
        details = {}

        # Get all field values
        for key, widget in self.entries.items():
            if isinstance(widget, tk.Text):
                details[key] = widget.get("1.0", "end-1c").strip()
            else:
                details[key] = widget.get().strip()

        # Required fields
        required = [
            "name", "dob", "phone", "blood_group",
            "doctor_name", "doctor_phone",
            "emergency_name", "emergency_relation",
            "emergency_phone", "aadhar", "age"
        ]

        for r in required:
            if not details[r]:
                return messagebox.showwarning("Missing", f"Field '{r}' is required")

        if not self.last_register_fp:
            return messagebox.showwarning("Missing", "Please capture a fingerprint first.")

        try:
            uid = insert_user_normalized(details, self.last_register_fp)
            messagebox.showinfo("Success", f"User registered with ID: {uid}")
            self.last_register_fp = None
            self.fp_preview_label.configure(text="No fingerprint captured", image="")
        except Exception as e:
            messagebox.showerror("Error", str(e))


    # ---------------- VERIFY TAB ----------------
    def _build_verify_tab(self):
        left = ttk.Frame(self.ver_frame, padding=12)
        left.pack(side="left", fill="y")

        self.verify_preview_label = ttk.Label(left, text="No capture yet",
                                              relief="sunken", width=40)
        self.verify_preview_label.pack(pady=6)

        ttk.Button(left, text="📸 Capture For Verification",
                   command=self._capture_verify_button).pack(pady=6)

        ttk.Button(left, text="🔍 Run Match",
                   command=self._run_match_button).pack(pady=6)

        right = ttk.Frame(self.ver_frame, padding=12)
        right.pack(side="left", fill="both", expand=True)

        self.result_text = tk.Text(right, width=70, height=28,
                                   relief="solid", bd=1)
        self.result_text.pack(fill="both", expand=True)


    def _capture_verify_button(self):
        saved = capture_fingerprint_preview(VERIFY_DIR)
        if saved:
            self.last_verify_fp = saved
            img = Image.open(saved)
            img = img.resize((150, 150))
            tk_img = ImageTk.PhotoImage(img)
            self.verify_preview_label.configure(image=tk_img, text="")
            self.verify_preview_label.image = tk_img
            messagebox.showinfo("Captured", f"Verification fingerprint: {saved}")


    def _run_match_button(self):
        if not self.last_verify_fp:
            return messagebox.showerror("Error", "Capture fingerprint first.")

        best_user, good, ratio = find_best_match(self.last_verify_fp)
        self.result_text.delete("1.0", tk.END)

        if best_user and good >= MIN_GOOD_MATCHES and ratio >= MIN_MATCH_RATIO:

            user_id = best_user[0]
            log_verification(user_id)
            count = get_monthly_verification_count(user_id)

            fields = [
                "ID", "Name", "Age", "DOB", "Blood Group",
                "Phone", "Address", "Aadhar",
                "Doctor", "Doctor Phone",
                "Emergency Name", "Emergency Relation",
                "Emergency Phone", "Fingerprint Path"
            ]

            result = ""
            for i in range(len(fields)):
                result += f"{fields[i]}: {best_user[i]}\n"

            result += f"\nMatch Score: {good} good matches"
            result += f"\nMatch Ratio: {ratio*100:.2f}%"
            result += f"\nMonthly Verifications: {count}"

            self.result_text.insert("1.0", result)

        else:
            self.result_text.insert("1.0",
                f"❌ No match found OR low fingerprint quality.\n"
                f"Good Matches: {good}\n"
                f"Match Ratio: {ratio*100:.2f}%\n"
            )


    # ---------------- TRACKER TAB ----------------
    def _build_tracker_tab(self):
        top = ttk.Frame(self.tracker_frame)
        top.pack(fill="x", pady=8)

        ttk.Button(top, text="Refresh Tracker",
                   command=self._load_verification_log).pack(side="right", padx=10)

        self.tracker_tree = ttk.Treeview(
            self.tracker_frame,
            columns=("Name", "Count", "Last Verified"),
            show="headings"
        )

        self.tracker_tree.heading("Name", text="Name")
        self.tracker_tree.heading("Count", text="Verifications")
        self.tracker_tree.heading("Last Verified", text="Last Time")

        self.tracker_tree.pack(expand=True, fill="both", padx=20, pady=12)


    def _load_verification_log(self):
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute("""
            SELECT u.name, COUNT(v.id), MAX(v.verified_at)
            FROM verification_logs v
            JOIN users_new u ON v.user_id = u.id
            GROUP BY v.user_id
            ORDER BY COUNT(v.id) DESC
        """)

        rows = cur.fetchall()
        cur.close()
        conn.close()

        self.tracker_tree.delete(*self.tracker_tree.get_children())

        for r in rows:
            self.tracker_tree.insert("", "end", values=r)


# ---------------- RUN APP ----------------
if __name__ == "__main__":
    ensure_db_schema()
    root = tk.Tk()
    app = LifeLineApp(root)
    root.mainloop()
