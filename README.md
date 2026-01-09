# LifeLine ID 🩺

A biometric identity management system that uses **fingerprint recognition** for user registration, verification, and identity tracking. Built for medical and emergency identification purposes.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![MySQL](https://img.shields.io/badge/MySQL-8.0-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Database Setup](#database-setup)
- [Usage](#usage)
- [Screenshots](#screenshots)
- [API Endpoints](#api-endpoints)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

**LifeLine ID** is a comprehensive biometric identity management system designed for healthcare and emergency response scenarios. It enables:

- Patient/user registration with fingerprint biometrics
- Quick identity verification via fingerprint matching
- Medical information storage (blood group, emergency contacts, doctor details)
- Verification activity tracking and monitoring

The system uses **ORB (Oriented FAST and Rotated BRIEF)** algorithm for fingerprint feature extraction and matching, making it efficient for real-time identity verification.

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📸 **Fingerprint Capture** | Real-time webcam-based fingerprint capture with ROI detection |
| 🔐 **User Registration** | Complete user profile with medical and emergency information |
| 🔍 **Fingerprint Verification** | Fast biometric matching using ORB algorithm |
| 🩸 **Medical Data** | Blood group, doctor info, and emergency contacts storage |
| 📊 **Verification Tracker** | Monitor and track verification activity per user |
| 🔒 **Duplicate Prevention** | Hash-based and pattern-based duplicate detection |
| 📅 **DOB Calendar** | Interactive calendar for date of birth selection with auto-age calculation |
| 🖼️ **Vector Fingerprint Support** | Accept both photographic and stylized fingerprint images |

## 🛠️ Technology Stack

### Backend / Core Application

| Technology | Purpose |
|------------|---------|
| **Python 3.8+** | Primary programming language |
| **OpenCV (cv2)** | Image processing, fingerprint detection & matching |
| **NumPy** | Numerical operations for image analysis |
| **MySQL Connector** | Database connectivity |
| **Tkinter** | Desktop GUI framework |
| **Pillow (PIL)** | Image handling and display |
| **tkcalendar** | Calendar widget for date selection |

### Database

| Technology | Purpose |
|------------|---------|
| **MySQL 8.0** | Relational database for user and fingerprint data |
| **Normalized Schema** | Optimized tables for blood groups, doctors, emergency contacts |

### Web Interface (Alternative)


### Computer Vision / Biometrics

| Algorithm/Technique | Purpose |
|---------------------|---------|
| **ORB (Oriented FAST and Rotated BRIEF)** | Feature detection and description for fingerprint matching |
| **CLAHE** | Contrast Limited Adaptive Histogram Equalization for image enhancement |
| **Canny Edge Detection** | Edge detection for fingerprint pattern analysis |
| **BFMatcher (Brute Force)** | Feature matching with Hamming distance |
| **SHA-256 Hashing** | Fingerprint file integrity and duplicate detection |

## 📁 Project Structure

```
LifeLineID/
├── newfinal.py              # Main application (Desktop GUI - Loose mode)
├── appfinal.py              # Alternative app with stricter matching
├── database.sql             # Database schema and sample data
├── fingerprint_detector_demo.py  # Standalone fingerprint detection demo
│
├── api/                     # PHP API endpoints
│   ├── get_user.php         # Retrieve user data
│   └── save_user.php        # Save user registration
│
├── backend/                 # Python backend utilities
│   ├── app2.py              # Flask web server
│   ├── fingerprint_utils.py # Fingerprint matching utilities
│   └── __pycache__/
│
├── fingerprints/            # Fingerprint image storage
│   ├── enhanced/            # Enhanced fingerprint images
│   ├── register/            # Registration fingerprints
│   ├── registered/          # Confirmed registered fingerprints
│   ├── roi/                 # Region of Interest extractions
│   ├── verified/            # Verification capture images
│   └── verify/              # Verification process images
│
├── static/                  # Web static assets
│   ├── script.js            # JavaScript functions
│   └── style.css            # Stylesheet
│
├── templates/               # HTML templates
│   ├── index.html           # Main landing page
│   ├── index1.html          # Alternative index
│   ├── register.html        # Registration form
│   ├── verify.html          # Verification interface
│   └── save.php             # Form submission handler
│
├── trial/                   # Development/testing versions
│   ├── app.py - app5.py     # Various app iterations
│
├── captured_fingerprints/   # Captured fingerprint storage
├── captures/                # General captures
└── verify_fp/               # Verification fingerprints
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- MySQL Server 8.0
- Webcam (for fingerprint capture)
- pip (Python package manager)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/LifeLineID.git
cd LifeLineID
```

### Step 2: Install Python Dependencies

```bash
pip install opencv-python pillow mysql-connector-python numpy tkcalendar
```

### Step 3: Configure Database Connection

Edit the `DB_CONFIG` in `newfinal.py`:

```python
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "your_password",
    "database": "lifelineid_new"
}
```

### Step 4: Run the Application

```bash
python newfinal.py
```

## 🗄️ Database Setup

### Option 1: Automatic Setup

The application automatically creates the database and tables on first run.

### Option 2: Manual Setup

```sql
-- Create database
CREATE DATABASE lifelineid_new;
USE lifelineid_new;

-- Blood groups table
CREATE TABLE blood_groups (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(10) NOT NULL UNIQUE
);

INSERT INTO blood_groups (name) VALUES 
('A+'), ('A-'), ('B+'), ('B-'), ('AB+'), ('AB-'), ('O+'), ('O-');

-- Doctors table
CREATE TABLE doctors (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(200),
    phone VARCHAR(20) UNIQUE
);

-- Emergency contacts table
CREATE TABLE emergency_contacts (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(200),
    relation VARCHAR(50),
    phone VARCHAR(20) UNIQUE
);

-- Users table
CREATE TABLE users_new (
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
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (blood_group_id) REFERENCES blood_groups(id),
    FOREIGN KEY (doctor_id) REFERENCES doctors(id),
    FOREIGN KEY (emergency_contact_id) REFERENCES emergency_contacts(id)
);

-- Verification logs table
CREATE TABLE verification_logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT,
    verified_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users_new(id) ON DELETE CASCADE
);
```

## 📖 Usage

### Desktop Application

1. **Register Tab**: 
   - Fill in user details (name, DOB, blood group, phone, etc.)
   - Click "📸 Capture Fingerprint" to capture via webcam
   - Position finger in the green box and press 's' to save
   - Click "✅ Register" to save user

2. **Verify Tab**:
   - Click "📸 Capture For Verification" 
   - Position finger and capture
   - Click "🔍 Run Match" to find matching user

3. **Tracker Tab**:
   - View verification activity logs
   - Monitor users with high verification frequency

### Fingerprint Capture Controls

| Key | Action |
|-----|--------|
| `s` | Save/Capture fingerprint |
| `q` | Cancel/Quit capture |

## 🔧 API Endpoints

### PHP API (Optional Web Interface)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/save_user.php` | POST | Register new user with fingerprint |
| `/api/get_user.php` | GET | Retrieve user information |

### Flask API (Backend)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/capture` | POST | Capture fingerprint from webcam |
| `/register` | POST | Register user with captured fingerprint |
| `/verify` | POST | Verify fingerprint against database |

## ⚙️ Configuration

### Matching Thresholds

Adjust in `newfinal.py` for sensitivity:

```python
MIN_GOOD_MATCHES = 8      # Minimum ORB feature matches
MIN_MATCH_RATIO = 0.06    # Match ratio threshold
DUPLICATE_MATCH_RATIO = 0.18  # Duplicate detection threshold
ROI_SIZE = 300            # Region of interest size (pixels)
```

### Fingerprint Mode

```python
ALLOW_VECTOR_FINGERPRINTS = True  # Accept stylized/vector fingerprints
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- Your Name - Initial work

## 🙏 Acknowledgments

- OpenCV community for computer vision tools
- Python Tkinter for GUI framework
- MySQL for reliable database solutions

---

<p align="center">Made with ❤️ for healthcare and emergency identification</p>

