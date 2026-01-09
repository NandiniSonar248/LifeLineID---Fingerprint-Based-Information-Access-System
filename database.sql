CREATE DATABASE lifeline_db;

USE lifeline_db;

CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    age INT,
    dob DATE NOT NULL,
    blood_group VARCHAR(5) NOT NULL,
    phone VARCHAR(20) NOT NULL,
    address TEXT,
    emergency_contact VARCHAR(20) NOT NULL,
    aadhar VARCHAR(20),
    doctor_contact VARCHAR(20) NOT NULL,
    fingerprint_path VARCHAR(255) timestamp DATETIME
);

--After normalization

-- 1. Blood groups table
CREATE TABLE blood_groups (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(5) NOT NULL UNIQUE
);

INSERT INTO
    blood_groups (name)
VALUES ('A+'),
    ('A-'),
    ('B+'),
    ('B-'),
    ('AB+'),
    ('AB-'),
    ('O+'),
    ('O-');

-- 2. Doctors table
CREATE TABLE doctors (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    phone VARCHAR(15) NOT NULL
);

-- 3. Emergency contacts table
CREATE TABLE emergency_contacts (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    relation VARCHAR(50) NOT NULL,
    phone VARCHAR(15) NOT NULL
);

-- 4. Users table
CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    age INT NOT NULL,
    dob DATE NOT NULL,
    blood_group_id INT NOT NULL,
    phone VARCHAR(15) NOT NULL,
    address TEXT,
    emergency_contact_id INT NOT NULL,
    aadhar VARCHAR(20) NOT NULL UNIQUE,
    doctor_id INT NOT NULL,
    fingerprint_path VARCHAR(255),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (blood_group_id) REFERENCES blood_groups (id),
    FOREIGN KEY (doctor_id) REFERENCES doctors (id),
    FOREIGN KEY (emergency_contact_id) REFERENCES emergency_contacts (id)
);

--New Data Entry(Normalized)

INSERT INTO
    doctors (name, phone)
VALUES ('Dr. Sharma', '9876543210');

INSERT INTO
    emergency_contacts (name, relation, phone)
VALUES (
        'Ramesh Kumar',
        'Father',
        '9876543211'
    );

INSERT INTO
    users_new (
        name,
        age,
        dob,
        phone,
        address,
        aadhar,
        fingerprint_path,
        doctor_id,
        blood_group_id,
        emergency_contact_id
    )
VALUES (
        'Sham',
        21,
        '2004-01-01',
        '9876543212',
        'Pune, India',
        '1234-5678-9012',
        'fingerprints/sham.png',
        1,
        1,
        1
    );