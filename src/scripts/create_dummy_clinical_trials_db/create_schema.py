#!/usr/bin/env python3
import sqlite3
import os
import sys

def create_schema():
    # Get the path to the parent directory of src
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
    
    # Database file path
    db_path = os.path.join(project_root, "data", "clinical_trials.db")
    
    print(f"Creating database at: {db_path}")
    
    # Ensure the parent directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # Remove existing database if it exists
    if os.path.exists(db_path):
        os.remove(db_path)
        print("Removed existing database.")
    
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute('''
    CREATE TABLE patients (
        patient_id TEXT PRIMARY KEY,
        first_name TEXT,
        last_name TEXT,
        dob TEXT,
        gender TEXT,
        ethnicity TEXT,
        diagnosis TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE sequencing_data (
        seq_id TEXT PRIMARY KEY,
        patient_id TEXT,
        sample_id TEXT,
        sequencing_tech TEXT,
        mutation TEXT,
        mutation_effect TEXT,
        date_performed TEXT,
        FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE clinical_trials (
        trial_id TEXT PRIMARY KEY,
        trial_name TEXT,
        start_date TEXT,
        end_date TEXT,
        phase TEXT,
        condition TEXT,
        status TEXT,
        principal_investigator TEXT
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE drugs (
        drug_id TEXT PRIMARY KEY,
        drug_name TEXT,
        mechanism_of_action TEXT,
        target TEXT,
        approval_status TEXT,
        manufacturer TEXT
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE trial_enrollment (
        enrollment_id TEXT PRIMARY KEY,
        patient_id TEXT,
        trial_id TEXT,
        enrollment_date TEXT,
        FOREIGN KEY (patient_id) REFERENCES patients(patient_id),
        FOREIGN KEY (trial_id) REFERENCES clinical_trials(trial_id)
    )
    ''')
    
    # Commit and close the connection
    conn.commit()
    conn.close()
    
    print("Database schema created successfully.")
    return db_path

if __name__ == "__main__":
    create_schema() 