#!/usr/bin/env python3
import sqlite3
import pandas as pd
import os
import sys
from create_schema import create_schema

def import_data():
    # Get the path to the parent directory of src
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
    
    # CSV files path
    csv_dir = os.path.join(project_root, "data", "dummy_data_clinical_trials")
    
    # Create the database schema
    db_path = create_schema()
    
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    
    # List of CSV files and corresponding table names
    files_and_tables = [
        ("patients_data.csv", "patients"),
        ("sequencing_data.csv", "sequencing_data"),
        ("clinical_trials_data.csv", "clinical_trials"),
        ("drugs_data.csv", "drugs"),
        ("trial_enrollments_data.csv", "trial_enrollment")
    ]
    
    # Import each CSV file into the corresponding table
    for csv_file, table_name in files_and_tables:
        csv_path = os.path.join(csv_dir, csv_file)
        
        if not os.path.exists(csv_path):
            print(f"Warning: {csv_path} does not exist. Skipping...")
            continue
        
        print(f"Importing {csv_file} into {table_name}...")
        
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        # Import data into the table
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        
        # Get the number of records imported
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        print(f"Imported {count} records into {table_name}")
    
    # Commit and close the connection
    conn.commit()
    conn.close()
    
    print("Data import completed successfully.")

if __name__ == "__main__":
    import_data() 