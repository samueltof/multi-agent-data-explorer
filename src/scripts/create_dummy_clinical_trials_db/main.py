#!/usr/bin/env python3
"""
Script to create and populate a SQLite database with clinical trials data.
"""
import os
import sys
import time
from import_data import import_data

def main():
    start_time = time.time()
    print("Starting database creation and population process...")
    
    # Import the data (this also creates the schema)
    import_data()
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"Process completed in {duration:.2f} seconds.")
    
    # Get the database path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
    db_path = os.path.join(project_root, "databases", "dummy_db_clinical_trials", "clinical_trials.db")
    
    print(f"Database created at: {db_path}")
    print("You can connect to this database using:")
    print(f"  sqlite3 {db_path}")

if __name__ == "__main__":
    main() 