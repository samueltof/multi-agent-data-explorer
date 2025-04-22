import sqlite3
import pandas as pd

# Connect to SQLite
conn = sqlite3.connect("database/dummy_roche_data.db")
cursor = conn.cursor()

# Function to load CSV into table
def load_csv_to_db(csv_path, table_name):
    df = pd.read_csv(csv_path)
    df.to_sql(table_name, conn, if_exists='append', index=False)

# Load all data
load_csv_to_db("data/patients.csv", "patients")
load_csv_to_db("data/sequencing_data.csv", "sequencing_data")
load_csv_to_db("data/clinical_trials.csv", "clinical_trials")
load_csv_to_db("data/drugs.csv", "drugs")
load_csv_to_db("data/trial_enrollment.csv", "trial_enrollment")

print("Database successfully populated!")
conn.close()
