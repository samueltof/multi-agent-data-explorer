# Clinical Trials Database Creation

This directory contains scripts to create a SQLite database from CSV files with clinical trials data.

## Prerequisites

- Python 3.6+
- Required packages:
  - pandas
  - sqlite3 (included in Python standard library)

Install required packages:

```bash
pip install pandas
```

## Data Files

The scripts expect the following CSV files to be in the `data/dummy_data_clinical_trials` directory:

- `patients_data.csv`
- `sequencing_data.csv`
- `clinical_trials_data.csv`
- `drugs_data.csv`
- `trial_enrollments_data.csv`

## Usage

### Option 1: Run the main script

```bash
cd src/scripts/create_dummy_clinical_trials_db
python main.py
```

This will:
1. Create a new SQLite database at `data/clinical_trials.db`
2. Create the necessary tables
3. Import data from the CSV files

### Option 2: Run scripts individually

1. Create the database schema:

```bash
python create_schema.py
```

2. Import the data:

```bash
python import_data.py
```

## Database Schema

The database contains the following tables:

- `patients`: Patient demographics and diagnosis
- `sequencing_data`: Genomic sequencing data linked to patients
- `clinical_trials`: Clinical trial information
- `drugs`: Drug information
- `trial_enrollment`: Link between patients and trials they are enrolled in

## Querying the Database

You can query the database using SQLite:

```bash
sqlite3 data/clinical_trials.db
```

Example queries:

```sql
-- List all patients
SELECT * FROM patients;

-- Count patients enrolled in each trial
SELECT trial_id, COUNT(*) as patient_count 
FROM trial_enrollment 
GROUP BY trial_id;

-- Find patients with specific mutations
SELECT p.patient_id, p.first_name, p.last_name, s.mutation 
FROM patients p
JOIN sequencing_data s ON p.patient_id = s.patient_id
WHERE s.mutation LIKE '%BRAF%';
``` 