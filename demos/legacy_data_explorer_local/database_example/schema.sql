-- Patients Table
CREATE TABLE patients (
    patient_id UUID PRIMARY KEY,
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    dob DATE,
    gender VARCHAR(10),
    ethnicity VARCHAR(50),
    diagnosis TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Sequencing Data Table
CREATE TABLE sequencing_data (
    seq_id UUID PRIMARY KEY,
    patient_id UUID REFERENCES patients(patient_id),
    sample_id VARCHAR(50),
    sequencing_tech VARCHAR(50),
    mutation TEXT,
    mutation_effect TEXT,
    date_performed DATE
);

-- Clinical Trials Table
CREATE TABLE clinical_trials (
    trial_id UUID PRIMARY KEY,
    trial_name VARCHAR(100),
    start_date DATE,
    end_date DATE,
    phase VARCHAR(10),
    condition VARCHAR(100),
    status VARCHAR(50),
    principal_investigator VARCHAR(100)
);

-- Drugs Table
CREATE TABLE drugs (
    drug_id UUID PRIMARY KEY,
    drug_name VARCHAR(100),
    mechanism_of_action TEXT,
    target VARCHAR(100),
    approval_status VARCHAR(50),
    manufacturer VARCHAR(100)
);

-- Trial Enrollment Table
CREATE TABLE trial_enrollment (
    enrollment_id UUID PRIMARY KEY,
    patient_id UUID REFERENCES patients(patient_id),
    trial_id UUID REFERENCES clinical_trials(trial_id),
    enrollment_date DATE
);
