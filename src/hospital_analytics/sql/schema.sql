-- ===========================================================================
-- Module 1: the MySQL schema
--
-- SOURCE DATASET
--   Diabetes 130-US Hospitals for Years 1999-2008
--   UCI ML Repository dataset 296  |  DOI 10.24432/C5230J  |  CC BY 4.0
--   Strack et al. (2014), BioMed Research International, 781670.
--
-- DESIGN NOTES WORTH DEFENDING IN A DEMO
--   * Three tables, not one. A patient receives MANY medications, so that is a
--     one-to-many relationship and it needs its own table. Storing a
--     comma-separated drug list in a single column cannot be queried.
--   * `disease_reference` holds the ICD-9 group metadata in ONE place. Change
--     how a group is classified and you edit one row, not 100,000.
--   * FOREIGN KEYs make an orphan medication physically impossible to store.
--   * Indexes sit on the columns actually filtered in the dashboard, so charts
--     stay fast as the table grows.
--   * `data_load_log` records every ETL run - the auditability a real hospital
--     data team would insist on.
--
-- IMPORTANT: one row of `patients` is one ENCOUNTER, not one person. The same
-- person can appear many times, identified by `person_id`. That distinction is
-- why `person_id` is indexed - Module 4 must split train/test BY PERSON, or the
-- same individual lands in both sides and the accuracy score is inflated.
-- ===========================================================================

CREATE DATABASE IF NOT EXISTS hospital_analytics
    CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

USE hospital_analytics;

-- Dropped child-first so the foreign keys never block the drop.
DROP TABLE IF EXISTS medications;
DROP TABLE IF EXISTS patients;
DROP TABLE IF EXISTS disease_reference;
DROP TABLE IF EXISTS data_load_log;

-- ---------------------------------------------------------------------------
-- 1. Lookup table: ICD-9 diagnosis groups
-- ---------------------------------------------------------------------------
CREATE TABLE disease_reference (
    diagnosis        VARCHAR(64)  NOT NULL,
    icd9_range       VARCHAR(32)  NULL,
    disease_category VARCHAR(64)  NULL,
    is_chronic       TINYINT(1)   NOT NULL DEFAULT 0,
    department       VARCHAR(64)  NULL,
    PRIMARY KEY (diagnosis)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ---------------------------------------------------------------------------
-- 2. Main table: one row per hospital ENCOUNTER
-- ---------------------------------------------------------------------------
CREATE TABLE patients (
    patient_id             VARCHAR(32)  NOT NULL,   -- encounter_id in the source
    person_id              VARCHAR(32)  NULL,       -- patient_nbr: the human
    encounter_seq          INT          NULL,       -- 1 = first recorded visit
    is_repeat_patient      TINYINT(1)   NULL,

    -- Demographics. Age arrives as a 10-year band, so `age` is the midpoint.
    age                    DECIMAL(5,1) NULL,
    age_group              VARCHAR(16)  NULL,
    gender                 VARCHAR(16)  NULL,
    race                   VARCHAR(32)  NULL,

    -- Diagnosis: primary plus two secondary, grouped from ICD-9 codes
    diagnosis              VARCHAR(64)  NULL,
    diagnosis_secondary    VARCHAR(64)  NULL,
    diagnosis_tertiary     VARCHAR(64)  NULL,
    icd9_primary           VARCHAR(16)  NULL,
    num_diagnoses          INT          NULL,

    -- Admission and discharge context
    admission_type         VARCHAR(48)  NULL,
    admission_source       VARCHAR(64)  NULL,
    discharge_disposition  VARCHAR(64)  NULL,
    department             VARCHAR(80)  NULL,       -- medical_specialty

    -- Utilisation during this stay
    length_of_stay         INT          NULL,       -- 1-14 by dataset design
    num_lab_procedures     INT          NULL,
    num_procedures         INT          NULL,
    num_medications        INT          NULL,

    -- Utilisation in the year BEFORE this stay: the strongest predictor family
    num_prior_admissions   INT          NULL,
    num_prior_emergency    INT          NULL,
    num_prior_outpatient   INT          NULL,
    num_prior_visits_total INT          NULL,

    -- Glycaemic control. Categorical in the source, not numeric.
    a1c_result             VARCHAR(16)  NULL,       -- Not Tested / Norm / >7 / >8
    glucose_serum          VARCHAR(16)  NULL,       -- Not Tested / Norm / >200 / >300
    a1c_tested             TINYINT(1)   NULL,
    glucose_tested         TINYINT(1)   NULL,

    -- Treatment response
    medication_changed     TINYINT(1)   NULL,
    on_diabetes_meds       TINYINT(1)   NULL,

    -- Targets. readmitted_30d is the one Module 4 predicts.
    readmitted_30d         TINYINT(1)   NULL,
    readmitted_any         TINYINT(1)   NULL,

    ingested_at            TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (patient_id),
    KEY idx_person          (person_id),
    KEY idx_diagnosis       (diagnosis),
    KEY idx_readmitted      (readmitted_30d),
    KEY idx_department      (department),
    KEY idx_prior_admits    (num_prior_admissions),
    CONSTRAINT fk_patients_diagnosis
        FOREIGN KEY (diagnosis) REFERENCES disease_reference (diagnosis)
        ON UPDATE CASCADE ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ---------------------------------------------------------------------------
-- 3. Medications: one row per drug prescribed (the one-to-many side)
--
-- The source file stores 23 drugs as 23 columns of "No"/"Steady"/"Up"/"Down".
-- The adapter reshapes that into this long table, which is what makes questions
-- like "how many patients had insulin adjusted?" answerable in SQL.
-- ---------------------------------------------------------------------------
CREATE TABLE medications (
    medication_row_id BIGINT       NOT NULL AUTO_INCREMENT,
    patient_id        VARCHAR(32)  NOT NULL,
    medication_name   VARCHAR(64)  NOT NULL,
    drug_class        VARCHAR(64)  NULL,
    dose_status       VARCHAR(16)  NULL,   -- Steady / Up / Down
    PRIMARY KEY (medication_row_id),
    KEY idx_med_patient (patient_id),
    KEY idx_med_name    (medication_name),
    KEY idx_med_class   (drug_class),
    CONSTRAINT fk_medications_patient
        FOREIGN KEY (patient_id) REFERENCES patients (patient_id)
        ON UPDATE CASCADE ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ---------------------------------------------------------------------------
-- 4. Audit log: what was loaded, when, and whether it worked
-- ---------------------------------------------------------------------------
CREATE TABLE data_load_log (
    load_id       BIGINT       NOT NULL AUTO_INCREMENT,
    source_file   VARCHAR(255) NULL,
    target_table  VARCHAR(64)  NULL,
    rows_in_file  INT          NULL,
    rows_loaded   INT          NULL,
    rows_rejected INT          NULL,
    status        VARCHAR(32)  NULL,
    message       TEXT         NULL,
    loaded_at     TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (load_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
