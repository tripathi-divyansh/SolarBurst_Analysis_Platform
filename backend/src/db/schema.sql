-- SolarBurst PostgreSQL Database Schema
-- Provides structured persistent storage for datasets, background jobs, burst catalogs, models, and audit trails.

CREATE TABLE IF NOT EXISTS datasets (
    id VARCHAR(64) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    filename VARCHAR(255) NOT NULL,
    filepath TEXT NOT NULL,
    format VARCHAR(32) NOT NULL,
    size_bytes BIGINT NOT NULL,
    sha256 VARCHAR(64) NOT NULL,
    instrument VARCHAR(64) DEFAULT 'XSM',
    quantity VARCHAR(64) DEFAULT 'count_rate',
    unit VARCHAR(64) DEFAULT 'count / s',
    total_points INT DEFAULT 0,
    valid_points INT DEFAULT 0,
    span_s DOUBLE PRECISION DEFAULT 0.0,
    metadata_json JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS jobs (
    id VARCHAR(64) PRIMARY KEY,
    dataset_id VARCHAR(64) REFERENCES datasets(id) ON DELETE CASCADE,
    status VARCHAR(32) NOT NULL, -- 'queued', 'running', 'completed', 'failed', 'cancelled'
    progress INT DEFAULT 0,
    message TEXT,
    config_json JSONB,
    result_json JSONB,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS burst_catalog (
    id VARCHAR(64) PRIMARY KEY,
    job_id VARCHAR(64) REFERENCES jobs(id) ON DELETE CASCADE,
    burst_id VARCHAR(32) NOT NULL,
    candidate_id VARCHAR(32),
    peak_time_met DOUBLE PRECISION NOT NULL,
    peak_time_iso VARCHAR(64) NOT NULL,
    start_time_iso VARCHAR(64) NOT NULL,
    end_time_iso VARCHAR(64) NOT NULL,
    duration_s DOUBLE PRECISION NOT NULL,
    net_peak DOUBLE PRECISION NOT NULL,
    total_peak DOUBLE PRECISION NOT NULL,
    peak_snr DOUBLE PRECISION NOT NULL,
    fluence DOUBLE PRECISION NOT NULL,
    asymmetry_rho DOUBLE PRECISION,
    morphology_class VARCHAR(64),
    duration_class VARCHAR(64),
    intensity_class VARCHAR(128),
    reliability VARCHAR(32),
    reduced_chi2 DOUBLE PRECISION,
    ml_score DOUBLE PRECISION,
    decision VARCHAR(32), -- 'accepted', 'review', 'rejected'
    audit_status VARCHAR(32) DEFAULT 'auto', -- 'auto', 'approved', 'rejected'
    audit_notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS audit_trail (
    id VARCHAR(64) PRIMARY KEY,
    burst_id VARCHAR(64) REFERENCES burst_catalog(id) ON DELETE CASCADE,
    action VARCHAR(32) NOT NULL, -- 'approve', 'reject', 'flag', 'edit_notes'
    previous_decision VARCHAR(32),
    new_decision VARCHAR(32),
    notes TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS models (
    id VARCHAR(64) PRIMARY KEY,
    name VARCHAR(128) NOT NULL,
    model_type VARCHAR(64) NOT NULL,
    filepath TEXT NOT NULL,
    sha256 VARCHAR(64) NOT NULL,
    card_json JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indices for fast searching and filtering
CREATE INDEX IF NOT EXISTS idx_jobs_dataset ON jobs(dataset_id);
CREATE INDEX IF NOT EXISTS idx_catalog_job ON burst_catalog(job_id);
CREATE INDEX IF NOT EXISTS idx_catalog_decision ON burst_catalog(decision);
CREATE INDEX IF NOT EXISTS idx_catalog_peak ON burst_catalog(peak_time_met);
