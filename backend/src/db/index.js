/**
 * SolarBurst Database Layer
 * Dual-mode: Supports PostgreSQL (via pg.Pool) and local zero-dependency offline persistent storage.
 */

import fs from 'fs';
import path from 'path';
import pg from 'pg';
import { config } from '../config.js';

const { Pool } = pg;

class DatabaseManager {
  constructor() {
    this.isPg = false;
    this.pgPool = null;
    this.localDbPath = path.join(path.dirname(config.uploadDir), 'data/solarburst_store.json');
    this.localStore = {
      datasets: {},
      jobs: {},
      burst_catalog: {},
      audit_trail: [],
      models: {}
    };
  }

  async init() {
    if (config.databaseUrl) {
      try {
        console.log(`[DB] Connecting to PostgreSQL at ${config.databaseUrl.replace(/:[^:]*@/, ':***@')}...`);
        this.pgPool = new Pool({ connectionString: config.databaseUrl });
        const client = await this.pgPool.connect();
        
        // Execute schema migrations
        const schemaPath = path.join(path.dirname(new URL(import.meta.url).pathname), 'schema.sql').replace(/^\/([A-Za-z]:)/, '$1');
        if (fs.existsSync(schemaPath)) {
          const sql = fs.readFileSync(schemaPath, 'utf-8');
          await client.query(sql);
          console.log('[DB] PostgreSQL schema initialized successfully.');
        }
        client.release();
        this.isPg = true;
        return;
      } catch (err) {
        console.warn(`[DB] PostgreSQL connection failed (${err.message}). Falling back to local offline JSON store.`);
      }
    }

    // Local offline storage initialization
    const dir = path.dirname(this.localDbPath);
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
    
    if (fs.existsSync(this.localDbPath)) {
      try {
        const raw = fs.readFileSync(this.localDbPath, 'utf-8');
        this.localStore = JSON.parse(raw);
        console.log('[DB] Local persistent store loaded successfully.');
      } catch (e) {
        console.warn('[DB] Resetting corrupt local store.');
        this._saveLocal();
      }
    } else {
      this._saveLocal();
      console.log('[DB] Initialized fresh local offline store.');
    }
  }

  _saveLocal() {
    fs.writeFileSync(this.localDbPath, JSON.stringify(this.localStore, null, 2), 'utf-8');
  }

  // --- Dataset Operations ---
  async saveDataset(dataset) {
    if (this.isPg) {
      const q = `
        INSERT INTO datasets (id, name, filename, filepath, format, size_bytes, sha256, instrument, quantity, unit, total_points, valid_points, span_s, metadata_json)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
        ON CONFLICT (id) DO UPDATE SET
          name = EXCLUDED.name, total_points = EXCLUDED.total_points, valid_points = EXCLUDED.valid_points, span_s = EXCLUDED.span_s, metadata_json = EXCLUDED.metadata_json
        RETURNING *;
      `;
      const values = [
        dataset.id, dataset.name, dataset.filename, dataset.filepath, dataset.format,
        dataset.size_bytes, dataset.sha256, dataset.instrument || 'XSM', dataset.quantity || 'count_rate',
        dataset.unit || 'count / s', dataset.total_points || 0, dataset.valid_points || 0,
        dataset.span_s || 0.0, JSON.stringify(dataset.metadata || {})
      ];
      const res = await this.pgPool.query(q, values);
      return res.rows[0];
    } else {
      this.localStore.datasets[dataset.id] = {
        ...dataset,
        created_at: dataset.created_at || new Date().toISOString()
      };
      this._saveLocal();
      return this.localStore.datasets[dataset.id];
    }
  }

  async getDataset(id) {
    if (this.isPg) {
      const res = await this.pgPool.query('SELECT * FROM datasets WHERE id = $1', [id]);
      return res.rows[0] || null;
    } else {
      return this.localStore.datasets[id] || null;
    }
  }

  async listDatasets() {
    if (this.isPg) {
      const res = await this.pgPool.query('SELECT * FROM datasets ORDER BY created_at DESC');
      return res.rows;
    } else {
      return Object.values(this.localStore.datasets).sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
    }
  }

  // --- Job Operations ---
  async createJob(job) {
    const jobRecord = {
      ...job,
      status: job.status || 'queued',
      progress: job.progress || 0,
      created_at: new Date().toISOString(),
      completed_at: null
    };

    if (this.isPg) {
      const q = `
        INSERT INTO jobs (id, dataset_id, status, progress, message, config_json)
        VALUES ($1, $2, $3, $4, $5, $6)
        RETURNING *;
      `;
      const res = await this.pgPool.query(q, [
        jobRecord.id, jobRecord.dataset_id, jobRecord.status,
        jobRecord.progress, jobRecord.message || '', JSON.stringify(jobRecord.config || {})
      ]);
      return res.rows[0];
    } else {
      this.localStore.jobs[jobRecord.id] = jobRecord;
      this._saveLocal();
      return jobRecord;
    }
  }

  async updateJob(id, updates) {
    if (this.isPg) {
      const sets = [];
      const values = [id];
      let idx = 2;
      for (const [key, val] of Object.entries(updates)) {
        if (key === 'config_json' || key === 'result_json') {
          sets.push(`${key} = $${idx}`);
          values.push(JSON.stringify(val));
        } else {
          sets.push(`${key} = $${idx}`);
          values.push(val);
        }
        idx++;
      }
      const q = `UPDATE jobs SET ${sets.join(', ')} WHERE id = $1 RETURNING *;`;
      const res = await this.pgPool.query(q, values);
      return res.rows[0];
    } else {
      if (!this.localStore.jobs[id]) return null;
      this.localStore.jobs[id] = { ...this.localStore.jobs[id], ...updates };
      this._saveLocal();
      return this.localStore.jobs[id];
    }
  }

  async getJob(id) {
    if (this.isPg) {
      const res = await this.pgPool.query('SELECT * FROM jobs WHERE id = $1', [id]);
      return res.rows[0] || null;
    } else {
      return this.localStore.jobs[id] || null;
    }
  }

  // --- Burst Catalog Operations ---
  async saveBursts(jobId, bursts) {
    for (const b of bursts) {
      const id = `${jobId}_${b.burst_id}`;
      const record = { ...b, id, job_id: jobId, created_at: new Date().toISOString() };
      
      if (this.isPg) {
        const q = `
          INSERT INTO burst_catalog (
            id, job_id, burst_id, candidate_id, peak_time_met, peak_time_iso,
            start_time_iso, end_time_iso, duration_s, net_peak, total_peak,
            peak_snr, fluence, asymmetry_rho, morphology_class, duration_class,
            intensity_class, reliability, reduced_chi2, ml_score, decision,
            audit_status, audit_notes
          ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14,
            $15, $16, $17, $18, $19, $20, $21, $22, $23
          ) ON CONFLICT (id) DO NOTHING;
        `;
        await this.pgPool.query(q, [
          id, jobId, b.burst_id, b.candidate_id, b.peak_time_met, b.peak_time_iso,
          b.start_time_iso, b.end_time_iso, b.duration_s, b.net_peak, b.total_peak,
          b.peak_snr, b.fluence, b.asymmetry_rho, b.morphology_class, b.duration_class,
          b.intensity_class, b.reliability, b.reduced_chi2, b.ml_score, b.decision,
          b.audit_status || 'auto', b.audit_notes || ''
        ]);
      } else {
        this.localStore.burst_catalog[id] = record;
      }
    }
    if (!this.isPg) this._saveLocal();
  }

  async getBursts(jobId = null) {
    if (this.isPg) {
      const q = jobId ? 'SELECT * FROM burst_catalog WHERE job_id = $1 ORDER BY peak_time_met ASC' : 'SELECT * FROM burst_catalog ORDER BY created_at DESC';
      const values = jobId ? [jobId] : [];
      const res = await this.pgPool.query(q, values);
      return res.rows;
    } else {
      let list = Object.values(this.localStore.burst_catalog);
      if (jobId) list = list.filter(b => b.job_id === jobId);
      return list.sort((a, b) => a.peak_time_met - b.peak_time_met);
    }
  }

  async updateBurstAudit(burstId, { decision, audit_status, audit_notes }) {
    if (this.isPg) {
      const q = `
        UPDATE burst_catalog 
        SET decision = COALESCE($2, decision),
            audit_status = COALESCE($3, audit_status),
            audit_notes = COALESCE($4, audit_notes)
        WHERE id = $1 RETURNING *;
      `;
      const res = await this.pgPool.query(q, [burstId, decision, audit_status, audit_notes]);
      return res.rows[0];
    } else {
      const b = this.localStore.burst_catalog[burstId];
      if (!b) return null;
      if (decision) b.decision = decision;
      if (audit_status) b.audit_status = audit_status;
      if (audit_notes) b.audit_notes = audit_notes;
      this._saveLocal();
      return b;
    }
  }
}

export const db = new DatabaseManager();
