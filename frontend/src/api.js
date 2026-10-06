/**
 * SolarBurst Frontend API Client
 * Clean interface to Node.js Express backend.
 */

const API_BASE = '/api';

export const api = {
  // System Health
  async getHealth() {
    const res = await fetch(`${API_BASE}/health`);
    return res.json();
  },

  // Datasets
  async listDatasets() {
    const res = await fetch(`${API_BASE}/datasets`);
    return res.json();
  },

  async getDataset(id) {
    const res = await fetch(`${API_BASE}/datasets/${id}`);
    return res.json();
  },

  async uploadDataset(formData) {
    const res = await fetch(`${API_BASE}/datasets/upload`, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.error || 'Upload failed');
    }
    return res.json();
  },

  async inspectDataset(id) {
    const res = await fetch(`${API_BASE}/datasets/${id}/inspect`, { method: 'POST' });
    return res.json();
  },

  // Analysis & Jobs
  async startAnalysis(dataset_id, config) {
    const res = await fetch(`${API_BASE}/analysis/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ dataset_id, config }),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.error || 'Failed to start analysis');
    }
    return res.json();
  },

  async listJobs() {
    const res = await fetch(`${API_BASE}/analysis/jobs`);
    return res.json();
  },

  async getJob(id) {
    const res = await fetch(`${API_BASE}/analysis/jobs/${id}`);
    return res.json();
  },

  async cancelJob(id) {
    const res = await fetch(`${API_BASE}/analysis/jobs/${id}/cancel`, { method: 'POST' });
    return res.json();
  },

  // Burst Catalog & Audit
  async getCatalog(params = {}) {
    const q = new URLSearchParams(params).toString();
    const res = await fetch(`${API_BASE}/catalog?${q}`);
    return res.json();
  },

  async updateAudit(burstId, { decision, audit_status, audit_notes }) {
    const res = await fetch(`${API_BASE}/catalog/${burstId}/audit`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ decision, audit_status, audit_notes }),
    });
    return res.json();
  },

  // Wavelet Scalogram
  async computeWavelet(params) {
    const res = await fetch(`${API_BASE}/wavelet`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.error || 'Failed to compute wavelet');
    }
    return res.json();
  },

  // Models
  async listModels() {
    const res = await fetch(`${API_BASE}/models`);
    return res.json();
  },

  async trainModels(nSamples = 150) {
    const res = await fetch(`${API_BASE}/models/train`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ n_samples: nSamples }),
    });
    return res.json();
  },

  // Synthetic Demonstration
  async loadDemo() {
    const res = await fetch(`${API_BASE}/demo/load`, { method: 'POST' });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.error || 'Failed to load demo');
    }
    return res.json();
  },

  // Download URL builders
  getCsvExportUrl(jobId) {
    return `${API_BASE}/export/${jobId}/csv`;
  },
  getJsonExportUrl(jobId) {
    return `${API_BASE}/export/${jobId}/json`;
  },
  getReportExportUrl(jobId) {
    return `${API_BASE}/export/${jobId}/report`;
  },
};
