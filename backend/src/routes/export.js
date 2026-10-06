import express from 'express';
import fs from 'fs';
import path from 'path';
import { db } from '../db/index.js';
import { generateHtmlReport } from '../services/reportGenerator.js';

const router = express.Router();

// Download HTML Analysis Report
router.get('/:jobId/report', async (req, res) => {
  try {
    const job = await db.getJob(req.params.jobId);
    if (!job || !job.result_json) {
      return res.status(404).json({ error: 'Job or results not found' });
    }

    const dataset = await db.getDataset(job.dataset_id) || { name: 'Observation' };
    const html = generateHtmlReport({
      dataset,
      job,
      results: job.result_json
    });

    res.setHeader('Content-Type', 'text/html');
    res.setHeader('Content-Disposition', `attachment; filename="solarburst_report_${req.params.jobId}.html"`);
    res.send(html);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Download CSV Catalog
router.get('/:jobId/csv', async (req, res) => {
  try {
    const bursts = await db.getBursts(req.params.jobId);
    if (!bursts || bursts.length === 0) {
      return res.status(404).json({ error: 'No bursts found for this job' });
    }

    const headers = [
      'burst_id', 'peak_time_met', 'peak_time_iso', 'start_time_iso', 'end_time_iso',
      'duration_s', 'net_peak', 'total_peak', 'peak_snr', 'fluence', 'asymmetry_rho',
      'morphology_class', 'duration_class', 'intensity_class', 'reliability',
      'reduced_chi2', 'ml_score', 'decision', 'audit_status', 'audit_notes'
    ];

    const csvRows = [headers.join(',')];
    for (const b of bursts) {
      const row = headers.map(h => {
        const val = b[h];
        if (val === null || val === undefined) return '';
        if (typeof val === 'string' && val.includes(',')) return `"${val}"`;
        return val;
      });
      csvRows.push(row.join(','));
    }

    res.setHeader('Content-Type', 'text/csv');
    res.setHeader('Content-Disposition', `attachment; filename="solarburst_catalog_${req.params.jobId}.csv"`);
    res.send(csvRows.join('\n'));
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Download Provenance JSON
router.get('/:jobId/json', async (req, res) => {
  try {
    const job = await db.getJob(req.params.jobId);
    if (!job || !job.result_json) {
      return res.status(404).json({ error: 'Job or results not found' });
    }
    const dataset = await db.getDataset(job.dataset_id) || {};

    const manifest = {
      platform: 'SolarBurst — XSM Burst Analysis Platform',
      version: '1.0.0',
      job_id: job.id,
      created_at: job.created_at,
      dataset: {
        id: dataset.id,
        filename: dataset.filename,
        sha256: dataset.sha256,
        instrument: dataset.instrument,
        quantity: dataset.quantity,
        unit: dataset.unit
      },
      configuration: job.config_json,
      summary: job.result_json.summary,
      bursts: job.result_json.bursts
    };

    res.setHeader('Content-Type', 'application/json');
    res.setHeader('Content-Disposition', `attachment; filename="solarburst_provenance_${req.params.jobId}.json"`);
    res.json(manifest);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

export default router;
