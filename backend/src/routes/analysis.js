import express from 'express';
import { db } from '../db/index.js';
import { jobQueue } from '../services/jobQueue.js';

const router = express.Router();

// Start new analysis job
router.post('/start', async (req, res) => {
  try {
    const { dataset_id, config } = req.body;
    if (!dataset_id) {
      return res.status(400).json({ error: 'dataset_id is required' });
    }

    const dataset = await db.getDataset(dataset_id);
    if (!dataset) {
      return res.status(404).json({ error: 'Dataset not found' });
    }

    const job = await jobQueue.createAnalysisJob(dataset, config || {});
    res.json({
      status: 'queued',
      jobId: job.id,
      job
    });
  } catch (err) {
    console.error('[Analysis Start Error]', err);
    res.status(500).json({ error: err.message });
  }
});

// List recent jobs
router.get('/jobs', async (req, res) => {
  try {
    // If PG, query jobs table; if local store, query localStore.jobs
    if (db.isPg) {
      const q = 'SELECT * FROM jobs ORDER BY created_at DESC LIMIT 50';
      const results = await db.pgPool.query(q);
      res.json(results.rows);
    } else {
      const jobs = Object.values(db.localStore.jobs).sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
      res.json(jobs.slice(0, 50));
    }
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Get job status and result
router.get('/jobs/:id', async (req, res) => {
  try {
    const job = await db.getJob(req.params.id);
    if (!job) return res.status(404).json({ error: 'Job not found' });
    res.json(job);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Cancel running job
router.post('/jobs/:id/cancel', async (req, res) => {
  try {
    const cancelled = await jobQueue.cancelJob(req.params.id);
    res.json({ status: cancelled ? 'cancelled' : 'job_not_active' });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

export default router;
