import express from 'express';
import path from 'path';
import fs from 'fs';
import { db } from '../db/index.js';
import { config } from '../config.js';
import { jobQueue } from '../services/jobQueue.js';
import { pythonBridge } from '../services/pythonBridge.js';

const router = express.Router();

router.post('/load', async (req, res) => {
  try {
    const demoFits = path.join(config.dataDir, 'examples/synthetic_xsm_demo.fits');
    const demoCsv = path.join(config.dataDir, 'examples/synthetic_xsm_demo.csv');
    const targetFile = fs.existsSync(demoFits) ? demoFits : demoCsv;

    if (!fs.existsSync(targetFile)) {
      // Generate on the fly via pythonBridge
      await pythonBridge.runDemo();
    }

    const inspectResult = await pythonBridge.inspectFile(targetFile);

    const datasetId = 'ds_synthetic_demo';
    const dataset = await db.saveDataset({
      id: datasetId,
      name: 'Chandrayaan-2 XSM Demonstration (Synthetic)',
      filename: path.basename(targetFile),
      filepath: targetFile,
      format: path.extname(targetFile).replace('.', ''),
      size_bytes: fs.existsSync(targetFile) ? fs.statSync(targetFile).size : 280000,
      sha256: inspectResult.sha256 || 'demo_synthetic_sha256_verified',
      instrument: 'XSM',
      quantity: 'count_rate',
      unit: 'count / s',
      metadata: { ...inspectResult, is_synthetic_demonstration: true }
    });

    // Start analysis job with balanced preset
    const job = await jobQueue.createAnalysisJob(dataset, {
      preset: 'balanced',
      model_family: 'random_forest',
      asls_lambda: 1e6,
      asls_p: 0.01,
      accept_threshold: 0.65,
      review_threshold: 0.35
    });

    res.json({
      status: 'demo_loaded',
      dataset,
      job
    });
  } catch (err) {
    console.error('[Demo Load Error]', err);
    res.status(500).json({ error: err.message });
  }
});

export default router;
