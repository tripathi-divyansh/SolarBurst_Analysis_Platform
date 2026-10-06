import express from 'express';
import multer from 'multer';
import path from 'path';
import fs from 'fs';
import { v4 as uuidv4 } from 'uuid';
import { db } from '../db/index.js';
import { config } from '../config.js';
import { pythonBridge } from '../services/pythonBridge.js';

const router = express.Router();

// Ensure upload directory exists
if (!fs.existsSync(config.uploadDir)) {
  fs.mkdirSync(config.uploadDir, { recursive: true });
}

const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, config.uploadDir),
  filename: (req, file, cb) => {
    const ext = path.extname(file.originalname);
    cb(null, `${Date.now()}_${uuidv4().slice(0, 6)}${ext}`);
  }
});

const upload = multer({
  storage,
  limits: { fileSize: 250 * 1024 * 1024 } // 250MB limit
});

// Upload and inspect dataset
router.post('/upload', upload.single('file'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No file uploaded' });
    }

    const filepath = req.file.path;
    const inspectResult = await pythonBridge.inspectFile(filepath);

    const datasetId = `ds_${uuidv4().slice(0, 8)}`;
    const format = inspectResult.detection?.format || path.extname(req.file.originalname).replace('.', '') || 'unknown';

    const dataset = await db.saveDataset({
      id: datasetId,
      name: req.body.name || req.file.originalname,
      filename: req.file.originalname,
      filepath,
      format,
      size_bytes: req.file.size,
      sha256: inspectResult.sha256 || 'unknown',
      instrument: 'XSM',
      metadata: inspectResult
    });

    res.json({
      status: 'success',
      dataset,
      inspection: inspectResult
    });
  } catch (err) {
    console.error('[Datasets Route Error]', err);
    res.status(500).json({ error: err.message });
  }
});

// List all datasets
router.get('/', async (req, res) => {
  try {
    const list = await db.listDatasets();
    res.json(list);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Get specific dataset
router.get('/:id', async (req, res) => {
  try {
    const ds = await db.getDataset(req.params.id);
    if (!ds) return res.status(404).json({ error: 'Dataset not found' });
    res.json(ds);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Re-inspect with optional custom parameters
router.post('/:id/inspect', async (req, res) => {
  try {
    const ds = await db.getDataset(req.params.id);
    if (!ds) return res.status(404).json({ error: 'Dataset not found' });
    const inspection = await pythonBridge.inspectFile(ds.filepath);
    res.json(inspection);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

export default router;
