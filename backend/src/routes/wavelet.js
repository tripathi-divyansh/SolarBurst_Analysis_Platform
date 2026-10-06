import express from 'express';
import { db } from '../db/index.js';
import { pythonBridge } from '../services/pythonBridge.js';

const router = express.Router();

router.post('/', async (req, res) => {
  try {
    const { dataset_id, start_met, end_met, wavelet } = req.body;
    if (!dataset_id) {
      return res.status(400).json({ error: 'dataset_id is required' });
    }

    const dataset = await db.getDataset(dataset_id);
    if (!dataset) {
      return res.status(404).json({ error: 'Dataset not found' });
    }

    const scalogram = await pythonBridge.computeWavelet(
      dataset.filepath,
      start_met || null,
      end_met || null,
      wavelet || 'morlet'
    );

    res.json(scalogram);
  } catch (err) {
    console.error('[Wavelet Route Error]', err);
    res.status(500).json({ error: err.message });
  }
});

export default router;
