import express from 'express';
import fs from 'fs';
import path from 'path';
import { config } from '../config.js';
import { pythonBridge } from '../services/pythonBridge.js';

const router = express.Router();

// List available models and metadata cards
router.get('/', (req, res) => {
  try {
    const dir = path.join(config.modelsDir, 'pretrained');
    if (!fs.existsSync(dir)) {
      return res.json([]);
    }

    const files = fs.readdirSync(dir);
    const cardFiles = files.filter(f => f.endsWith('_card.json'));
    
    const models = cardFiles.map(cardFile => {
      const cardPath = path.join(dir, cardFile);
      const cardData = JSON.parse(fs.readFileSync(cardPath, 'utf-8'));
      const modelFile = cardFile.replace('_card.json', '.joblib');
      const modelPath = path.join(dir, modelFile);
      const exists = fs.existsSync(modelPath);

      return {
        ...cardData,
        model_file: modelFile,
        exists,
        size_bytes: exists ? fs.statSync(modelPath).size : 0
      };
    });

    res.json(models);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Train or retrain models
router.post('/train', async (req, res) => {
  try {
    const nSamples = parseInt(req.body.n_samples) || 150;
    const trainResult = await pythonBridge.trainModels(nSamples);
    res.json(trainResult);
  } catch (err) {
    console.error('[Model Train Error]', err);
    res.status(500).json({ error: err.message });
  }
});

export default router;
