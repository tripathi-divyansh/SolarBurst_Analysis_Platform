import express from 'express';
import cors from 'cors';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';

import { config } from './src/config.js';
import { db } from './src/db/index.js';
import { pythonBridge } from './src/services/pythonBridge.js';

import datasetsRouter from './src/routes/datasets.js';
import analysisRouter from './src/routes/analysis.js';
import catalogRouter from './src/routes/catalog.js';
import waveletRouter from './src/routes/wavelet.js';
import modelsRouter from './src/routes/models.js';
import exportRouter from './src/routes/export.js';
import demoRouter from './src/routes/demo.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();

app.use(cors());
app.use(express.json({ limit: '100mb' }));
app.use(express.urlencoded({ extended: true, limit: '100mb' }));

// Health / Status endpoint
app.get('/api/health', async (req, res) => {
  const pyHealthy = await pythonBridge.isServiceHealthy();
  res.json({
    status: 'online',
    backend: 'Node.js Express',
    db_mode: db.isPg ? 'PostgreSQL' : 'Local Offline Store',
    python_engine: pyHealthy ? 'Connected (HTTP Microservice)' : 'Available (CLI Subprocess Fallback)',
    version: '1.0.0'
  });
});

// API Routes
app.use('/api/datasets', datasetsRouter);
app.use('/api/analysis', analysisRouter);
app.use('/api/catalog', catalogRouter);
app.use('/api/wavelet', waveletRouter);
app.use('/api/models', modelsRouter);
app.use('/api/export', exportRouter);
app.use('/api/demo', demoRouter);

// Serve frontend if built (Desktop / Standalone mode)
const distPath = path.resolve(__dirname, '../frontend/dist');
if (fs.existsSync(distPath)) {
  app.use(express.static(distPath));
  app.get('*', (req, res) => {
    if (!req.path.startsWith('/api/')) {
      res.sendFile(path.join(distPath, 'index.html'));
    }
  });
}

// Start Server
async function startServer() {
  await db.init();
  // Ensure Python service is checked
  pythonBridge.ensureServiceRunning().catch(e => console.warn('[Server] Python startup note:', e.message));

  app.listen(config.port, () => {
    console.log(`=======================================================`);
    console.log(`  SolarBurst — XSM Burst Analysis Platform API`);
    console.log(`  Server running on http://127.0.0.1:${config.port}`);
    console.log(`  Database mode: ${db.isPg ? 'PostgreSQL' : 'Local Offline Zero-Config'}`);
    console.log(`=======================================================`);
  });
}

startServer();
