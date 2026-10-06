import path from 'path';
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.resolve(__dirname, '../../');

export const config = {
  rootDir,
  srcDir: path.join(rootDir, 'src'),
  port: process.env.PORT || 5000,
  pythonServiceUrl: process.env.PYTHON_SERVICE_URL || 'http://127.0.0.1:8000',
  databaseUrl: process.env.DATABASE_URL || null,
  uploadDir: path.join(rootDir, 'backend/uploads'),
  dataDir: path.join(rootDir, 'data'),
  modelsDir: path.join(rootDir, 'models'),
  configsDir: path.join(rootDir, 'configs'),
  resultsDir: path.join(rootDir, 'results'),
  pythonExecutable: process.env.PYTHON_EXEC || 'python',
};
