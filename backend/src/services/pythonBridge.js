/**
 * Python Engine Bridge
 * Communicates with the Python scientific analysis engine via HTTP microservice
 * or automatic child-process CLI fallback.
 */

import { spawn } from 'child_process';
import { config } from '../config.js';

class PythonBridge {
  constructor() {
    this.serviceUrl = config.pythonServiceUrl;
    this.childProcess = null;
  }

  async isServiceHealthy() {
    try {
      const resp = await fetch(`${this.serviceUrl}/health`, { signal: AbortSignal.timeout(2000) });
      if (resp.ok) {
        const data = await resp.json();
        return data.status === 'healthy';
      }
      return false;
    } catch {
      return false;
    }
  }

  async ensureServiceRunning() {
    const healthy = await this.isServiceHealthy();
    if (healthy) return true;

    console.log('[PythonBridge] Starting Python microservice on port 8000...');
    // Start Python microservice as a child process
    const py = spawn(config.pythonExecutable, ['-m', 'uvicorn', 'solarburst.service:api_app', '--host', '127.0.0.1', '--port', '8000'], {
      cwd: config.rootDir,
      env: { ...process.env, PYTHONPATH: config.srcDir }
    });

    py.stdout.on('data', (d) => console.log(`[Python] ${d.toString().trim()}`));
    py.stderr.on('data', (d) => console.log(`[Python] ${d.toString().trim()}`));

    this.childProcess = py;

    // Wait up to 6 seconds for server to be healthy
    for (let i = 0; i < 12; i++) {
      await new Promise(r => setTimeout(r, 500));
      if (await this.isServiceHealthy()) {
        console.log('[PythonBridge] Python scientific service is live and ready.');
        return true;
      }
    }
    console.warn('[PythonBridge] Could not verify Python service via HTTP; will attempt direct CLI fallback.');
    return false;
  }

  async inspectFile(filepath) {
    await this.ensureServiceRunning();
    try {
      const res = await fetch(`${this.serviceUrl}/inspect`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filepath })
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Inspection failed');
      }
      return await res.json();
    } catch (e) {
      // Direct Python CLI fallback
      return this._runPythonCommand('inspect', { filepath });
    }
  }

  async runAnalysis(filepath, options = {}, onProgress = null) {
    await this.ensureServiceRunning();
    try {
      if (onProgress) onProgress(15, 'Sending observation to scientific engine...');
      const res = await fetch(`${this.serviceUrl}/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          filepath,
          preset: options.preset || 'balanced',
          model_family: options.model_family || 'random_forest',
          asls_lambda: options.asls_lambda || 1e6,
          asls_p: options.asls_p || 0.01,
          accept_threshold: options.accept_threshold || 0.65,
          review_threshold: options.review_threshold || 0.35,
          max_components: options.max_components || 3,
          fast_uncertainty: options.fast_uncertainty !== false,
          column_mapping: options.column_mapping || null,
        })
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Analysis failed');
      }

      if (onProgress) onProgress(100, 'Analysis completed.');
      return await res.json();
    } catch (e) {
      console.warn(`[PythonBridge] HTTP analyze failed (${e.message}). Falling back to CLI execution.`);
      return this._runPythonCLIAnalyze(filepath, options, onProgress);
    }
  }

  async computeWavelet(filepath, start_met = null, end_met = null, wavelet = 'morlet') {
    await this.ensureServiceRunning();
    const res = await fetch(`${this.serviceUrl}/wavelet`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ filepath, start_met, end_met, wavelet })
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Wavelet computation failed');
    }
    return await res.json();
  }

  async runDemo() {
    await this.ensureServiceRunning();
    const res = await fetch(`${this.serviceUrl}/demo`, { method: 'POST' });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Demo generation failed');
    }
    return await res.json();
  }

  async trainModels(nSamples = 150) {
    await this.ensureServiceRunning();
    const res = await fetch(`${this.serviceUrl}/train`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ n_synthetic_samples: nSamples, output_dir: config.modelsDir + '/pretrained' })
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Model training failed');
    }
    return await res.json();
  }

  _runPythonCLIAnalyze(filepath, options, onProgress) {
    return new Promise((resolve, reject) => {
      if (onProgress) onProgress(20, 'Spawning Python CLI engine...');
      const normalizedPath = filepath.replace(/\\/g, '/');
      const normalizedSrc = config.srcDir.replace(/\\/g, '/');
      const script = `
import sys, json
sys.path.insert(0, r"${normalizedSrc}")
from solarburst.io.xsm_adapter import load_lightcurve
from solarburst.engine import SolarBurstEngine
from solarburst.core.schema import AnalysisConfig

lc = load_lightcurve(r"${normalizedPath}")
cfg = AnalysisConfig(name="${options.preset || 'balanced'}", model_family="${options.model_family || 'random_forest'}")
engine = SolarBurstEngine()
res = engine.run_analysis(lc, cfg)
print("___SOLARBURST_RESULT_START___")
print(json.dumps(res))
print("___SOLARBURST_RESULT_END___")
`;
      const py = spawn(config.pythonExecutable, ['-c', script], {
        cwd: config.rootDir,
        env: { ...process.env, PYTHONPATH: config.srcDir }
      });

      let stdout = '';
      let stderr = '';
      py.stdout.on('data', d => stdout += d.toString());
      py.stderr.on('data', d => stderr += d.toString());

      py.on('close', code => {
        if (code !== 0) {
          return reject(new Error(`Python process exited with code ${code}: ${stderr}`));
        }
        try {
          const match = stdout.match(/___SOLARBURST_RESULT_START___([\s\S]*?)___SOLARBURST_RESULT_END___/);
          if (match && match[1]) {
            const data = JSON.parse(match[1].trim());
            resolve(data);
          } else {
            reject(new Error(`Failed to parse Python result: ${stdout.slice(0, 500)}`));
          }
        } catch (err) {
          reject(err);
        }
      });
    });
  }

  _runPythonCommand(cmd, payload) {
    return new Promise((resolve, reject) => {
      const normalizedPath = payload.filepath.replace(/\\/g, '/');
      const normalizedSrc = config.srcDir.replace(/\\/g, '/');
      const script = `
import sys, json
sys.path.insert(0, r"${normalizedSrc}")
from solarburst.io.xsm_adapter import inspect_file
res = inspect_file(r"${normalizedPath}")
print(json.dumps(res))
`;
      const py = spawn(config.pythonExecutable, ['-c', script], {
        cwd: config.rootDir,
        env: { ...process.env, PYTHONPATH: config.srcDir }
      });
      let out = '';
      let errStr = '';
      py.stdout.on('data', d => out += d.toString());
      py.stderr.on('data', d => errStr += d.toString());
      py.on('close', code => {
        if (code === 0) {
          try { resolve(JSON.parse(out)); } catch (e) { reject(e); }
        } else {
          reject(new Error(`Inspect command failed with code ${code}: ${errStr}`));
        }
      });
    });
  }
}

export const pythonBridge = new PythonBridge();
