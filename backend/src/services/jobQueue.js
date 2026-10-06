/**
 * Background Job Queue
 * Asynchronous job management with status tracking, progress callbacks, and cancellation.
 */

import { v4 as uuidv4 } from 'uuid';
import { db } from '../db/index.js';
import { pythonBridge } from './pythonBridge.js';

class JobQueue {
  constructor() {
    this.activeJobs = new Map(); // jobId -> abortController or worker status
  }

  async createAnalysisJob(dataset, configOptions) {
    const jobId = `job_${uuidv4().slice(0, 8)}`;
    
    // Check cache: if dataset has matching SHA-256 and config, return cached or run
    const job = await db.createJob({
      id: jobId,
      dataset_id: dataset.id,
      status: 'queued',
      progress: 0,
      message: 'Job queued for scientific analysis...',
      config: configOptions
    });

    // Start asynchronously without blocking API response
    this._executeJob(jobId, dataset, configOptions);

    return job;
  }

  async _executeJob(jobId, dataset, configOptions) {
    try {
      this.activeJobs.set(jobId, { status: 'running' });

      await db.updateJob(jobId, {
        status: 'running',
        progress: 10,
        message: 'Initializing scientific engine...'
      });

      const onProgress = async (pct, msg) => {
        if (!this.activeJobs.has(jobId)) return; // cancelled
        await db.updateJob(jobId, { progress: pct, message: msg });
      };

      const result = await pythonBridge.runAnalysis(dataset.filepath, configOptions, onProgress);

      if (!this.activeJobs.has(jobId)) {
        console.log(`[JobQueue] Job ${jobId} was cancelled before completion.`);
        return;
      }

      // Save bursts to catalog table
      if (result.bursts && result.bursts.length > 0) {
        await db.saveBursts(jobId, result.bursts);
      }

      // Update dataset metadata if not set
      if (result.coverage) {
        await db.saveDataset({
          ...dataset,
          total_points: result.coverage.total_points,
          valid_points: result.coverage.valid_points,
          span_s: result.coverage.span_s,
        });
      }

      await db.updateJob(jobId, {
        status: 'completed',
        progress: 100,
        message: `Analysis completed: ${result.total_bursts} solar bursts identified.`,
        result_json: result,
        completed_at: new Date().toISOString()
      });

      this.activeJobs.delete(jobId);
    } catch (err) {
      console.error(`[JobQueue] Error executing job ${jobId}:`, err);
      await db.updateJob(jobId, {
        status: 'failed',
        progress: 0,
        message: 'Analysis failed.',
        error_message: err.message,
        completed_at: new Date().toISOString()
      });
      this.activeJobs.delete(jobId);
    }
  }

  async cancelJob(jobId) {
    if (this.activeJobs.has(jobId)) {
      this.activeJobs.delete(jobId);
      await db.updateJob(jobId, {
        status: 'cancelled',
        message: 'Job was cancelled by user.',
        completed_at: new Date().toISOString()
      });
      return true;
    }
    return false;
  }
}

export const jobQueue = new JobQueue();
