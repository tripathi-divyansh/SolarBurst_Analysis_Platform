import React, { useState, useEffect } from 'react';
import { BrainCircuit, Play, CheckCircle2, ShieldCheck, FileText, RefreshCw, AlertCircle } from 'lucide-react';
import { api } from '../api';

const FEATURE_SCHEMA = [
  { name: 'peak_snr', group: 'Signal Strength', unit: 'Dimensionless', desc: 'Net peak divided by robust local MAD noise estimate' },
  { name: 'net_peak', group: 'Signal Strength', unit: 'count / s', desc: 'Observed maximum count rate minus fitted background' },
  { name: 'integrated_excess', group: 'Signal Strength', unit: 'Counts', desc: 'Total background-subtracted excess counts in candidate interval' },
  { name: 'prelim_duration_s', group: 'Shape', unit: 'Seconds', desc: 'Candidate proposal interval duration' },
  { name: 'prelim_width_s', group: 'Shape', unit: 'Seconds', desc: 'Preliminary FWHM width estimate' },
  { name: 'rise_slope', group: 'Shape', unit: 'c / s²', desc: 'Net peak divided by rise duration' },
  { name: 'decay_slope', group: 'Shape', unit: 'c / s²', desc: 'Net peak divided by decay duration' },
  { name: 'rise_decay_ratio', group: 'Shape', unit: 'Ratio', desc: 'Ratio of rise duration to decay duration' },
  { name: 'mf_max_score', group: 'Multiscale', unit: 'SNR', desc: 'Maximum score across matched filter template bank' },
  { name: 'mf_best_rise_s', group: 'Multiscale', unit: 'Seconds', desc: 'Best matching template rise timescale' },
  { name: 'mf_best_decay_s', group: 'Multiscale', unit: 'Seconds', desc: 'Best matching template decay timescale' },
  { name: 'cwt_max_score', group: 'Multiscale', unit: 'Wavelet SNR', desc: 'Maximum Mexican-hat CWT ridge response' },
  { name: 'cwt_best_scale_s', group: 'Multiscale', unit: 'Seconds', desc: 'Timescale yielding highest wavelet coefficient' },
  { name: 'local_noise', group: 'Background', unit: 'count / s', desc: 'Robust difference-series MAD noise estimate' },
  { name: 'local_baseline_level', group: 'Background', unit: 'count / s', desc: 'AsLS estimated background rate at peak time' },
  { name: 'local_baseline_slope', group: 'Background', unit: 'c / s²', desc: 'Linear slope of background across event context' },
  { name: 'pre_post_level_diff', group: 'Background', unit: 'SNR', desc: 'Level difference between pre-event and post-event background' },
  { name: 'num_peaks', group: 'Complexity', unit: 'Count', desc: 'Number of distinct significant peaks in candidate core' },
  { name: 'exposure_fraction', group: 'Quality', unit: 'Fraction [0,1]', desc: 'Fraction of expected bins with valid telemetry exposure' },
  { name: 'gap_fraction', group: 'Quality', unit: 'Fraction [0,1]', desc: 'Fraction of missing or flagged invalid telemetry' },
  { name: 'edge_distance_s', group: 'Quality', unit: 'Seconds', desc: 'Distance to observation file boundary or major telemetry gap' },
  { name: 'cadence_s', group: 'Sampling', unit: 'Seconds', desc: 'Median observation bin width' },
];

export default function ModelTraining() {
  const [models, setModels] = useState([]);
  const [isTraining, setIsTraining] = useState(false);
  const [trainStatus, setTrainStatus] = useState(null);

  const fetchModels = async () => {
    try {
      const data = await api.listModels();
      setModels(data);
    } catch (err) {
      console.error(err);
    }
  };

  useEffect(() => {
    fetchModels();
  }, []);

  const handleRetrain = async () => {
    setIsTraining(true);
    setTrainStatus(null);
    try {
      const res = await api.trainModels(150);
      setTrainStatus(res);
      await fetchModels();
    } catch (err) {
      alert('Training failed: ' + err.message);
    } finally {
      setIsTraining(false);
    }
  };

  return (
    <div className="page-container" style={{ maxWidth: '1050px', margin: '0 auto' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
        <div>
          <h1 style={{ fontSize: '24px', marginBottom: '4px' }}>Model Training & Evaluation</h1>
          <p style={{ color: 'var(--text-muted)', margin: 0, fontSize: '13px' }}>
            Random Forest baseline and XGBoost challenger models evaluated on identical feature representations.
          </p>
        </div>

        <button className="btn btn-primary" onClick={handleRetrain} disabled={isTraining}>
          {isTraining ? <RefreshCw size={15} className="animate-spin" /> : <Play size={15} />}
          <span>{isTraining ? 'Training Models...' : 'Retrain Pipeline Models'}</span>
        </button>
      </div>

      {trainStatus && (
        <div style={{ background: 'rgba(34, 197, 94, 0.1)', border: '1px solid rgba(34, 197, 94, 0.3)', padding: '14px', borderRadius: '8px', marginBottom: '20px', fontSize: '13px', color: '#86efac' }}>
          <strong>Training Succeeded!</strong> Trained on {trainStatus.n_train_samples} samples ({trainStatus.train_positives} bursts, {trainStatus.train_negatives} negatives). Validated on {trainStatus.n_val_samples} held-out samples.
        </div>
      )}

      {/* Model Cards Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', marginBottom: '24px' }}>
        {models.map((m) => (
          <div key={m.model_name} className="card">
            <div className="card-title">
              <span className="mono font-bold">{m.model_name}</span>
              <span className="badge badge-cyan">{m.model_type?.toUpperCase()}</span>
            </div>

            <div style={{ fontSize: '12.5px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
              <div><span className="text-muted">Intended Use:</span> {m.intended_use}</div>
              <div><span className="text-muted">SHA-256 Checksum:</span> <span className="mono" style={{ fontSize: '11px' }}>{m.file_sha256}</span></div>
              <div><span className="text-muted">Accept Threshold:</span> <span className="mono font-bold text-green">{m.accept_threshold}</span></div>
              <div><span className="text-muted">Review Threshold:</span> <span className="mono font-bold text-amber">{m.review_threshold}</span></div>
              <div><span className="text-muted">Limitations:</span> {m.limitations}</div>
            </div>
          </div>
        ))}
      </div>

      {/* Standardized Feature Schema Table */}
      <div className="card">
        <div className="card-title">
          <span>Standardized 24-Feature Schema (v1.0)</span>
          <span className="text-muted" style={{ fontSize: '12px' }}>Identical order enforced during training and inference</span>
        </div>

        <div className="table-container">
          <table className="scientific-table">
            <thead>
              <tr>
                <th>Feature Column</th>
                <th>Group</th>
                <th>Physical Units</th>
                <th>Scientific Description & Handling</th>
              </tr>
            </thead>
            <tbody>
              {FEATURE_SCHEMA.map((f) => (
                <tr key={f.name}>
                  <td className="mono font-bold text-cyan">{f.name}</td>
                  <td><span className="badge badge-muted">{f.group}</span></td>
                  <td className="mono">{f.unit}</td>
                  <td style={{ fontSize: '12px', color: 'var(--text-muted)' }}>{f.desc}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
