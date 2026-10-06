import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Activity, Sun, Layers, Clock, ShieldCheck, ArrowRight, Upload, Play, AlertCircle } from 'lucide-react';
import MetricCard from '../components/MetricCard';

export default function Dashboard({ activeDataset, activeJob, onLoadDemo, isDemoLoading }) {
  const navigate = useNavigate();
  const results = activeJob?.result_json;
  const summary = results?.summary || {};
  const coverage = results?.coverage || {};
  const bursts = results?.bursts || [];

  return (
    <div className="page-container">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
        <div>
          <h1 style={{ fontSize: '24px', marginBottom: '4px' }}>Observatory Dashboard</h1>
          <p style={{ color: 'var(--text-muted)', margin: 0, fontSize: '13px' }}>
            Chandrayaan-2 Solar X-ray Monitor (XSM) Burst Analysis & Pipeline Monitoring
          </p>
        </div>

        <div style={{ display: 'flex', gap: '10px' }}>
          <button className="btn btn-secondary" onClick={() => navigate('/upload')}>
            <Upload size={15} />
            <span>Upload Observation</span>
          </button>
          <button className="btn btn-primary" onClick={onLoadDemo} disabled={isDemoLoading}>
            <Play size={15} />
            <span>{isDemoLoading ? 'Loading Demo...' : 'Run Synthetic Demo'}</span>
          </button>
        </div>
      </div>

      {activeDataset?.metadata?.is_synthetic_demonstration && (
        <div style={{ background: 'rgba(245, 158, 11, 0.1)', border: '1px solid rgba(245, 158, 11, 0.3)', padding: '12px 16px', borderRadius: '8px', marginBottom: '20px', display: 'flex', alignItems: 'center', gap: '10px' }}>
          <AlertCircle size={18} className="text-amber" />
          <span style={{ fontSize: '13px', color: 'var(--callout-amber-text)' }}>
            <strong>Active Demonstration:</strong> Using verified synthetic Chandrayaan-2 XSM light curve with noise-correct Poisson injections.
          </span>
        </div>
      )}

      {/* Metrics Row */}
      <div className="metrics-grid">
        <MetricCard
          label="Valid Exposure"
          value={coverage.valid_exposure_s ? `${(coverage.valid_exposure_s / 60).toFixed(1)} m` : '—'}
          sub={`${coverage.valid_points || 0} valid cadence bins`}
          icon={Clock}
          color="cyan"
        />
        <MetricCard
          label="Identified Bursts"
          value={results ? bursts.length : '—'}
          sub={`${summary.accepted_count || 0} accepted, ${summary.review_count || 0} review`}
          icon={Sun}
          color="green"
        />
        <MetricCard
          label="Candidate Proposals"
          value={results ? results.total_candidates : '—'}
          sub="Multiscale union proposals"
          icon={Layers}
          color="amber"
        />
        <MetricCard
          label="Reliable Fits"
          value={results ? summary.reliable_count : '—'}
          sub={`${summary.poor_fit_count || 0} flagged poor fit`}
          icon={ShieldCheck}
          color="cyan"
        />
      </div>

      {/* Two Column Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: '20px' }}>
        {/* Active Dataset Overview */}
        <div className="card">
          <div className="card-title">
            <span>Observation Summary</span>
            {activeDataset && <span className="badge badge-cyan">{activeDataset.format.toUpperCase()}</span>}
          </div>

          {activeDataset ? (
            <div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', marginBottom: '16px', fontSize: '13px' }}>
                <div><span className="text-muted">Filename:</span> <span className="mono">{activeDataset.filename}</span></div>
                <div><span className="text-muted">Instrument:</span> <span className="mono">Chandrayaan-2 XSM</span></div>
                <div><span className="text-muted">Quantity:</span> <span className="mono">{activeDataset.quantity || 'count_rate'} ({activeDataset.unit || 'count / s'})</span></div>
                <div><span className="text-muted">SHA-256:</span> <span className="mono" style={{ fontSize: '11px' }}>{activeDataset.sha256?.slice(0, 16)}...</span></div>
                <div><span className="text-muted">Span:</span> <span className="mono">{coverage.span_s ? `${(coverage.span_s / 3600).toFixed(2)} hours` : '—'}</span></div>
                <div><span className="text-muted">Gap Fraction:</span> <span className="mono">{coverage.gap_fraction !== undefined ? `${(coverage.gap_fraction * 100).toFixed(1)}%` : '—'}</span></div>
              </div>

              <div style={{ display: 'flex', gap: '10px' }}>
                <button className="btn btn-secondary btn-sm" onClick={() => navigate('/explorer')}>
                  <span>Open Light-Curve Explorer</span>
                  <ArrowRight size={14} />
                </button>
                <button className="btn btn-secondary btn-sm" onClick={() => navigate('/controls')}>
                  <span>Adjust Detection Settings</span>
                </button>
              </div>
            </div>
          ) : (
            <div style={{ padding: '30px 0', textAlign: 'center', color: 'var(--text-dim)' }}>
              No dataset loaded. Upload an observation or click "Run Synthetic Demo" above.
            </div>
          )}
        </div>

        {/* Current Analysis Pipeline Status */}
        <div className="card">
          <div className="card-title">
            <span>Pipeline Status</span>
            <span className={`badge ${activeJob?.status === 'completed' ? 'badge-green' : (activeJob?.status === 'running' ? 'badge-cyan' : 'badge-muted')}`}>
              {activeJob?.status ? activeJob.status.toUpperCase() : 'IDLE'}
            </span>
          </div>

          {activeJob ? (
            <div style={{ fontSize: '13px' }}>
              <div style={{ marginBottom: '12px' }}>
                <strong>Active Job ID:</strong> <span className="mono text-cyan">{activeJob.id}</span>
              </div>
              <div style={{ marginBottom: '12px' }}>
                <strong>Preset:</strong> <span className="mono" style={{ textTransform: 'capitalize' }}>{results?.config?.preset || 'Balanced'}</span> | <strong>Scorer:</strong> <span className="mono">{results?.config?.model_family || 'Random Forest'}</span>
              </div>
              <div style={{ marginBottom: '12px' }}>
                <strong>Status Message:</strong> <span className="text-muted">{activeJob.message || 'Ready'}</span>
              </div>

              {bursts.length > 0 && (
                <div style={{ marginTop: '16px', borderTop: '1px solid var(--border-subtle)', paddingTop: '12px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                    <span className="text-muted">Latest Detected Burst:</span>
                    <span className="mono font-bold text-cyan">{bursts[0].burst_id}</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                    <span className="text-muted">Peak Time:</span>
                    <span className="mono">{bursts[0].peak_time_iso}</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '12px' }}>
                    <span className="text-muted">Net Peak Rate:</span>
                    <span className="mono">{bursts[0].net_peak?.toFixed(2)} count/s</span>
                  </div>
                  <button className="btn btn-primary btn-sm" style={{ width: '100%' }} onClick={() => navigate('/catalog')}>
                    View All {bursts.length} Events in Catalog
                  </button>
                </div>
              )}
            </div>
          ) : (
            <div style={{ padding: '30px 0', textAlign: 'center', color: 'var(--text-dim)' }}>
              No active job running. Load a dataset to begin automated detection.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
