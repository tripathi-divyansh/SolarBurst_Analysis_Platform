import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Sliders, CheckCircle2, Play, AlertCircle, RefreshCw, XCircle } from 'lucide-react';

export default function DetectionControls({ activeDataset, activeJob, onStartAnalysis, onCancelJob }) {
  const navigate = useNavigate();

  // Preset state
  const [preset, setPreset] = useState('balanced');
  const [modelFamily, setModelFamily] = useState('random_forest');

  // Tunable parameters
  const [aslsLambda, setAslsLambda] = useState(1e6);
  const [aslsP, setAslsP] = useState(0.01);
  const [baselineIter, setBaselineIter] = useState(3);

  const [mfThresh, setMfThresh] = useState(3.5);
  const [cwtThresh, setCwtThresh] = useState(3.0);
  const [derivThresh, setDerivThresh] = useState(2.5);
  const [seedSigma, setSeedSigma] = useState(4.0);
  const [growSigma, setGrowSigma] = useState(1.8);

  const [acceptThresh, setAcceptThresh] = useState(0.65);
  const [reviewThresh, setReviewThresh] = useState(0.35);

  const handleApplyPreset = (name) => {
    setPreset(name);
    if (name === 'conservative') {
      setAslsLambda(1e7);
      setAslsP(0.005);
      setMfThresh(4.5);
      setCwtThresh(4.0);
      setDerivThresh(3.5);
      setSeedSigma(5.0);
      setGrowSigma(2.2);
      setAcceptThresh(0.75);
      setReviewThresh(0.45);
    } else if (name === 'sensitive') {
      setAslsLambda(1e5);
      setAslsP(0.02);
      setMfThresh(2.8);
      setCwtThresh(2.5);
      setDerivThresh(2.0);
      setSeedSigma(3.2);
      setGrowSigma(1.5);
      setAcceptThresh(0.55);
      setReviewThresh(0.25);
    } else {
      // Balanced
      setAslsLambda(1e6);
      setAslsP(0.01);
      setMfThresh(3.5);
      setCwtThresh(3.0);
      setDerivThresh(2.5);
      setSeedSigma(4.0);
      setGrowSigma(1.8);
      setAcceptThresh(0.65);
      setReviewThresh(0.35);
    }
  };

  const handleRun = async () => {
    if (!activeDataset) {
      alert('Please select or upload a dataset first.');
      return;
    }

    const config = {
      preset,
      model_family: modelFamily,
      asls_lambda: aslsLambda,
      asls_p: aslsP,
      baseline_iterations: baselineIter,
      matched_filter_snr_threshold: mfThresh,
      cwt_snr_threshold: cwtThresh,
      derivative_threshold: derivThresh,
      hysteresis_seed_sigma: seedSigma,
      hysteresis_grow_sigma: growSigma,
      accept_threshold: acceptThresh,
      review_threshold: reviewThresh,
      max_components: 3,
      fast_uncertainty: true
    };

    await onStartAnalysis(activeDataset.id, config);
  };

  const isRunning = activeJob?.status === 'running';

  return (
    <div className="page-container" style={{ maxWidth: '1000px', margin: '0 auto' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
        <div>
          <h1 style={{ fontSize: '24px', marginBottom: '4px' }}>Detection Controls & Parameters</h1>
          <p style={{ color: 'var(--text-muted)', margin: 0, fontSize: '13px' }}>
            Configure candidate proposals, machine learning models, and operating points
          </p>
        </div>

        {isRunning && (
          <button className="btn btn-danger" onClick={() => onCancelJob(activeJob.id)}>
            <XCircle size={15} />
            <span>Cancel Active Analysis</span>
          </button>
        )}
      </div>

      {/* Preset Selector */}
      <div className="card" style={{ marginBottom: '20px' }}>
        <div className="card-title">
          <span>Operating Presets</span>
          <span className="badge badge-cyan">VALIDATED CONFIGURATIONS</span>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '16px' }}>
          <div
            style={{
              background: preset === 'conservative' ? 'rgba(0, 229, 255, 0.08)' : '#0a0f1d',
              border: `1px solid ${preset === 'conservative' ? 'var(--accent-cyan)' : 'var(--border-subtle)'}`,
              padding: '16px',
              borderRadius: '8px',
              cursor: 'pointer'
            }}
            onClick={() => handleApplyPreset('conservative')}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
              <strong style={{ fontSize: '15px' }}>Conservative</strong>
              {preset === 'conservative' && <CheckCircle2 size={16} className="text-cyan" />}
            </div>
            <p style={{ fontSize: '12px', color: 'var(--text-muted)', margin: 0 }}>
              Prioritizes near-zero false alarms for published science catalogs. High proposal seeds (5.0&sigma;) and strict ML acceptance (0.75).
            </p>
          </div>

          <div
            style={{
              background: preset === 'balanced' ? 'rgba(0, 229, 255, 0.08)' : '#0a0f1d',
              border: `1px solid ${preset === 'balanced' ? 'var(--accent-cyan)' : 'var(--border-subtle)'}`,
              padding: '16px',
              borderRadius: '8px',
              cursor: 'pointer'
            }}
            onClick={() => handleApplyPreset('balanced')}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
              <strong style={{ fontSize: '15px', color: 'var(--accent-cyan)' }}>Balanced (Recommended)</strong>
              {preset === 'balanced' && <CheckCircle2 size={16} className="text-cyan" />}
            </div>
            <p style={{ fontSize: '12px', color: 'var(--text-muted)', margin: 0 }}>
              Optimizes completeness subject to &le; 1.0 false alarms per 24 hours of valid observing exposure. Standard 4.0&sigma; seed and 0.65 threshold.
            </p>
          </div>

          <div
            style={{
              background: preset === 'sensitive' ? 'rgba(0, 229, 255, 0.08)' : '#0a0f1d',
              border: `1px solid ${preset === 'sensitive' ? 'var(--accent-cyan)' : 'var(--border-subtle)'}`,
              padding: '16px',
              borderRadius: '8px',
              cursor: 'pointer'
            }}
            onClick={() => handleApplyPreset('sensitive')}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
              <strong style={{ fontSize: '15px' }}>Sensitive</strong>
              {preset === 'sensitive' && <CheckCircle2 size={16} className="text-cyan" />}
            </div>
            <p style={{ fontSize: '12px', color: 'var(--text-muted)', margin: 0 }}>
              Permissive search for weak sub-A class thermal transients and micro-flares. Low seed threshold (3.2&sigma;) with relaxed review boundaries.
            </p>
          </div>
        </div>
      </div>

      {/* Model Selection */}
      <div className="card" style={{ marginBottom: '20px' }}>
        <div className="card-title">
          <span>Candidate Classifier Selection</span>
          <span className="text-muted" style={{ fontSize: '12px' }}>Independent learned scoring on identical candidate features</span>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '16px' }}>
          <label style={{ display: 'flex', alignItems: 'flex-start', gap: '10px', background: 'var(--bg-inset)', padding: '14px', borderRadius: '8px', cursor: 'pointer', border: modelFamily === 'random_forest' ? '1px solid var(--accent-cyan)' : '1px solid var(--border-subtle)' }}>
            <input
              type="radio"
              name="modelFamily"
              value="random_forest"
              checked={modelFamily === 'random_forest'}
              onChange={() => setModelFamily('random_forest')}
            />
            <div>
              <strong style={{ fontSize: '14px' }}>Random Forest (Baseline)</strong>
              <div style={{ fontSize: '11.5px', color: 'var(--text-muted)', marginTop: '4px' }}>
                Combines randomized decision trees with balanced class weights. Robust to correlated candidate features.
              </div>
            </div>
          </label>

          <label style={{ display: 'flex', alignItems: 'flex-start', gap: '10px', background: 'var(--bg-inset)', padding: '14px', borderRadius: '8px', cursor: 'pointer', border: modelFamily === 'xgboost' ? '1px solid var(--accent-cyan)' : '1px solid var(--border-subtle)' }}>
            <input
              type="radio"
              name="modelFamily"
              value="xgboost"
              checked={modelFamily === 'xgboost'}
              onChange={() => setModelFamily('xgboost')}
            />
            <div>
              <strong style={{ fontSize: '14px' }}>XGBoost (Challenger)</strong>
              <div style={{ fontSize: '11.5px', color: 'var(--text-muted)', marginTop: '4px' }}>
                Sequentially boosted trees with AUCPR early stopping. Evaluated on identical data partitions.
              </div>
            </div>
          </label>

          <label style={{ display: 'flex', alignItems: 'flex-start', gap: '10px', background: 'var(--bg-inset)', padding: '14px', borderRadius: '8px', cursor: 'pointer', border: modelFamily === 'classical_only' ? '1px solid var(--accent-cyan)' : '1px solid var(--border-subtle)' }}>
            <input
              type="radio"
              name="modelFamily"
              value="classical_only"
              checked={modelFamily === 'classical_only'}
              onChange={() => setModelFamily('classical_only')}
            />
            <div>
              <strong style={{ fontSize: '14px' }}>Classical Only</strong>
              <div style={{ fontSize: '11.5px', color: 'var(--text-muted)', marginTop: '4px' }}>
                Bypasses machine learning scorer; directly relies on matched filter and excess thresholds.
              </div>
            </div>
          </label>
        </div>
      </div>

      {/* Advanced Parameters */}
      <div className="card" style={{ marginBottom: '24px' }}>
        <div className="card-title">
          <span>Signal Processing & Candidate Parameters</span>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '18px' }}>
          <div className="form-group">
            <label className="form-label">AsLS Smoothing (&lambda;)</label>
            <input
              type="number"
              style={{ width: '100%' }}
              value={aslsLambda}
              onChange={(e) => setAslsLambda(parseFloat(e.target.value))}
            />
            <div style={{ fontSize: '11px', color: 'var(--text-dim)', marginTop: '4px' }}>Default: 1,000,000</div>
          </div>

          <div className="form-group">
            <label className="form-label">Asymmetry Parameter (p)</label>
            <input
              type="number"
              step="0.005"
              style={{ width: '100%' }}
              value={aslsP}
              onChange={(e) => setAslsP(parseFloat(e.target.value))}
            />
            <div style={{ fontSize: '11px', color: 'var(--text-dim)', marginTop: '4px' }}>Downweights positive flare residuals</div>
          </div>

          <div className="form-group">
            <label className="form-label">Hysteresis Seed (&sigma;)</label>
            <input
              type="number"
              step="0.5"
              style={{ width: '100%' }}
              value={seedSigma}
              onChange={(e) => setSeedSigma(parseFloat(e.target.value))}
            />
            <div style={{ fontSize: '11px', color: 'var(--text-dim)', marginTop: '4px' }}>Minimum standardized excess to initiate candidate</div>
          </div>

          <div className="form-group">
            <label className="form-label">Matched Filter SNR</label>
            <input
              type="number"
              step="0.5"
              style={{ width: '100%' }}
              value={mfThresh}
              onChange={(e) => setMfThresh(parseFloat(e.target.value))}
            />
          </div>

          <div className="form-group">
            <label className="form-label">CWT Wavelet SNR</label>
            <input
              type="number"
              step="0.5"
              style={{ width: '100%' }}
              value={cwtThresh}
              onChange={(e) => setCwtThresh(parseFloat(e.target.value))}
            />
          </div>

          <div className="form-group">
            <label className="form-label">ML Accept Threshold</label>
            <input
              type="number"
              step="0.05"
              min="0"
              max="1"
              style={{ width: '100%' }}
              value={acceptThresh}
              onChange={(e) => setAcceptThresh(parseFloat(e.target.value))}
            />
            <div style={{ fontSize: '11px', color: 'var(--text-dim)', marginTop: '4px' }}>Score &ge; threshold automatically accepted</div>
          </div>
        </div>

        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '12px', marginTop: '16px' }}>
          <button className="btn btn-primary" onClick={handleRun} disabled={isRunning}>
            {isRunning ? <RefreshCw size={15} className="animate-spin" /> : <Play size={15} />}
            <span>{isRunning ? 'Analyzing Light Curve...' : 'Execute Analysis Pipeline'}</span>
          </button>
        </div>
      </div>
    </div>
  );
}
