import React from 'react';
import { useNavigate } from 'react-router-dom';
import { ZoomIn, ArrowLeft, ArrowRight, ShieldCheck, Waves, Info } from 'lucide-react';
import PlotlyChart from '../components/PlotlyChart';

export default function EventDetail({ activeJob, selectedBurst, onSelectBurst }) {
  const navigate = useNavigate();
  const bursts = activeJob?.result_json?.bursts || [];
  const lcData = activeJob?.result_json?.lightcurve;

  if (!selectedBurst) {
    return (
      <div className="page-container" style={{ textAlign: 'center', paddingTop: '80px' }}>
        <h2 style={{ color: 'var(--text-muted)' }}>No Burst Selected</h2>
        <p style={{ color: 'var(--text-dim)', marginBottom: '20px' }}>
          Select an event from the Burst Catalog or Light-Curve Explorer to view its profile fit.
        </p>
        <button className="btn btn-primary" onClick={() => navigate('/catalog')}>
          Go to Burst Catalog
        </button>
      </div>
    );
  }

  // Find index in catalog for prev/next
  const currentIndex = bursts.findIndex((b) => b.burst_id === selectedBurst.burst_id);
  const prevBurst = currentIndex > 0 ? bursts[currentIndex - 1] : null;
  const nextBurst = currentIndex < bursts.length - 1 ? bursts[currentIndex + 1] : null;

  // Extract observation slice around event context
  let sliceTimes = [];
  let sliceValues = [];
  let sliceErrors = [];

  if (lcData && lcData.met_seconds) {
    const tStart = selectedBurst.start_time_met - 40.0;
    const tEnd = selectedBurst.end_time_met + 60.0;
    for (let i = 0; i < lcData.met_seconds.length; i++) {
      const t = lcData.met_seconds[i];
      if (t >= tStart && t <= tEnd) {
        sliceTimes.push(lcData.time_iso[i]);
        sliceValues.push(lcData.value[i]);
        sliceErrors.push(lcData.error ? lcData.error[i] : null);
      }
    }
  }

  // Generate synthetic smooth curve for the fitted EMG model
  const nDense = 200;
  const t0 = selectedBurst.start_time_met - 20.0;
  const t1 = selectedBurst.end_time_met + 40.0;
  const step = (t1 - t0) / nDense;
  
  const modelTimes = [];
  const modelValues = [];
  const modelBaselines = [];
  const residuals = [];
  const residualTimes = [];

  for (let i = 0; i < nDense; i++) {
    const t = t0 + i * step;
    // Approximated EMG shape for display
    const dt = t - selectedBurst.peak_time_met;
    let sig = 0;
    if (dt < 0) {
      sig = selectedBurst.net_peak * Math.exp(-0.5 * Math.pow(dt / Math.max(selectedBurst.rise_time_s * 0.5, 1), 2));
    } else {
      sig = selectedBurst.net_peak * Math.exp(-dt / Math.max(selectedBurst.decay_tau_s, 1));
    }
    const bg = selectedBurst.background_at_peak + (selectedBurst.background_slope || 0) * (t - selectedBurst.peak_time_met);
    
    // Convert to ISO string approximation
    const isoApprox = new Date(new Date(selectedBurst.peak_time_iso).getTime() + dt * 1000).toISOString();
    modelTimes.push(isoApprox);
    modelValues.push(bg + sig);
    modelBaselines.push(bg);
  }

  // Interpolated residuals for data slice
  for (let i = 0; i < sliceValues.length; i++) {
    if (sliceValues[i] !== null) {
      // Find nearest model value
      const modelVal = modelValues[Math.min(Math.floor((i / sliceValues.length) * nDense), nDense - 1)];
      residuals.push(sliceValues[i] - modelVal);
      residualTimes.push(sliceTimes[i]);
    }
  }

  // Trace definitions
  const fitTraces = [
    {
      x: sliceTimes,
      y: sliceValues,
      error_y: { type: 'data', array: sliceErrors, visible: true, color: 'rgba(0, 229, 255, 0.3)' },
      mode: 'markers',
      name: 'Observed Rate',
      marker: { color: '#00e5ff', size: 5 },
      type: 'scatter'
    },
    {
      x: modelTimes,
      y: modelValues,
      mode: 'lines',
      name: 'Fitted EMG Model',
      line: { color: '#4ade80', width: 2 },
      type: 'scatter'
    },
    {
      x: modelTimes,
      y: modelBaselines,
      mode: 'lines',
      name: 'Fitted Baseline',
      line: { color: '#f59e0b', width: 1.5, dash: 'dot' },
      type: 'scatter'
    }
  ];

  const fitLayout = {
    title: { text: `Profile Fit: ${selectedBurst.burst_id} (${selectedBurst.morphology_class})`, font: { color: '#fff', size: 13 } },
    xaxis: { title: 'Time (UTC)', type: 'date' },
    yaxis: { title: 'Count Rate (c/s)', rangemode: 'tozero' },
    margin: { l: 55, r: 20, t: 30, b: 35 },
    legend: { orientation: 'h', y: -0.2 }
  };

  const residTraces = [
    {
      x: residualTimes,
      y: residuals,
      mode: 'markers',
      name: 'Residual (Obs - Fit)',
      marker: { color: '#cbd5e1', size: 4 },
      type: 'scatter'
    }
  ];

  const residLayout = {
    title: { text: 'Fit Residuals and Uncertainty Band', font: { color: '#cbd5e1', size: 12 } },
    xaxis: { title: 'Time (UTC)', type: 'date' },
    yaxis: { title: 'Residual (c/s)', zeroline: true, zerolinecolor: 'rgba(255,255,255,0.4)' },
    margin: { l: 55, r: 20, t: 25, b: 35 }
  };

  return (
    <div className="page-container" style={{ display: 'flex', flexDirection: 'column' }}>
      {/* Header Bar */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <button className="btn btn-secondary btn-sm" onClick={() => navigate('/catalog')}>
            <ArrowLeft size={14} />
            <span>Back to Catalog</span>
          </button>
          <h1 style={{ fontSize: '20px', margin: 0 }}>Event Detail: <span className="text-cyan mono">{selectedBurst.burst_id}</span></h1>
          <span className={`badge ${selectedBurst.decision === 'accepted' ? 'badge-green' : 'badge-amber'}`}>
            {selectedBurst.decision.toUpperCase()}
          </span>
          <span className={`badge ${selectedBurst.reliability === 'reliable' ? 'badge-cyan' : 'badge-amber'}`}>
            {selectedBurst.reliability}
          </span>
        </div>

        <div style={{ display: 'flex', gap: '8px' }}>
          <button
            className="btn btn-secondary btn-sm"
            disabled={!prevBurst}
            onClick={() => onSelectBurst(prevBurst)}
          >
            <ArrowLeft size={13} />
            <span>Previous</span>
          </button>
          <button
            className="btn btn-secondary btn-sm"
            disabled={!nextBurst}
            onClick={() => onSelectBurst(nextBurst)}
          >
            <span>Next</span>
            <ArrowRight size={13} />
          </button>
        </div>
      </div>

      {/* Main Plots Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: '16px', marginBottom: '20px' }}>
        <div className="card" style={{ height: '340px', padding: '12px' }}>
          <PlotlyChart data={fitTraces} layout={fitLayout} />
        </div>
        <div className="card" style={{ height: '340px', padding: '12px' }}>
          <PlotlyChart data={residTraces} layout={residLayout} />
        </div>
      </div>

      {/* Parameter Cards Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '16px', marginBottom: '20px' }}>
        {/* Timing */}
        <div className="card" style={{ padding: '14px' }}>
          <div className="card-title" style={{ fontSize: '13px', marginBottom: '10px' }}>
            <span>Timing (MET & UTC)</span>
          </div>
          <div style={{ fontSize: '12px', display: 'flex', flexDirection: 'column', gap: '6px' }}>
            <div><span className="text-muted">Peak Time:</span> <span className="mono">{selectedBurst.peak_time_iso}</span></div>
            <div><span className="text-muted">Start (5%):</span> <span className="mono">{selectedBurst.start_time_iso}</span></div>
            <div><span className="text-muted">End (5%):</span> <span className="mono">{selectedBurst.end_time_iso}</span></div>
            <div><span className="text-muted">Heating Center &mu;:</span> <span className="mono">{selectedBurst.heating_center_met?.toFixed(1)}s</span></div>
          </div>
        </div>

        {/* Durations */}
        <div className="card" style={{ padding: '14px' }}>
          <div className="card-title" style={{ fontSize: '13px', marginBottom: '10px' }}>
            <span>Durations & Timescales</span>
          </div>
          <div style={{ fontSize: '12px', display: 'flex', flexDirection: 'column', gap: '6px' }}>
            <div><span className="text-muted">Duration (T5%):</span> <strong className="mono">{selectedBurst.duration_s?.toFixed(1)} s</strong></div>
            <div><span className="text-muted">Rise Time:</span> <span className="mono">{selectedBurst.rise_time_s?.toFixed(1)} s</span></div>
            <div><span className="text-muted">Decay Duration:</span> <span className="mono">{selectedBurst.decay_duration_s?.toFixed(1)} s</span></div>
            <div><span className="text-muted">Decay Constant &tau;:</span> <span className="mono text-cyan">{selectedBurst.decay_tau_s?.toFixed(1)} s</span></div>
            <div><span className="text-muted">FWHM:</span> <span className="mono">{selectedBurst.fwhm_s?.toFixed(1)} s</span></div>
          </div>
        </div>

        {/* Amplitudes */}
        <div className="card" style={{ padding: '14px' }}>
          <div className="card-title" style={{ fontSize: '13px', marginBottom: '10px' }}>
            <span>Amplitudes & Fluence</span>
          </div>
          <div style={{ fontSize: '12px', display: 'flex', flexDirection: 'column', gap: '6px' }}>
            <div><span className="text-muted">Net Peak Rate:</span> <strong className="mono text-green">{selectedBurst.net_peak?.toFixed(2)} c/s</strong></div>
            <div><span className="text-muted">Total Peak Rate:</span> <span className="mono">{selectedBurst.total_peak?.toFixed(2)} c/s</span></div>
            <div><span className="text-muted">Background at Peak:</span> <span className="mono">{selectedBurst.background_at_peak?.toFixed(2)} c/s</span></div>
            <div><span className="text-muted">Peak SNR:</span> <span className="mono font-bold text-cyan">{selectedBurst.peak_snr?.toFixed(1)}&sigma;</span></div>
            <div><span className="text-muted">Fluence (Counts):</span> <span className="mono">{selectedBurst.fluence?.toFixed(1)}</span></div>
          </div>
        </div>

        {/* Diagnostics */}
        <div className="card" style={{ padding: '14px' }}>
          <div className="card-title" style={{ fontSize: '13px', marginBottom: '10px' }}>
            <span>Fit Diagnostics & ML</span>
          </div>
          <div style={{ fontSize: '12px', display: 'flex', flexDirection: 'column', gap: '6px' }}>
            <div><span className="text-muted">Reduced &chi;&sup2;:</span> <span className="mono">{selectedBurst.reduced_chi2?.toFixed(2)}</span></div>
            <div><span className="text-muted">AICc / BIC:</span> <span className="mono">{selectedBurst.aicc?.toFixed(1)} / {selectedBurst.bic?.toFixed(1)}</span></div>
            <div><span className="text-muted">Coefficient R&sup2;:</span> <span className="mono">{selectedBurst.r_squared?.toFixed(3)}</span></div>
            <div><span className="text-muted">ML Candidate Score:</span> <strong className="mono text-cyan">{selectedBurst.ml_score?.toFixed(2)}</strong></div>
            <div><span className="text-muted">Morphology Ratio &rho;:</span> <span className="mono">{selectedBurst.asymmetry_rho?.toFixed(2)}</span></div>
          </div>
        </div>
      </div>
    </div>
  );
}
