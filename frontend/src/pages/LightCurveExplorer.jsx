import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { ZoomIn, Sliders, Layers, Eye, ArrowRight } from 'lucide-react';
import PlotlyChart from '../components/PlotlyChart';
import { useTheme } from '../context/ThemeContext';

export default function LightCurveExplorer({ activeDataset, activeJob, selectedBurst, onSelectBurst }) {
  const navigate = useNavigate();
  const { isLight } = useTheme();
  const [showBaseline, setShowBaseline] = useState(true);
  const [showBurstOverlays, setShowBurstOverlays] = useState(true);

  const results = activeJob?.result_json;
  const lcData = results?.lightcurve;
  const bursts = results?.bursts || [];

  if (!activeJob || !lcData) {
    return (
      <div className="page-container" style={{ textAlign: 'center', paddingTop: '60px', maxWidth: '680px', margin: '0 auto' }}>
        <div className="card" style={{ padding: '40px 28px', border: '1px solid var(--border-subtle)' }}>
          <div style={{ width: '48px', height: '48px', borderRadius: '50%', background: 'rgba(0, 229, 255, 0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 16px auto', color: 'var(--accent-cyan)' }}>
            <Sliders size={24} />
          </div>
          <h2 style={{ color: 'var(--text-main)', fontSize: '20px', marginBottom: '10px' }}>
            {activeDataset ? 'Dataset Loaded — Detection Not Run Yet' : 'No Analysis Results Available'}
          </h2>
          <p style={{ color: 'var(--text-muted)', fontSize: '13.5px', marginBottom: '24px', lineHeight: '1.6' }}>
            {activeDataset ? (
              <>
                Observation <strong>{activeDataset.name || activeDataset.filename}</strong> is selected as the active dataset. To calculate baseline trends and detect solar bursts, run the analysis pipeline in Detection Controls.
              </>
            ) : (
              'Please upload an observation file or click "Load XSM Demo" in the top bar to explore pre-cataloged bursts.'
            )}
          </p>
          <div style={{ display: 'flex', gap: '12px', justifyContent: 'center', flexWrap: 'wrap' }}>
            {activeDataset && (
              <button className="btn btn-primary" onClick={() => navigate('/detection')}>
                <Sliders size={16} /> Go to Detection Controls
              </button>
            )}
            <button className="btn btn-secondary" onClick={() => navigate('/upload')}>
              Upload & Inspect
            </button>
          </div>
        </div>
      </div>
    );
  }

  // Build Plotly Traces
  const traces = [];

  // 1. Raw Observations
  traces.push({
    x: lcData.time_iso,
    y: lcData.value,
    error_y: {
      type: 'data',
      array: lcData.error,
      visible: true,
      color: 'rgba(0, 229, 255, 0.25)',
      thickness: 1,
      width: 0
    },
    mode: 'lines',
    name: `Observed Rate (${lcData.metadata?.unit || 'count/s'})`,
    line: { color: '#00e5ff', width: 1.2 },
    hoverinfo: 'x+y'
  });

  // 2. Baseline
  if (showBaseline && lcData.baseline) {
    traces.push({
      x: lcData.time_iso,
      y: lcData.baseline,
      mode: 'lines',
      name: 'Estimated Baseline (AsLS + Iterative)',
      line: { color: '#f59e0b', width: 1.5, dash: 'dot' },
      hoverinfo: 'x+y'
    });
  }

  // 3. Burst Markers Overlay
  if (showBurstOverlays && bursts.length > 0) {
    traces.push({
      x: bursts.map(b => b.peak_time_iso),
      y: bursts.map(b => b.total_peak),
      mode: 'markers+text',
      name: 'Detected Bursts',
      text: bursts.map(b => b.burst_id),
      textposition: 'top center',
      textfont: { color: isLight ? '#0f172a' : '#cbd5e1', size: 10, family: 'monospace' },
      marker: {
        symbol: 'diamond',
        size: 9,
        color: bursts.map(b => b.decision === 'accepted' ? '#22c55e' : (b.decision === 'review' ? '#eab308' : '#ef4444')),
        line: { color: isLight ? '#0f172a' : '#ffffff', width: 1 }
      },
      hovertext: bursts.map(b => `${b.burst_id} (${b.decision.toUpperCase()})<br>Net: ${b.net_peak.toFixed(1)} c/s<br>Dur: ${b.duration_s.toFixed(1)}s`),
      hoverinfo: 'text'
    });
  }

  // Highlight selected burst with vertical region
  const shapes = [];
  if (selectedBurst && showBurstOverlays) {
    shapes.push({
      type: 'rect',
      xref: 'x',
      yref: 'paper',
      x0: selectedBurst.start_time_iso,
      x1: selectedBurst.end_time_iso,
      y0: 0,
      y1: 1,
      fillcolor: isLight ? 'rgba(2, 132, 199, 0.12)' : 'rgba(0, 229, 255, 0.15)',
      line: { color: 'var(--accent-cyan)', width: 1.5, dash: 'dash' }
    });
  }

  const layout = {
    title: {
      text: `Light Curve: ${activeDataset?.name || 'Observation'} (${lcData.sampled_points} points displayed)`,
      font: { color: isLight ? '#0f172a' : '#ffffff', size: 14 }
    },
    xaxis: {
      title: 'Time (UTC)',
      rangeslider: { visible: true, bgcolor: isLight ? '#f8fafc' : '#090e1c', thickness: 0.1, bordercolor: isLight ? '#cbd5e1' : '#1e293b' },
      type: 'date'
    },
    yaxis: {
      title: `Count Rate (${lcData.metadata?.unit || 'count / s'})`,
      rangemode: 'tozero'
    },
    legend: { orientation: 'h', y: -0.25 },
    shapes
  };

  const handlePlotClick = (data) => {
    if (data.points && data.points[0]) {
      const pt = data.points[0];
      // Check if clicked near a burst
      if (pt.data.name === 'Detected Bursts') {
        const clickedBurst = bursts.find(b => b.burst_id === pt.text);
        if (clickedBurst) onSelectBurst(clickedBurst);
      }
    }
  };

  return (
    <div className="page-container" style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      {/* Controls Bar */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px', flexShrink: 0 }}>
        <div>
          <h1 style={{ fontSize: '20px', margin: 0 }}>Light-Curve Explorer</h1>
          <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
            Zoom, pan, and inspect events across the full observational timeline.
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <label style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '13px', cursor: 'pointer' }}>
            <input
              type="checkbox"
              checked={showBaseline}
              onChange={(e) => setShowBaseline(e.target.checked)}
            />
            <span>Show Baseline</span>
          </label>

          <label style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '13px', cursor: 'pointer' }}>
            <input
              type="checkbox"
              checked={showBurstOverlays}
              onChange={(e) => setShowBurstOverlays(e.target.checked)}
            />
            <span>Show Burst Markers</span>
          </label>

          {selectedBurst && (
            <button className="btn btn-primary btn-sm" onClick={() => navigate('/detail')}>
              <span>View {selectedBurst.burst_id} Profile Detail</span>
              <ArrowRight size={13} />
            </button>
          )}
        </div>
      </div>

      {/* Main Plot Area */}
      <div className="card" style={{ flex: 1, minHeight: '450px', padding: '12px', display: 'flex', flexDirection: 'column' }}>
        <PlotlyChart
          data={traces}
          layout={layout}
          onClick={handlePlotClick}
          style={{ flex: 1 }}
        />
      </div>

      {/* Selected Event Quick Bar */}
      {selectedBurst && (
        <div style={{ background: 'var(--bg-card)', border: '1px solid var(--border-subtle)', borderRadius: '8px', padding: '12px 18px', marginTop: '10px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', boxShadow: 'var(--shadow-card)' }}>
          <div style={{ display: 'flex', gap: '20px', fontSize: '13px' }}>
            <div><span className="text-muted">Selected Burst:</span> <strong className="mono text-cyan">{selectedBurst.burst_id}</strong></div>
            <div><span className="text-muted">Peak Time:</span> <span className="mono">{selectedBurst.peak_time_iso}</span></div>
            <div><span className="text-muted">Net Peak:</span> <span className="mono font-bold text-green">{selectedBurst.net_peak?.toFixed(2)} c/s</span></div>
            <div><span className="text-muted">Duration:</span> <span className="mono">{selectedBurst.duration_s?.toFixed(1)}s</span></div>
            <div><span className="text-muted">Morphology:</span> <span className="badge badge-cyan">{selectedBurst.morphology_class}</span></div>
          </div>
          <button className="btn btn-secondary btn-sm" onClick={() => navigate('/detail')}>
            Inspect Profile & Residuals
          </button>
        </div>
      )}
    </div>
  );
}
