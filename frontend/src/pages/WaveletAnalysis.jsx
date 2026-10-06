import React, { useState, useEffect } from 'react';
import { Waves, RefreshCw, ZoomIn, Info, AlertCircle } from 'lucide-react';
import PlotlyChart from '../components/PlotlyChart';
import { api } from '../api';

export default function WaveletAnalysis({ activeDataset, selectedBurst }) {
  const [waveletType, setWaveletType] = useState('morlet');
  const [scalogramData, setScalogramData] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState(null);

  const fetchScalogram = async () => {
    if (!activeDataset) return;
    setIsLoading(true);
    setErrorMsg(null);

    try {
      const params = {
        dataset_id: activeDataset.id,
        wavelet: waveletType,
        num_scales: 35
      };

      if (selectedBurst) {
        params.start_met = selectedBurst.start_time_met - 60.0;
        params.end_met = selectedBurst.end_time_met + 120.0;
      }

      const res = await api.computeWavelet(params);
      setScalogramData(res);
    } catch (err) {
      setErrorMsg(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    if (activeDataset) {
      fetchScalogram();
    }
  }, [activeDataset, selectedBurst, waveletType]);

  const heatmapTrace = scalogramData ? [
    {
      z: scalogramData.power,
      x: scalogramData.times,
      y: scalogramData.scales_s,
      type: 'heatmap',
      colorscale: 'Viridis',
      colorbar: { title: 'Wavelet Power', len: 0.8 },
      hoverinfo: 'x+y+z'
    },
    {
      x: scalogramData.times,
      y: scalogramData.coi_s,
      mode: 'lines',
      name: 'Cone of Influence (COI)',
      line: { color: 'rgba(255, 255, 255, 0.6)', dash: 'dash', width: 1.5 },
      hoverinfo: 'none'
    }
  ] : [];

  const layout = {
    title: {
      text: `Scalogram: ${activeDataset?.name || 'Observation'} (${waveletType.toUpperCase()} Wavelet)`,
      font: { color: '#fff', size: 13 }
    },
    xaxis: { title: 'Time (MET Seconds)' },
    yaxis: { title: 'Timescale (Seconds)', type: 'log' },
    margin: { l: 60, r: 20, t: 30, b: 40 }
  };

  return (
    <div className="page-container" style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      {/* Control Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px', flexShrink: 0 }}>
        <div>
          <h1 style={{ fontSize: '20px', margin: 0 }}>Wavelet Time-Frequency Scalogram</h1>
          <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
            Continuous Wavelet Transform (CWT) across physical timescales with edge influence masking.
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '13px' }}>
            <span className="text-muted">Wavelet Family:</span>
            <select
              value={waveletType}
              onChange={(e) => setWaveletType(e.target.value)}
              style={{ padding: '5px 10px' }}
            >
              <option value="morlet">Morlet (Time-Frequency / Oscillations)</option>
              <option value="mexh">Mexican Hat / Ricker (Impulsive Bursts)</option>
            </select>
          </div>

          <button className="btn btn-secondary btn-sm" onClick={fetchScalogram} disabled={isLoading}>
            {isLoading ? <RefreshCw size={13} className="animate-spin" /> : <RefreshCw size={13} />}
            <span>Refresh</span>
          </button>
        </div>
      </div>

      {errorMsg && (
        <div style={{ background: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.3)', padding: '10px 14px', borderRadius: '6px', marginBottom: '16px', color: '#fca5a5', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <AlertCircle size={15} />
          <span>{errorMsg}</span>
        </div>
      )}

      {selectedBurst && (
        <div style={{ background: 'rgba(0, 229, 255, 0.08)', border: '1px solid rgba(0, 229, 255, 0.25)', padding: '8px 14px', borderRadius: '6px', marginBottom: '16px', fontSize: '12.5px', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Info size={15} className="text-cyan" />
          <span>Focused on selected burst interval: <strong>{selectedBurst.burst_id}</strong> (Span: ~{(selectedBurst.duration_s).toFixed(1)}s)</span>
        </div>
      )}

      {/* Main Scalogram Plot */}
      <div className="card" style={{ flex: 1, minHeight: '400px', padding: '12px', display: 'flex', flexDirection: 'column' }}>
        {scalogramData ? (
          <PlotlyChart data={heatmapTrace} layout={layout} style={{ flex: 1 }} />
        ) : (
          <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-dim)' }}>
            {isLoading ? 'Computing Continuous Wavelet Transform on backend...' : 'No scalogram available.'}
          </div>
        )}
      </div>

      <div style={{ fontSize: '11.5px', color: 'var(--text-dim)', marginTop: '8px' }}>
        <strong>Scientific Note:</strong> The dashed white curve marks the Cone of Influence (COI). Wavelet coefficients below the COI are affected by observational boundaries and should not be interpreted as physical periodicities.
      </div>
    </div>
  );
}
