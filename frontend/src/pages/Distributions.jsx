import React from 'react';
import { BarChart3, PieChart, Activity } from 'lucide-react';
import PlotlyChart from '../components/PlotlyChart';
import MetricCard from '../components/MetricCard';

export default function Distributions({ activeJob }) {
  const bursts = activeJob?.result_json?.bursts || [];

  if (bursts.length === 0) {
    return (
      <div className="page-container" style={{ textAlign: 'center', paddingTop: '80px' }}>
        <h2 style={{ color: 'var(--text-muted)' }}>No Burst Catalog Data Available</h2>
        <p style={{ color: 'var(--text-dim)' }}>Run an analysis on a dataset to generate parameter distributions.</p>
      </div>
    );
  }

  const durations = bursts.map(b => b.duration_s);
  const decays = bursts.map(b => b.decay_tau_s);
  const netPeaks = bursts.map(b => b.net_peak);
  const fluences = bursts.map(b => b.fluence);
  const asyms = bursts.map(b => b.asymmetry_rho);

  // Median calculations
  const median = (arr) => {
    const s = [...arr].sort((a, b) => a - b);
    const mid = Math.floor(s.length / 2);
    return s.length % 2 !== 0 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
  };

  // 1. Duration Histogram
  const durTrace = [
    {
      x: durations,
      type: 'histogram',
      nbinsx: 15,
      marker: { color: '#00e5ff', line: { color: '#090e1c', width: 1 } },
      name: 'Duration (s)'
    }
  ];
  const durLayout = {
    title: { text: 'Burst Duration Distribution (s)', font: { color: '#fff', size: 13 } },
    xaxis: { title: 'Duration (s)' },
    yaxis: { title: 'Event Count' },
    margin: { l: 50, r: 20, t: 30, b: 35 }
  };

  // 2. Decay Constant Histogram
  const decayTrace = [
    {
      x: decays,
      type: 'histogram',
      nbinsx: 15,
      marker: { color: '#4ade80', line: { color: '#090e1c', width: 1 } },
      name: 'Decay Tau (s)'
    }
  ];
  const decayLayout = {
    title: { text: 'Effective Cooling Decay Constant \\tau (s)', font: { color: '#fff', size: 13 } },
    xaxis: { title: 'Decay Constant \\tau (s)' },
    yaxis: { title: 'Event Count' },
    margin: { l: 50, r: 20, t: 30, b: 35 }
  };

  // 3. Scatter: Fluence vs Duration
  const scatterTrace = [
    {
      x: durations,
      y: fluences,
      text: bursts.map(b => `${b.burst_id} (${b.morphology_class})`),
      mode: 'markers',
      type: 'scatter',
      marker: {
        size: 8,
        color: bursts.map(b => b.morphology_class.includes('Fast') ? '#00e5ff' : '#f59e0b'),
        line: { color: '#fff', width: 0.5 }
      }
    }
  ];
  const scatterLayout = {
    title: { text: 'Fluence vs Duration (Colored by Morphology)', font: { color: '#fff', size: 13 } },
    xaxis: { title: 'Duration (s)' },
    yaxis: { title: 'Fluence (Integrated Counts)', type: 'log' },
    margin: { l: 60, r: 20, t: 30, b: 35 }
  };

  // 4. Asymmetry Ratio Histogram
  const asymTrace = [
    {
      x: asyms,
      type: 'histogram',
      nbinsx: 15,
      marker: { color: '#f59e0b', line: { color: '#090e1c', width: 1 } },
      name: 'Asymmetry Ratio'
    }
  ];
  const asymLayout = {
    title: { text: 'Morphology Asymmetry Ratio \\rho', font: { color: '#fff', size: 13 } },
    xaxis: { title: 'Asymmetry \\rho (< 0.5 = Fast-rise/slow-decay)' },
    yaxis: { title: 'Event Count' },
    margin: { l: 50, r: 20, t: 30, b: 35 }
  };

  return (
    <div className="page-container">
      <div style={{ marginBottom: '20px' }}>
        <h1 style={{ fontSize: '22px', marginBottom: '4px' }}>Parameter Population Distributions</h1>
        <p style={{ color: 'var(--text-muted)', margin: 0, fontSize: '13px' }}>
          Statistical summary of {bursts.length} detected solar bursts across physical timescales and amplitudes.
        </p>
      </div>

      {/* Summary Metrics */}
      <div className="metrics-grid">
        <MetricCard
          label="Median Duration"
          value={`${median(durations).toFixed(1)} s`}
          sub="T5% crossing span"
          color="cyan"
        />
        <MetricCard
          label="Median Decay \\tau"
          value={`${median(decays).toFixed(1)} s`}
          sub="EMG response timescale"
          color="green"
        />
        <MetricCard
          label="Median Net Peak"
          value={`${median(netPeaks).toFixed(1)} c/s`}
          sub="Above fitted baseline"
          color="amber"
        />
        <MetricCard
          label="Median Asymmetry \\rho"
          value={median(asyms).toFixed(2)}
          sub="Rise-to-decay ratio"
          color="cyan"
        />
      </div>

      {/* Plots Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
        <div className="card" style={{ height: '320px', padding: '12px' }}>
          <PlotlyChart data={durTrace} layout={durLayout} />
        </div>
        <div className="card" style={{ height: '320px', padding: '12px' }}>
          <PlotlyChart data={decayTrace} layout={decayLayout} />
        </div>
        <div className="card" style={{ height: '320px', padding: '12px' }}>
          <PlotlyChart data={scatterTrace} layout={scatterLayout} />
        </div>
        <div className="card" style={{ height: '320px', padding: '12px' }}>
          <PlotlyChart data={asymTrace} layout={asymLayout} />
        </div>
      </div>
    </div>
  );
}
