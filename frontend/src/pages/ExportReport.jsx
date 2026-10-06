import React from 'react';
import { FileDown, FileText, Code2, Globe, ShieldCheck } from 'lucide-react';
import { api } from '../api';

export default function ExportReport({ activeDataset, activeJob }) {
  const jobId = activeJob?.id;
  const results = activeJob?.result_json;
  const bursts = results?.bursts || [];

  if (!activeJob) {
    return (
      <div className="page-container" style={{ textAlign: 'center', paddingTop: '80px' }}>
        <h2 style={{ color: 'var(--text-muted)' }}>No Analysis Available to Export</h2>
        <p style={{ color: 'var(--text-dim)' }}>Please load and analyze an observation first.</p>
      </div>
    );
  }

  return (
    <div className="page-container" style={{ maxWidth: '1000px', margin: '0 auto' }}>
      <div style={{ marginBottom: '24px' }}>
        <h1 style={{ fontSize: '24px', marginBottom: '4px' }}>Export Catalogs & Reports</h1>
        <p style={{ color: 'var(--text-muted)', margin: 0, fontSize: '13px' }}>
          Download auditable event catalogs, complete provenance manifests, and self-contained reports.
        </p>
      </div>

      {/* Export Options Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '20px', marginBottom: '30px' }}>
        {/* CSV */}
        <div className="card" style={{ display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px' }}>
              <FileText size={20} className="text-cyan" />
              <strong style={{ fontSize: '16px' }}>CSV Catalog</strong>
            </div>
            <p style={{ fontSize: '12.5px', color: 'var(--text-muted)', lineHeight: '1.4' }}>
              Tabular catalog containing all {bursts.length} events with exact timing, durations, net peaks, fluences, classifications, and parameter uncertainties.
            </p>
          </div>
          <a
            href={api.getCsvExportUrl(jobId)}
            download
            className="btn btn-primary"
            style={{ width: '100%', marginTop: '16px' }}
          >
            <FileDown size={15} />
            <span>Download CSV</span>
          </a>
        </div>

        {/* HTML Report */}
        <div className="card" style={{ display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px' }}>
              <Globe size={20} className="text-green" />
              <strong style={{ fontSize: '16px' }}>HTML Report</strong>
            </div>
            <p style={{ fontSize: '12.5px', color: 'var(--text-muted)', lineHeight: '1.4' }}>
              Self-contained, auditable analysis report formatted with observatory styling, summary statistics, and complete burst catalog.
            </p>
          </div>
          <a
            href={api.getReportExportUrl(jobId)}
            download
            className="btn btn-secondary"
            style={{ width: '100%', marginTop: '16px' }}
          >
            <FileDown size={15} />
            <span>Download HTML Report</span>
          </a>
        </div>

        {/* JSON Manifest */}
        <div className="card" style={{ display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px' }}>
              <Code2 size={20} className="text-amber" />
              <strong style={{ fontSize: '16px' }}>Provenance JSON</strong>
            </div>
            <p style={{ fontSize: '12.5px', color: 'var(--text-muted)', lineHeight: '1.4' }}>
              Full machine-readable provenance manifest with SHA-256 source hash, algorithm parameters, environment metadata, and raw fit diagnostics.
            </p>
          </div>
          <a
            href={api.getJsonExportUrl(jobId)}
            download
            className="btn btn-secondary"
            style={{ width: '100%', marginTop: '16px' }}
          >
            <FileDown size={15} />
            <span>Download JSON</span>
          </a>
        </div>
      </div>

      {/* Parameter Definitions Glossary */}
      <div className="card">
        <div className="card-title">
          <span>Catalog Parameter Definitions & Units</span>
        </div>

        <div className="table-container">
          <table className="scientific-table">
            <thead>
              <tr>
                <th>Column Name</th>
                <th>Standard Unit</th>
                <th>Mathematical Definition</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td className="mono text-cyan">PEAK_TIME_MET</td>
                <td className="mono">Seconds</td>
                <td>Time of maximum fitted burst component s(t), distinct from Gaussian heating center &mu;.</td>
              </tr>
              <tr>
                <td className="mono text-cyan">NET_PEAK</td>
                <td className="mono">count / s</td>
                <td>Background-subtracted maximum amplitude of the fitted profile.</td>
              </tr>
              <tr>
                <td className="mono text-cyan">DURATION_S</td>
                <td className="mono">Seconds</td>
                <td>Time interval between pre-peak and post-peak 5% crossings of the net peak.</td>
              </tr>
              <tr>
                <td className="mono text-cyan">DECAY_TAU_S</td>
                <td className="mono">Seconds</td>
                <td>Effective exponential cooling response timescale &tau;_c from bin-averaged EMG convolution.</td>
              </tr>
              <tr>
                <td className="mono text-cyan">FLUENCE</td>
                <td className="mono">Counts</td>
                <td>Analytic integral of the net fitted profile s(t) across all time.</td>
              </tr>
              <tr>
                <td className="mono text-cyan">ASYMMETRY_RHO</td>
                <td className="mono">Ratio</td>
                <td>(t_peak - t_rise,10%) / (t_decay,10% - t_peak). Operational temporal morphology ratio.</td>
              </tr>
              <tr>
                <td className="mono text-cyan">REDUCED_CHI2</td>
                <td className="mono">Dimensionless</td>
                <td>Residual sum of squares divided by degrees of freedom (&nu; = N - 6).</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
