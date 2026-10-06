/**
 * HTML Report Generator
 * Assembles an auditable, self-contained scientific report for an observation analysis.
 */

export function generateHtmlReport({ dataset, job, results }) {
  const summary = results.summary || {};
  const bursts = results.bursts || [];
  const coverage = results.coverage || {};
  const cfg = results.config || {};

  const rows = bursts.map(b => `
    <tr>
      <td class="mono font-bold">${b.burst_id}</td>
      <td class="mono">${b.peak_time_iso}</td>
      <td class="mono">${b.duration_s.toFixed(1)}s</td>
      <td class="mono">${b.net_peak.toFixed(2)}</td>
      <td class="mono">${b.fluence.toFixed(1)}</td>
      <td><span class="badge ${b.morphology_class.includes('Fast') ? 'badge-cyan' : 'badge-slate'}">${b.morphology_class}</span></td>
      <td><span class="badge ${b.reliability === 'reliable' ? 'badge-green' : 'badge-yellow'}">${b.reliability}</span></td>
      <td class="mono">${b.reduced_chi2.toFixed(2)}</td>
      <td class="mono">${b.ml_score.toFixed(2)}</td>
      <td><span class="badge ${b.decision === 'accepted' ? 'badge-green' : (b.decision === 'review' ? 'badge-yellow' : 'badge-red')}">${b.decision.toUpperCase()}</span></td>
    </tr>
  `).join('');

  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>SolarBurst Analysis Report — ${dataset.name || 'Observation'}</title>
  <style>
    :root {
      --bg: #0a0f1d;
      --card: #121a2d;
      --border: #1e293b;
      --cyan: #00e5ff;
      --text: #f8fafc;
      --muted: #94a3b8;
    }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background-color: var(--bg);
      color: var(--text);
      margin: 0;
      padding: 40px;
      line-height: 1.5;
    }
    .container { max-width: 1100px; margin: 0 auto; }
    .header { border-bottom: 2px solid var(--border); padding-bottom: 24px; margin-bottom: 32px; }
    h1 { color: var(--cyan); margin: 0 0 8px 0; font-size: 28px; }
    .meta-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin: 24px 0; }
    .meta-card { background: var(--card); border: 1px solid var(--border); padding: 16px; border-radius: 8px; }
    .meta-title { font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; }
    .meta-value { font-size: 22px; font-weight: bold; margin-top: 4px; color: #fff; }
    table { width: 100%; border-collapse: collapse; margin: 24px 0; background: var(--card); border-radius: 8px; overflow: hidden; border: 1px solid var(--border); }
    th { background: #162036; text-align: left; padding: 12px 14px; font-size: 12px; color: var(--muted); text-transform: uppercase; border-bottom: 1px solid var(--border); }
    td { padding: 12px 14px; border-bottom: 1px solid var(--border); font-size: 13px; }
    tr:last-child td { border-bottom: none; }
    .mono { font-family: ui-monospace, monospace; }
    .badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; text-transform: uppercase; }
    .badge-cyan { background: rgba(0,229,255,0.15); color: var(--cyan); }
    .badge-green { background: rgba(34,197,94,0.15); color: #4ade80; }
    .badge-yellow { background: rgba(234,179,8,0.15); color: #facc15; }
    .badge-red { background: rgba(239,68,68,0.15); color: #f87171; }
    .badge-slate { background: rgba(148,163,184,0.15); color: var(--muted); }
    .footer { margin-top: 40px; padding-top: 20px; border-top: 1px solid var(--border); font-size: 12px; color: var(--muted); }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>SolarBurst — XSM Solar Burst Analysis Report</h1>
      <p style="color: var(--muted); margin: 0;">Authoritative analysis of Chandrayaan-2 Solar X-ray Monitor observation</p>
    </div>

    <div class="meta-grid">
      <div class="meta-card">
        <div class="meta-title">Dataset</div>
        <div class="meta-value" style="font-size: 16px;">${dataset.name || dataset.filename}</div>
        <div style="font-size: 11px; color: var(--muted); margin-top: 4px;">SHA: ${dataset.sha256 ? dataset.sha256.slice(0, 12) : 'demo'}...</div>
      </div>
      <div class="meta-card">
        <div class="meta-title">Valid Exposure</div>
        <div class="meta-value">${coverage.valid_exposure_s ? (coverage.valid_exposure_s / 60).toFixed(1) : '0'} min</div>
        <div style="font-size: 11px; color: var(--muted); margin-top: 4px;">${coverage.valid_points || 0} valid points</div>
      </div>
      <div class="meta-card">
        <div class="meta-title">Detected Bursts</div>
        <div class="meta-value" style="color: var(--cyan);">${bursts.length}</div>
        <div style="font-size: 11px; color: var(--muted); margin-top: 4px;">${summary.accepted_count || 0} accepted, ${summary.review_count || 0} review</div>
      </div>
      <div class="meta-card">
        <div class="meta-title">Analysis Preset</div>
        <div class="meta-value" style="font-size: 18px; text-transform: capitalize;">${cfg.preset || 'Balanced'}</div>
        <div style="font-size: 11px; color: var(--muted); margin-top: 4px;">Model: ${cfg.model_family || 'Random Forest'}</div>
      </div>
    </div>

    <h2>Identified Solar Bursts Catalog</h2>
    <table>
      <thead>
        <tr>
          <th>Burst ID</th>
          <th>Peak Time (UTC)</th>
          <th>Duration</th>
          <th>Net Peak</th>
          <th>Fluence</th>
          <th>Morphology</th>
          <th>Reliability</th>
          <th>Red. &chi;&sup2;</th>
          <th>ML Score</th>
          <th>Status</th>
        </tr>
      </thead>
      <tbody>
        ${rows || '<tr><td colspan="10" style="text-align:center; padding: 24px;">No solar bursts detected in this observation.</td></tr>'}
      </tbody>
    </table>

    <div class="footer">
      Generated automatically by <strong>SolarBurst — XSM Burst Analysis Platform v1.0.0</strong> | Inter IIT Tech Meet 10.0 ISRO Solution | All parameters calculated via Python scientific pipeline (EMG convolution & multiscale candidate generators).
    </div>
  </div>
</body>
</html>`;
}
