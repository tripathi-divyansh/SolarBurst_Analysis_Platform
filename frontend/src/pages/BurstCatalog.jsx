import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { TableProperties, ShieldCheck, ArrowRight, Filter, Download } from 'lucide-react';
import AuditModal from '../components/AuditModal';

export default function BurstCatalog({ activeJob, selectedBurst, onSelectBurst, onUpdateAudit }) {
  const navigate = useNavigate();
  const [decisionFilter, setDecisionFilter] = useState('all');
  const [reliabilityFilter, setReliabilityFilter] = useState('all');
  const [minSnr, setMinSnr] = useState(0);
  const [auditingBurst, setAuditingBurst] = useState(null);

  const bursts = activeJob?.result_json?.bursts || [];

  // Filter bursts
  const filtered = bursts.filter((b) => {
    if (decisionFilter !== 'all' && b.decision !== decisionFilter) return false;
    if (reliabilityFilter !== 'all' && b.reliability !== reliabilityFilter) return false;
    if (minSnr > 0 && b.peak_snr < minSnr) return false;
    return true;
  });

  const handleRowClick = (b) => {
    onSelectBurst(b);
  };

  const handleInspectDetail = (b, e) => {
    e.stopPropagation();
    onSelectBurst(b);
    navigate('/detail');
  };

  const handleOpenAudit = (b, e) => {
    e.stopPropagation();
    setAuditingBurst(b);
  };

  return (
    <div className="page-container" style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      {/* Page Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px', flexShrink: 0 }}>
        <div>
          <h1 style={{ fontSize: '20px', margin: 0 }}>Identified Solar Burst Catalog</h1>
          <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
            Physical profile parameters, classifications, uncertainties, and auditable decisions.
          </div>
        </div>

        <div style={{ display: 'flex', gap: '10px' }}>
          <button className="btn btn-secondary btn-sm" onClick={() => navigate('/export')}>
            <Download size={14} />
            <span>Export Catalogs</span>
          </button>
        </div>
      </div>

      {/* Filter Bar */}
      <div className="card" style={{ padding: '12px 16px', marginBottom: '16px', flexShrink: 0 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '20px', flexWrap: 'wrap' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '13px' }}>
            <Filter size={15} className="text-muted" />
            <span className="text-muted">Decision:</span>
            <select value={decisionFilter} onChange={(e) => setDecisionFilter(e.target.value)} style={{ padding: '4px 8px' }}>
              <option value="all">All Decisions</option>
              <option value="accepted">Accepted Only</option>
              <option value="review">Review Only</option>
              <option value="rejected">Rejected Only</option>
            </select>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '13px' }}>
            <span className="text-muted">Reliability:</span>
            <select value={reliabilityFilter} onChange={(e) => setReliabilityFilter(e.target.value)} style={{ padding: '4px 8px' }}>
              <option value="all">All Statuses</option>
              <option value="reliable">Reliable Fits</option>
              <option value="poor_fit">Poor Fits</option>
              <option value="truncated">Truncated / Gaps</option>
            </select>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '13px' }}>
            <span className="text-muted">Min Peak SNR:</span>
            <input
              type="number"
              min="0"
              style={{ width: '70px', padding: '4px 8px' }}
              value={minSnr}
              onChange={(e) => setMinSnr(parseFloat(e.target.value) || 0)}
            />
          </div>

          <div style={{ marginLeft: 'auto', fontSize: '12.5px', color: 'var(--text-muted)' }}>
            Showing <strong>{filtered.length}</strong> of <strong>{bursts.length}</strong> events
          </div>
        </div>
      </div>

      {/* Catalog Table */}
      <div className="table-container" style={{ flex: 1, overflowY: 'auto' }}>
        <table className="scientific-table">
          <thead>
            <tr>
              <th>Burst ID</th>
              <th>Peak Time (UTC)</th>
              <th>Net Peak</th>
              <th>Duration (s)</th>
              <th>Decay &tau; (s)</th>
              <th>Fluence</th>
              <th>Peak SNR</th>
              <th>Morphology</th>
              <th>Reliability</th>
              <th>Red. &chi;&sup2;</th>
              <th>ML Score</th>
              <th>Decision</th>
              <th style={{ textAlign: 'right' }}>Actions</th>
            </tr>
          </thead>
          <tbody>
            {filtered.length > 0 ? (
              filtered.map((b) => {
                const isSelected = selectedBurst?.burst_id === b.burst_id;
                return (
                  <tr
                    key={b.burst_id}
                    onClick={() => handleRowClick(b)}
                    className={isSelected ? 'selected' : ''}
                    style={{ cursor: 'pointer' }}
                  >
                    <td className="mono font-bold text-cyan">{b.burst_id}</td>
                    <td className="mono">{b.peak_time_iso}</td>
                    <td className="mono font-bold text-green">{b.net_peak?.toFixed(2)}</td>
                    <td className="mono">{b.duration_s?.toFixed(1)}</td>
                    <td className="mono">{b.decay_tau_s?.toFixed(1)}</td>
                    <td className="mono">{b.fluence?.toFixed(1)}</td>
                    <td className="mono">{b.peak_snr?.toFixed(1)}</td>
                    <td>
                      <span className={`badge ${b.morphology_class.includes('Fast') ? 'badge-cyan' : 'badge-muted'}`}>
                        {b.morphology_class}
                      </span>
                    </td>
                    <td>
                      <span className={`badge ${b.reliability === 'reliable' ? 'badge-green' : 'badge-amber'}`}>
                        {b.reliability}
                      </span>
                    </td>
                    <td className="mono">{b.reduced_chi2?.toFixed(2)}</td>
                    <td className="mono font-bold">{b.ml_score?.toFixed(2)}</td>
                    <td>
                      <span className={`badge ${b.decision === 'accepted' ? 'badge-green' : (b.decision === 'review' ? 'badge-amber' : 'badge-red')}`}>
                        {b.decision.toUpperCase()}
                      </span>
                    </td>
                    <td style={{ textAlign: 'right' }}>
                      <button
                        className="btn btn-secondary btn-sm"
                        style={{ marginRight: '6px' }}
                        onClick={(e) => handleOpenAudit(b, e)}
                        title="Audit / Approve / Reject"
                      >
                        <ShieldCheck size={13} />
                        <span>Audit</span>
                      </button>
                      <button
                        className="btn btn-primary btn-sm"
                        onClick={(e) => handleInspectDetail(b, e)}
                        title="View Detailed Fit"
                      >
                        <span>Inspect</span>
                        <ArrowRight size={13} />
                      </button>
                    </td>
                  </tr>
                );
              })
            ) : (
              <tr>
                <td colSpan="13" style={{ textAlign: 'center', padding: '36px', color: 'var(--text-dim)' }}>
                  No bursts match current filter criteria.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {/* Audit Modal */}
      {auditingBurst && (
        <AuditModal
          burst={auditingBurst}
          onClose={() => setAuditingBurst(null)}
          onSave={onUpdateAudit}
        />
      )}
    </div>
  );
}
