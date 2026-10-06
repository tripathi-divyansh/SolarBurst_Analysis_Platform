import React, { useState } from 'react';
import { X, Check, AlertTriangle, ShieldCheck } from 'lucide-react';

export default function AuditModal({ burst, onClose, onSave }) {
  const [decision, setDecision] = useState(burst?.decision || 'accepted');
  const [notes, setNotes] = useState(burst?.audit_notes || '');
  const [isSubmitting, setIsSubmitting] = useState(false);

  if (!burst) return null;

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsSubmitting(true);
    try {
      await onSave(burst.id, {
        decision,
        audit_status: decision === 'accepted' ? 'approved' : 'reviewed',
        audit_notes: notes,
      });
      onClose();
    } catch (err) {
      alert('Error updating audit: ' + err.message);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <ShieldCheck size={20} className="text-cyan" />
            <h3 style={{ margin: 0, fontSize: '18px' }}>Audit Event: {burst.burst_id}</h3>
          </div>
          <button className="btn btn-secondary btn-sm" onClick={onClose}>
            <X size={16} />
          </button>
        </div>

        <div className="inset-panel" style={{ marginBottom: '16px', fontSize: '12px' }}>
          <div><strong>Peak Time:</strong> {burst.peak_time_iso}</div>
          <div><strong>Net Peak:</strong> {burst.net_peak?.toFixed(2)} count/s | <strong>SNR:</strong> {burst.peak_snr?.toFixed(1)}</div>
          <div><strong>ML Score:</strong> {burst.ml_score?.toFixed(2)} | <strong>Fit &chi;&sup2;:</strong> {burst.reduced_chi2?.toFixed(2)}</div>
        </div>

        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label className="form-label">Classification Decision</label>
            <select
              style={{ width: '100%' }}
              value={decision}
              onChange={(e) => setDecision(e.target.value)}
            >
              <option value="accepted">Accepted (Verified Real Burst)</option>
              <option value="review">Review (Ambiguous / Incomplete Context)</option>
              <option value="rejected">Rejected (Instrument Artifact / Fluctuation)</option>
            </select>
          </div>

          <div className="form-group">
            <label className="form-label">Scientific Audit Notes / Rationale</label>
            <textarea
              style={{ width: '100%', minHeight: '80px' }}
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              placeholder="e.g., Confirmed with impulsive rise and clear cooling tail; background uncontaminated."
            />
          </div>

          <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '10px', marginTop: '20px' }}>
            <button type="button" className="btn btn-secondary" onClick={onClose}>
              Cancel
            </button>
            <button type="submit" className="btn btn-primary" disabled={isSubmitting}>
              {isSubmitting ? 'Saving...' : 'Save Audit Decision'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
