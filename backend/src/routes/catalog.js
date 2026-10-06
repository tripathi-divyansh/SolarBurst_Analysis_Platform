import express from 'express';
import { db } from '../db/index.js';

const router = express.Router();

// Get burst catalog with filtering
router.get('/', async (req, res) => {
  try {
    const { job_id, decision, reliability, min_snr } = req.query;
    let bursts = await db.getBursts(job_id || null);

    if (decision) {
      bursts = bursts.filter(b => b.decision === decision);
    }
    if (reliability) {
      bursts = bursts.filter(b => b.reliability === reliability);
    }
    if (min_snr) {
      const snrVal = parseFloat(min_snr);
      bursts = bursts.filter(b => b.peak_snr >= snrVal);
    }

    res.json(bursts);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Update human audit status and notes
router.patch('/:id/audit', async (req, res) => {
  try {
    const { decision, audit_status, audit_notes } = req.body;
    const updated = await db.updateBurstAudit(req.params.id, {
      decision,
      audit_status: audit_status || (decision === 'accepted' ? 'approved' : 'rejected'),
      audit_notes: audit_notes || ''
    });

    if (!updated) {
      return res.status(404).json({ error: 'Burst record not found' });
    }

    // Add entry to audit trail
    if (db.isPg) {
      await db.pgPool.query(
        'INSERT INTO audit_trail (id, burst_id, action, new_decision, notes) VALUES ($1, $2, $3, $4, $5)',
        [`audit_${Date.now()}`, req.params.id, 'review_action', decision, audit_notes || '']
      );
    } else {
      db.localStore.audit_trail.push({
        id: `audit_${Date.now()}`,
        burst_id: req.params.id,
        action: 'review_action',
        new_decision: decision,
        notes: audit_notes || '',
        timestamp: new Date().toISOString()
      });
      db._saveLocal();
    }

    res.json({ status: 'updated', burst: updated });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

export default router;
