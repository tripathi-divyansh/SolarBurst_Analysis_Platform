import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { UploadCloud, FileText, CheckCircle2, AlertCircle, ArrowRight, Play, Database } from 'lucide-react';
import { api } from '../api';

export default function UploadInspect({ onDatasetLoaded, onStartAnalysis }) {
  const navigate = useNavigate();
  const [file, setFile] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [inspection, setInspection] = useState(null);
  const [uploadedDataset, setUploadedDataset] = useState(null);
  const [timeCol, setTimeCol] = useState('TIME');
  const [valueCol, setValueCol] = useState('RATE');
  const [errorCol, setErrorCol] = useState('ERROR');
  const [timeFormat, setTimeFormat] = useState('xsm_met');
  const [unit, setUnit] = useState('count / s');
  const [errorMsg, setErrorMsg] = useState(null);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setErrorMsg(null);
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    setIsUploading(true);
    setErrorMsg(null);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('name', file.name);

    try {
      const res = await api.uploadDataset(formData);
      setUploadedDataset(res.dataset);
      setInspection(res.inspection);
      onDatasetLoaded(res.dataset);

      // Populate proposed mappings
      if (res.inspection?.proposed_mapping) {
        const pm = res.inspection.proposed_mapping;
        if (pm.time_col) setTimeCol(pm.time_col);
        if (pm.value_col) setValueCol(pm.value_col);
        if (pm.error_col) setErrorCol(pm.error_col);
      }
    } catch (err) {
      setErrorMsg(err.message);
    } finally {
      setIsUploading(false);
    }
  };

  const handleLaunchAnalysis = async () => {
    if (!uploadedDataset) return;
    try {
      await onStartAnalysis(uploadedDataset.id, {
        preset: 'balanced',
        column_mapping: {
          time_col: timeCol,
          value_col: valueCol,
          error_col: errorCol,
          time_format: timeFormat,
          unit: unit
        }
      });
      navigate('/explorer');
    } catch (err) {
      setErrorMsg(err.message);
    }
  };

  return (
    <div className="page-container" style={{ maxWidth: '1000px', margin: '0 auto' }}>
      <div style={{ marginBottom: '24px' }}>
        <h1 style={{ fontSize: '24px', marginBottom: '4px' }}>Upload & Inspect Observation</h1>
        <p style={{ color: 'var(--text-muted)', margin: 0, fontSize: '13px' }}>
          Supports Chandrayaan-2 XSM FITS (.fits, .lc), CSV/TSV, Excel (.xlsx, .xls), and NASA CDF (.cdf)
        </p>
      </div>

      {errorMsg && (
        <div style={{ background: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.3)', padding: '12px 16px', borderRadius: '8px', marginBottom: '20px', color: '#fca5a5', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <AlertCircle size={16} />
          <span>{errorMsg}</span>
        </div>
      )}

      {/* Upload Dropzone */}
      <div className="card" style={{ borderStyle: 'dashed', borderWidth: '2px', borderColor: file ? 'var(--accent-cyan)' : 'var(--border-subtle)', textAlign: 'center', padding: '36px 20px' }}>
        <UploadCloud size={44} className={file ? 'text-cyan' : 'text-muted'} style={{ margin: '0 auto 12px auto' }} />
        <h3 style={{ fontSize: '16px', marginBottom: '6px' }}>
          {file ? file.name : 'Select or drag & drop light curve observation file'}
        </h3>
        <p style={{ color: 'var(--text-dim)', fontSize: '12.5px', marginBottom: '16px' }}>
          Genuine parsing for FITS, gzip compressed FITS, CSV, TSV, XLSX, XLS, and CDF formats.
        </p>
        
        <input
          type="file"
          id="fileInput"
          style={{ display: 'none' }}
          onChange={handleFileChange}
          accept=".fits,.lc,.fit,.gz,.csv,.tsv,.tab,.xlsx,.xls,.cdf"
        />
        
        <div style={{ display: 'flex', justifyContent: 'center', gap: '12px' }}>
          <label htmlFor="fileInput" className="btn btn-secondary">
            <span>Browse Files</span>
          </label>
          <button className="btn btn-primary" onClick={handleUpload} disabled={!file || isUploading}>
            {isUploading ? 'Inspecting Format...' : 'Upload & Inspect'}
          </button>
        </div>
      </div>

      {/* Inspection Results */}
      {inspection && (
        <div className="card">
          <div className="card-title">
            <span>Inspection Summary</span>
            <span className="badge badge-green">FORMAT DETECTED: {inspection.detection?.format?.toUpperCase()}</span>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '12px', marginBottom: '20px', fontSize: '12px' }}>
            <div className="inset-panel">
              <div className="text-muted">Signature</div>
              <div className="mono font-bold" style={{ marginTop: '2px' }}>{inspection.detection?.details}</div>
            </div>
            <div className="inset-panel">
              <div className="text-muted">File Size</div>
              <div className="mono font-bold" style={{ marginTop: '2px' }}>{(uploadedDataset.size_bytes / 1024).toFixed(1)} KB</div>
            </div>
            <div className="inset-panel">
              <div className="text-muted">SHA-256 Provenance</div>
              <div className="mono font-bold" style={{ marginTop: '2px', fontSize: '10px' }}>{inspection.sha256?.slice(0, 16)}...</div>
            </div>
            <div className="inset-panel">
              <div className="text-muted">GTI Extensions</div>
              <div className="mono font-bold" style={{ marginTop: '2px' }}>{inspection.hdus?.find(h => h.name === 'GTI') ? 'Present' : 'Standard Span'}</div>
            </div>
          </div>

          {/* Column Mapping Form */}
          <h4 style={{ fontSize: '14px', marginBottom: '12px' }}>Validate Column and Unit Mappings</h4>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '16px', marginBottom: '20px' }}>
            <div className="form-group">
              <label className="form-label">Time Column</label>
              <input
                type="text"
                style={{ width: '100%' }}
                value={timeCol}
                onChange={(e) => setTimeCol(e.target.value)}
                placeholder="e.g. TIME, MET"
              />
            </div>

            <div className="form-group">
              <label className="form-label">Rate / Value Column</label>
              <input
                type="text"
                style={{ width: '100%' }}
                value={valueCol}
                onChange={(e) => setValueCol(e.target.value)}
                placeholder="e.g. RATE, COUNTS, FLUX"
              />
            </div>

            <div className="form-group">
              <label className="form-label">Error Column (Optional)</label>
              <input
                type="text"
                style={{ width: '100%' }}
                value={errorCol}
                onChange={(e) => setErrorCol(e.target.value)}
                placeholder="e.g. ERROR, STAT_ERR"
              />
            </div>

            <div className="form-group">
              <label className="form-label">Time Scale & Epoch</label>
              <select style={{ width: '100%' }} value={timeFormat} onChange={(e) => setTimeFormat(e.target.value)}>
                <option value="xsm_met">XSM MET (Seconds from 2017-01-01 00:00 UTC)</option>
                <option value="isot">UTC ISO-8601 Strings</option>
                <option value="unix">Unix Timestamps (Seconds)</option>
                <option value="mjd">Modified Julian Date (MJD)</option>
              </select>
            </div>

            <div className="form-group">
              <label className="form-label">Measurement Unit</label>
              <select style={{ width: '100%' }} value={unit} onChange={(e) => setUnit(e.target.value)}>
                <option value="count / s">Count Rate (count / s) [XSM Default]</option>
                <option value="counts">Raw Bin Counts (counts)</option>
                <option value="W / m^2">Calibrated GOES-band Flux (W / m²)</option>
              </select>
            </div>
          </div>

          <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '12px' }}>
            <button className="btn btn-primary" onClick={handleLaunchAnalysis}>
              <span>Start Scientific Analysis Pipeline</span>
              <ArrowRight size={15} />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
