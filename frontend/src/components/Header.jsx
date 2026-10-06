import React from 'react';
import { Database, Play, CheckCircle2, AlertCircle, Loader2, Sun, Moon } from 'lucide-react';
import { useTheme } from '../context/ThemeContext';

export default function Header({ activeDataset, activeJob, onLoadDemo, isDemoLoading }) {
  const { theme, toggleTheme, isLight } = useTheme();

  return (
    <header className="top-header">
      <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Database size={16} className="text-muted" />
          <span style={{ fontSize: '13px', color: 'var(--text-muted)' }}>Active Dataset:</span>
          {activeDataset ? (
            <span style={{ fontSize: '13.5px', fontWeight: 600, color: 'var(--heading-color)' }}>
              {activeDataset.name || activeDataset.filename}
              {activeDataset.metadata?.is_synthetic_demonstration && (
                <span className="badge badge-amber" style={{ marginLeft: '8px', fontSize: '10px' }}>
                  SYNTHETIC DEMO
                </span>
              )}
            </span>
          ) : (
            <span style={{ fontSize: '13px', color: 'var(--text-dim)', fontStyle: 'italic' }}>
              None selected
            </span>
          )}
        </div>

        {activeJob && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', paddingLeft: '16px', borderLeft: '1px solid var(--border-subtle)' }}>
            {activeJob.status === 'running' && (
              <>
                <Loader2 size={15} className="animate-spin text-cyan" />
                <span style={{ fontSize: '12.5px', color: 'var(--accent-cyan)' }}>
                  {activeJob.message || 'Processing...'} ({activeJob.progress}%)
                </span>
              </>
            )}
            {activeJob.status === 'completed' && (
              <>
                <CheckCircle2 size={15} className="text-green" />
                <span style={{ fontSize: '12.5px', color: 'var(--accent-green)' }}>Analysis Ready</span>
              </>
            )}
            {activeJob.status === 'failed' && (
              <>
                <AlertCircle size={15} className="text-red" />
                <span style={{ fontSize: '12.5px', color: 'var(--accent-red)' }}>Analysis Failed</span>
              </>
            )}
          </div>
        )}
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
        {/* Light/Dark Mode Switcher */}
        <button
          className="theme-toggle-btn"
          onClick={toggleTheme}
          title={isLight ? 'Switch to Observatory Dark Mode' : 'Switch to Clean Light Mode'}
          id="theme-mode-toggle"
        >
          <span className="icon">
            {isLight ? <Moon size={14} className="text-cyan" /> : <Sun size={14} className="text-amber" />}
          </span>
          <span>{isLight ? 'Dark Mode' : 'Light Mode'}</span>
        </button>

        <button
          className="btn btn-secondary btn-sm"
          onClick={onLoadDemo}
          disabled={isDemoLoading}
        >
          {isDemoLoading ? <Loader2 size={14} className="animate-spin" /> : <Play size={14} />}
          <span>{isDemoLoading ? 'Loading Demo...' : 'Load XSM Demo'}</span>
        </button>
      </div>
    </header>
  );
}
