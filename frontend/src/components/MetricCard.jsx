import React from 'react';

export default function MetricCard({ label, value, sub, icon: Icon, color = 'cyan' }) {
  const colorClass = {
    cyan: 'text-cyan',
    green: 'text-green',
    amber: 'text-amber',
    red: 'text-red',
    white: 'text-main',
  }[color] || 'text-cyan';

  return (
    <div className="metric-card">
      <div className="metric-label">
        {Icon && <Icon size={14} className={colorClass} />}
        <span>{label}</span>
      </div>
      <div className={`metric-val ${colorClass}`}>
        {value !== undefined && value !== null ? value : '—'}
      </div>
      {sub && <div className="metric-sub">{sub}</div>}
    </div>
  );
}
