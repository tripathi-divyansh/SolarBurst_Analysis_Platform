import React from 'react';
import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  Upload,
  Activity,
  Sliders,
  TableProperties,
  ZoomIn,
  Waves,
  BrainCircuit,
  BarChart3,
  FileDown,
  BookOpen,
  Sun
} from 'lucide-react';

const NAV_ITEMS = [
  { path: '/', label: 'Dashboard', icon: LayoutDashboard },
  { path: '/upload', label: 'Upload & Inspect', icon: Upload },
  { path: '/explorer', label: 'Light-Curve Explorer', icon: Activity },
  { path: '/controls', label: 'Detection Controls', icon: Sliders },
  { path: '/catalog', label: 'Burst Catalog', icon: TableProperties },
  { path: '/detail', label: 'Event Detail', icon: ZoomIn },
  { path: '/wavelet', label: 'Wavelet Analysis', icon: Waves },
  { path: '/models', label: 'Model Training', icon: BrainCircuit },
  { path: '/distributions', label: 'Distributions', icon: BarChart3 },
  { path: '/export', label: 'Export & Report', icon: FileDown },
  { path: '/help', label: 'Methods & Help', icon: BookOpen },
];

export default function Navigation() {
  return (
    <aside className="sidebar">
      <div className="sidebar-logo">
        <Sun className="text-cyan" size={24} />
        <div>
          <h2>SolarBurst</h2>
          <div style={{ fontSize: '10.5px', color: 'var(--text-dim)', letterSpacing: '0.04em' }}>
            ISRO XSM PLATFORM
          </div>
        </div>
      </div>

      <nav className="sidebar-nav">
        {NAV_ITEMS.map((item) => {
          const Icon = item.icon;
          return (
            <NavLink
              key={item.path}
              to={item.path}
              className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
            >
              <Icon size={17} />
              <span>{item.label}</span>
            </NavLink>
          );
        })}
      </nav>

      <div className="sidebar-footer">
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: 'var(--accent-green)' }}>
          <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: 'var(--accent-green)', display: 'inline-block' }}></span>
          <span>Engine Active</span>
        </div>
        <div style={{ marginTop: '4px' }}>CH2 XSM v1.5 • Inter IIT 10.0</div>
      </div>
    </aside>
  );
}
