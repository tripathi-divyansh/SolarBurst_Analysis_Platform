import React from 'react';
import { BookOpen, ShieldAlert, Award, Layers, Cpu, Compass } from 'lucide-react';

export default function MethodsHelp() {
  return (
    <div className="page-container" style={{ maxWidth: '950px', margin: '0 auto', lineHeight: '1.6' }}>
      <div style={{ marginBottom: '24px' }}>
        <h1 style={{ fontSize: '24px', marginBottom: '4px' }}>Scientific Methods & Platform Guide</h1>
        <p style={{ color: 'var(--text-muted)', margin: 0, fontSize: '13px' }}>
          Authoritative scientific documentation for the Chandrayaan-2 Solar X-ray Monitor (XSM) Burst Analysis Engine.
        </p>
      </div>

      {/* Scientific Safeguard Callout */}
      <div style={{ background: 'rgba(239, 68, 68, 0.08)', border: '1px solid rgba(239, 68, 68, 0.3)', padding: '16px', borderRadius: '8px', marginBottom: '24px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px' }}>
          <ShieldAlert size={18} className="text-red" />
          <strong style={{ color: '#ef4444', fontSize: '14px' }}>Critical Scientific Safeguard: GOES Solar Flare Classification</strong>
        </div>
        <p style={{ fontSize: '12.5px', color: 'var(--text-main)', margin: 0 }}>
          Standard GOES flare classes (A, B, C, M, X) refer specifically to peak soft X-ray solar irradiance in the <strong>0.1–0.8 nm (1.5–12.4 keV)</strong> band measured in <strong>W/m²</strong>. <strong>Never apply GOES threshold classes directly to raw XSM counts/s or uncalibrated flux.</strong> SolarBurst strictly reports an uncalibrated count rate tier unless calibrated GOES-band energy flux with verified provenance is supplied.
        </p>
      </div>

      {/* 1. Ingestion & Time */}
      <div className="card">
        <h3 style={{ fontSize: '16px', display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--accent-cyan)' }}>
          <Compass size={18} />
          <span>1. XSM Mission Time & Observational Ingestion</span>
        </h3>
        <p style={{ fontSize: '13px', color: 'var(--text-main)' }}>
          Chandrayaan-2 XSM uses Mission Elapsed Time (MET) defined as continuous seconds elapsed from <strong>2017-01-01 00:00:00 UTC</strong>. The engine performs high-precision conversions to UTC ISO-8601 timestamps and Modified Julian Date (MJD) via Astropy. Observational GTIs (Good Time Intervals) and beryllium (Be) filter status flags are retained alongside raw data arrays.
        </p>
      </div>

      {/* 2. Preprocessing & Baseline */}
      <div className="card">
        <h3 style={{ fontSize: '16px', display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--accent-cyan)' }}>
          <Layers size={18} />
          <span>2. Iterative Event Masking & AsLS Baseline</span>
        </h3>
        <p style={{ fontSize: '13px', color: 'var(--text-main)' }}>
          Solar background variability is estimated using an Asymmetric Least Squares (AsLS) smoother with penalization parameter &lambda; and asymmetry weight p = 0.01:
        </p>
        <div className="inset-panel" style={{ fontFamily: 'monospace', fontSize: '12.5px', color: 'var(--accent-cyan)', textAlign: 'center', margin: '12px 0' }}>
          min_b &sum; w_i (y_i - b_i)² + &lambda; &sum; (&Delta;² b_i)²
        </div>
        <p style={{ fontSize: '13px', color: 'var(--text-main)' }}>
          Provisional excess events (z &gt; 3.0&sigma;) are identified, expanded with tail margins, and masked. The baseline is then iteratively refitted on unmasked quiet background points.
        </p>
      </div>

      {/* 3. Hybrid Proposals */}
      <div className="card">
        <h3 style={{ fontSize: '16px', display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--accent-cyan)' }}>
          <Cpu size={18} />
          <span>3. Multiscale Candidate Proposals & Tree Scorer</span>
        </h3>
        <p style={{ fontSize: '13px', color: 'var(--text-main)' }}>
          Proposals are formed through the <strong>union</strong> of three complementary generators:
        </p>
        <ul style={{ fontSize: '13px', color: 'var(--text-main)', paddingLeft: '20px' }}>
          <li><strong>Matched Filter Bank:</strong> Correlates normalized Gaussian-rise / exponential-decay templates across rise timescales (4–256s) and decay timescales (16–4096s).</li>
          <li><strong>Continuous Wavelet Ridges:</strong> Mexican-hat / Ricker wavelets mapping multiscale localized energy in physical seconds.</li>
          <li><strong>Smoothed Derivative Onset:</strong> Detects fast impulsive onsets followed by sustained post-peak excess.</li>
        </ul>
        <p style={{ fontSize: '13px', color: 'var(--text-main)' }}>
          Proposals are deduplicated via hysteresis growth (seed ~4.0&sigma;, grow ~1.8&sigma;). 24 standardized features are extracted and scored using a Random Forest baseline or XGBoost challenger.
        </p>
      </div>

      {/* 4. Physical Fitting */}
      <div className="card">
        <h3 style={{ fontSize: '16px', display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--accent-cyan)' }}>
          <Award size={18} />
          <span>4. Gaussian Heating / Exponential Cooling Profile Fitting</span>
        </h3>
        <p style={{ fontSize: '13px', color: 'var(--text-main)' }}>
          Following Gryciuk et al. (2017) and Aschwanden & Freeland (2012), physical flare light curves are modeled as a brief Gaussian thermal heating pulse convolved with an exponential cooling response:
        </p>
        <div className="inset-panel" style={{ fontFamily: 'monospace', fontSize: '12.5px', color: 'var(--accent-green)', textAlign: 'center', margin: '12px 0' }}>
          s(t) = &int; H_0 exp[-(u - &mu;)² / (2 &sigma;_h²)] exp[-(t - u) / &tau;_c] du
        </div>
        <p style={{ fontSize: '13px', color: 'var(--text-main)' }}>
          Implemented via bin-averaged exponentially modified Gaussian (EMG) formulation with bounded least squares and Jacobian/bootstrap parameter uncertainty propagation.
        </p>
      </div>
    </div>
  );
}
