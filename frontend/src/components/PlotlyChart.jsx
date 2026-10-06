import React from 'react';
import Plot from 'react-plotly.js';
import { useTheme } from '../context/ThemeContext';

export default function PlotlyChart({ data, layout, style, config: customConfig, onSelected, onClick }) {
  const { isLight } = useTheme();

  const paperBg = isLight ? '#ffffff' : '#0f172a';
  const plotBg = isLight ? '#f8fafc' : '#090e1c';
  const textColor = isLight ? '#334155' : '#cbd5e1';
  const gridColor = isLight ? '#e2e8f0' : '#1e293b';
  const lineDim = isLight ? '#cbd5e1' : '#334155';

  const defaultLayout = {
    autosize: true,
    paper_bgcolor: layout?.paper_bgcolor || paperBg,
    plot_bgcolor: layout?.plot_bgcolor || plotBg,
    margin: { l: 55, r: 25, t: 35, b: 45, ...layout?.margin },
    font: {
      color: textColor,
      family: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
      size: 11,
      ...layout?.font
    },
    xaxis: {
      gridcolor: gridColor,
      zerolinecolor: lineDim,
      tickcolor: lineDim,
      ...layout?.xaxis
    },
    yaxis: {
      gridcolor: gridColor,
      zerolinecolor: lineDim,
      tickcolor: lineDim,
      ...layout?.yaxis
    },
    ...layout
  };

  if (defaultLayout.title && typeof defaultLayout.title === 'object') {
    defaultLayout.title = {
      ...defaultLayout.title,
      font: {
        color: isLight ? '#0f172a' : '#ffffff',
        size: defaultLayout.title.font?.size || 13,
        ...defaultLayout.title.font
      }
    };
    if (isLight) {
      defaultLayout.title.font.color = '#0f172a';
    }
  }

  // Ensure high-contrast trace elements in light mode
  const themedData = isLight ? data?.map(trace => {
    const updated = { ...trace };
    if (updated.line?.color === '#fff' || updated.line?.color === 'white' || updated.line?.color?.includes('255, 255, 255')) {
      updated.line = { ...updated.line, color: 'rgba(15, 23, 42, 0.7)' };
    }
    if (updated.marker?.color === '#cbd5e1') {
      updated.marker = { ...updated.marker, color: '#475569' };
    }
    return updated;
  }) : data;

  const defaultConfig = {
    responsive: true,
    displayModeBar: true,
    displaylogo: false,
    modeBarButtonsToRemove: ['lasso2d'],
    toImageButtonOptions: {
      format: 'png',
      filename: 'solarburst_plot',
      height: 600,
      width: 1000,
      scale: 2
    },
    ...customConfig
  };

  return (
    <div style={{ width: '100%', height: '100%', minHeight: '300px', ...style }}>
      <Plot
        data={themedData}
        layout={defaultLayout}
        config={defaultConfig}
        useResizeHandler={true}
        style={{ width: '100%', height: '100%' }}
        onSelected={onSelected}
        onClick={onClick}
      />
    </div>
  );
}
