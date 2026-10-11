import React from 'react';
import { useApi } from '../hooks/useApi';
import { Skeleton, ErrorBanner } from './ui';

export default function DashboardRisk({ apiBase }) {
  const { data, loading, error } = useApi(`${apiBase}/risk-analysis`);

  if (loading) return <Skeleton style={{ height: '400px', marginTop: '16px' }} />;
  if (error) return <ErrorBanner message={error} />;
  if (!data) return <div className="text-secondary p-4">No risk data available.</div>;

  const rankings = data.rankings || [];

  return (
    <div className="mt-4 flex-col gap-4" style={{ animation: 'fadeIn var(--transition-smooth)' }}>
      <div className="card" style={{ padding: '0', overflow: 'hidden' }}>
        <div style={{ padding: 'var(--space-3) var(--space-4)', borderBottom: '1px solid var(--border-subtle)', background: 'rgba(255,255,255,0.02)' }}>
          <h3 className="text-lg font-bold text-main">Risk Classification Backtest and Ranking</h3>
          <p className="text-sm text-secondary mt-1">Relative risk class (terciles of the 2023 rate) predicted from 2019–2022 features by a model that did not train on that state. Out-of-fold backtest on unverified sample data, 12 states.</p>
        </div>
        <div style={{ overflowX: 'auto', padding: 'var(--space-2)' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: 'var(--font-size-sm)' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid var(--border-strong)', color: 'var(--text-tertiary)' }}>
                <th style={{ padding: 'var(--space-3) var(--space-4)', fontWeight: 'var(--font-weight-bold)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Region / State</th>
                <th style={{ padding: 'var(--space-3) var(--space-4)', fontWeight: 'var(--font-weight-bold)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Predicted Class</th>
                <th style={{ padding: 'var(--space-3) var(--space-4)', fontWeight: 'var(--font-weight-bold)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Observed Class</th>
                <th style={{ padding: 'var(--space-3) var(--space-4)', fontWeight: 'var(--font-weight-bold)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Risk Score (1–3)</th>
                <th style={{ padding: 'var(--space-3) var(--space-4)', fontWeight: 'var(--font-weight-bold)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Observed 2023 Rate</th>
                <th style={{ padding: 'var(--space-3) var(--space-4)', fontWeight: 'var(--font-weight-bold)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Change 2021→22</th>
              </tr>
            </thead>
            <tbody>
              {rankings.map((row, i) => (
                <tr key={row.state} style={{ 
                  borderBottom: '1px solid var(--border-subtle)', 
                  backgroundColor: i % 2 === 0 ? 'transparent' : 'rgba(255,255,255,0.01)',
                  transition: 'background-color var(--transition-fast)'
                }}
                className="hover:bg-[rgba(255,255,255,0.04)]"
                >
                  <td style={{ padding: 'var(--space-3) var(--space-4)', fontWeight: 'var(--font-weight-medium)', color: 'var(--text-main)' }}>{row.state}</td>
                  <td style={{ padding: 'var(--space-3) var(--space-4)' }}>
                    <span style={{
                      padding: '4px 10px',
                      borderRadius: '12px',
                      fontSize: '11px',
                      fontWeight: '800',
                      letterSpacing: '0.05em',
                      textTransform: 'uppercase',
                      backgroundColor: row.Predicted_Risk_Class === 'high' ? 'rgba(248, 113, 113, 0.15)' : row.Predicted_Risk_Class === 'medium' ? 'rgba(251, 191, 36, 0.15)' : 'rgba(16, 185, 129, 0.15)',
                      color: row.Predicted_Risk_Class === 'high' ? '#fca5a5' : row.Predicted_Risk_Class === 'medium' ? '#fcd34d' : '#6ee7b7',
                      border: `1px solid ${row.Predicted_Risk_Class === 'high' ? 'rgba(248, 113, 113, 0.3)' : row.Predicted_Risk_Class === 'medium' ? 'rgba(251, 191, 36, 0.3)' : 'rgba(16, 185, 129, 0.3)'}`
                    }}>
                      {row.Predicted_Risk_Class}
                    </span>
                  </td>
                  <td style={{ padding: 'var(--space-3) var(--space-4)', color: 'var(--text-secondary)', textTransform: 'uppercase', fontSize: '11px' }}>{row.Actual_Risk_Class}</td>
                  <td style={{ padding: 'var(--space-3) var(--space-4)', color: 'var(--text-secondary)' }}><strong className="text-main">{row.Risk_Score?.toFixed(2)}</strong></td>
                  <td style={{ padding: 'var(--space-3) var(--space-4)', color: 'var(--text-secondary)' }}>{row.rate_2023?.toFixed(2)}</td>
                  <td style={{ padding: 'var(--space-3) var(--space-4)', color: row.momentum_21_22 > 0 ? 'var(--error)' : 'var(--success)' }}>
                    {row.momentum_21_22 > 0 ? '+' : ''}{row.momentum_21_22?.toFixed(2)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
