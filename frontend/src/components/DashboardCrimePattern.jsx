import React from 'react';
import { useApi } from '../hooks/useApi';
import { Skeleton, Card, ErrorBanner } from './ui';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';

const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    return (
      <div className="custom-recharts-tooltip">
        <div className="label">{label}</div>
        {payload.map((p, i) => (
          <div key={i} className="value" style={{ color: p.color }}>
            {p.name}: {(p.value * 100).toFixed(1)}%
          </div>
        ))}
      </div>
    );
  }
  return null;
};

export default function DashboardCrimePattern({ state, apiBase }) {
  const url = state ? `${apiBase}/crime-pattern?state=${encodeURIComponent(state)}` : null;
  const { data, loading, error } = useApi(url);

  if (!state) return <div className="text-secondary p-4">Select a state in the sidebar to see its crime pattern.</div>;
  if (loading) return <Skeleton style={{ height: '400px', marginTop: '16px' }} />;
  if (error) return <ErrorBanner message={error} />;
  if (!data) return <div className="text-secondary p-4">No crime pattern data available for this state.</div>;

  // Top 5 = the five crime types with the highest average share across all years shown.
  const years = data.trend || [];
  const avgShare = {};
  years.forEach(row => {
    Object.keys(row).filter(k => k !== 'year').forEach(k => {
      avgShare[k] = (avgShare[k] || 0) + row[k] / years.length;
    });
  });
  const topTrendKeys = Object.keys(avgShare).sort((a, b) => avgShare[b] - avgShare[a]).slice(0, 5);
  const colors = ['#06B6D4', '#3B82F6', '#8B5CF6', '#EC4899', '#10B981'];

  return (
    <div className="mt-4 flex-col gap-4" style={{ animation: 'fadeIn var(--transition-smooth)' }}>
      {data.prediction && (
        <Card className="mb-6 border-subtle" style={{ background: 'linear-gradient(90deg, rgba(59, 130, 246, 0.1) 0%, transparent 100%)' }}>
          <div className="flex justify-between items-center">
            <div>
              <span className="text-sm font-semibold text-accent uppercase tracking-widest block mb-1">Held-out Prediction: Dominant 2023</span>
              <h2 className="text-3xl font-bold text-main">{data.prediction.label}</h2>
            </div>
            {data.actual_2023 && (
              <div className="text-right">
                <span className="text-sm font-semibold text-secondary uppercase tracking-widest block mb-1">Observed Dominant 2023</span>
                <h2 className="text-3xl font-bold text-text-secondary">{data.actual_2023[0]?.type}</h2>
              </div>
            )}
          </div>
        </Card>
      )}

      {data.prediction && (
        <p className="text-xs text-tertiary mb-4">
          Prediction made by a model that did not train on {state}. Dominant-crime classification is unreliable here: only 12
          states, imbalanced classes (one has a single state). Out-of-fold accuracy: Random Forest {(data.metrics?.rf?.oof_accuracy * 100).toFixed(0)}%,
          Logistic {(data.metrics?.logit?.oof_accuracy * 100).toFixed(0)}%.
        </p>
      )}

      <div className="card" style={{ padding: '0', overflow: 'hidden' }}>
        <div style={{ padding: 'var(--space-3) var(--space-4)', borderBottom: '1px solid var(--border-subtle)' }}>
          <h3 className="text-lg font-bold">Crime Type Share Trend (Top 5 by average share)</h3>
        </div>
        <div style={{ height: '450px', padding: 'var(--space-4)' }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data.trend}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--border-subtle)" />
              <XAxis dataKey="year" tick={{fontSize: 12, fill: 'var(--text-secondary)'}} axisLine={false} tickLine={false} dy={10} />
              <YAxis tick={{fontSize: 12, fill: 'var(--text-secondary)'}} axisLine={false} tickLine={false} dx={-10} tickFormatter={(val) => `${(val*100).toFixed(0)}%`} />
              <Tooltip content={<CustomTooltip />} cursor={{ stroke: 'var(--border-strong)', strokeWidth: 1, strokeDasharray: '4 4' }} />
              <Legend wrapperStyle={{ fontSize: '12px', color: 'var(--text-secondary)', paddingTop: '20px' }} />
              {topTrendKeys.map((key, i) => (
                <Line 
                  key={key} type="monotone" dataKey={key} 
                  stroke={colors[i]} strokeWidth={3} 
                  dot={{r: 4, fill: 'var(--bg-surface)', strokeWidth: 2}} 
                  activeDot={{r: 6, strokeWidth: 0, fill: colors[i], stroke: 'white'}} 
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
