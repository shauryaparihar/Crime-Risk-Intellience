import React, { useState, useEffect } from 'react';
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
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(false);

  useEffect(() => {
    if (!state) return;
    async function fetchPattern() {
      setLoading(true);
      try {
        const res = await fetch(`${apiBase}/crime-pattern?state=${encodeURIComponent(state)}`);
        if (!res.ok) throw new Error();
        const json = await res.json();
        setData(json);
        setError(false);
      } catch (err) {
        console.error(err);
        setError(true);
      } finally {
        setLoading(false);
      }
    }
    fetchPattern();
  }, [state, apiBase]);

  if (loading) return <Skeleton style={{ height: '400px', marginTop: '16px' }} />;
  if (!data) return <div className="text-secondary p-4">No data available</div>;

  const topTrendKeys = data.trend && data.trend.length > 0 ? Object.keys(data.trend[0]).filter(k => k !== 'year').slice(0, 5) : [];
  const colors = ['#06B6D4', '#3B82F6', '#8B5CF6', '#EC4899', '#10B981'];

  return (
    <div className="mt-4 flex-col gap-4" style={{ animation: 'fadeIn var(--transition-smooth)' }}>
      {error && <ErrorBanner />}
      {data.prediction && (
        <Card className="mb-6 border-subtle" style={{ background: 'linear-gradient(90deg, rgba(59, 130, 246, 0.1) 0%, transparent 100%)' }}>
          <div className="flex justify-between items-center">
            <div>
              <span className="text-sm font-semibold text-accent uppercase tracking-widest block mb-1">Predicted Dominant 2023</span>
              <h2 className="text-3xl font-bold text-main">{data.prediction.label}</h2>
            </div>
            {data.actual_2023 && (
              <div className="text-right">
                <span className="text-sm font-semibold text-secondary uppercase tracking-widest block mb-1">Actual Dominant 2023</span>
                <h2 className="text-3xl font-bold text-text-secondary">{data.actual_2023[0]?.type}</h2>
              </div>
            )}
          </div>
        </Card>
      )}

      <div className="card" style={{ padding: '0', overflow: 'hidden' }}>
        <div style={{ padding: 'var(--space-3) var(--space-4)', borderBottom: '1px solid var(--border-subtle)' }}>
          <h3 className="text-lg font-bold">Crime Type Share Trend (Top 5)</h3>
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
