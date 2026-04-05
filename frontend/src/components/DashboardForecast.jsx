import React, { useState, useEffect } from 'react';
import { Skeleton, Card, ErrorBanner } from './ui';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { Activity, Target, Zap } from 'lucide-react';

// Custom Tooltip for Obsidian Theme
const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    return (
      <div className="custom-recharts-tooltip">
        <div className="label">{label}</div>
        <div className="value">{payload[0].value}</div>
      </div>
    );
  }
  return null;
};

export default function DashboardForecast({ state, apiBase }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(false);

  useEffect(() => {
    if (!state) return;
    async function fetchForecast() {
      setLoading(true);
      try {
        const res = await fetch(`${apiBase}/forecast?state=${encodeURIComponent(state)}`);
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
    fetchForecast();
  }, [state, apiBase]);

  if (loading) {
    return (
      <div className="grid grid-cols-2 gap-4 mt-4">
        <Skeleton style={{ height: '350px' }} />
        <Skeleton style={{ height: '350px' }} />
      </div>
    );
  }

  if (!data) return <div className="p-4 text-center text-tertiary">No intelligence data streaming.</div>;

  const comparisonData = [
    { name: 'Linear', value: data.predictions.linear },
    { name: 'Ridge', value: data.predictions.ridge },
    { name: 'Random F', value: data.predictions.random_forest },
    { name: 'Ensemble', value: data.predictions.ensemble },
    { name: 'Actual', value: data.predictions.actual },
  ];

  const shapData = data.shap.slice(0, 7).reverse();

  return (
    <div className="mt-4 flex-col gap-4" style={{ animation: 'fadeIn var(--transition-smooth)' }}>
      {error && <ErrorBanner />}
      
      {/* Metric Cards */}
      <div className="grid grid-cols-3 gap-4 mb-6">
        <Card className="card-metric border-subtle">
          <div className="flex items-center gap-2 mb-2 text-secondary">
            <Activity size={18} className="text-warning" />
            <span className="text-sm font-semibold uppercase tracking-wide">Ensemble Forecast</span>
          </div>
          <span className="text-4xl font-bold text-main">{data.predictions.ensemble}</span>
        </Card>
        <Card className="card-metric border-subtle">
          <div className="flex items-center gap-2 mb-2 text-secondary">
            <Target size={18} className="text-accent" />
            <span className="text-sm font-semibold uppercase tracking-wide">Actual Rate</span>
          </div>
          <span className="text-4xl font-bold text-main">{data.predictions.actual}</span>
        </Card>
        <Card className="card-metric border-subtle">
          <div className="flex items-center gap-2 mb-2 text-secondary">
            <Zap size={18} className="text-error" />
            <span className="text-sm font-semibold uppercase tracking-wide">Absolute Error</span>
          </div>
          <span className="text-4xl font-bold text-main">{data.predictions.absolute_error}</span>
        </Card>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-2 gap-4">
        {/* Model Comparison */}
        <div className="card" style={{ padding: '0', overflow: 'hidden' }}>
          <div style={{ padding: 'var(--space-3) var(--space-4)', borderBottom: '1px solid var(--border-subtle)' }}>
            <h3 className="text-lg font-bold">Model Comparison</h3>
          </div>
          <div style={{ height: '300px', padding: 'var(--space-3)' }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={comparisonData}>
                <defs>
                  <linearGradient id="colorValue" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="var(--accent-primary)" stopOpacity={1}/>
                    <stop offset="95%" stopColor="var(--accent-cyan)" stopOpacity={0.6}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--border-subtle)" />
                <XAxis dataKey="name" tick={{fontSize: 12, fill: 'var(--text-secondary)'}} axisLine={false} tickLine={false} dy={10} />
                <YAxis tick={{fontSize: 12, fill: 'var(--text-secondary)'}} axisLine={false} tickLine={false} dx={-10} />
                <Tooltip content={<CustomTooltip />} cursor={{fill: 'var(--bg-surface-hover)'}} />
                <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                  {comparisonData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.name === 'Actual' ? 'var(--text-main)' : 'url(#colorValue)'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* SHAP Chart */}
        <div className="card" style={{ padding: '0', overflow: 'hidden' }}>
          <div style={{ padding: 'var(--space-3) var(--space-4)', borderBottom: '1px solid var(--border-subtle)' }}>
            <h3 className="text-lg font-bold">SHAP Feature Impact</h3>
          </div>
          <div style={{ height: '300px', padding: 'var(--space-3) var(--space-3) var(--space-3) 0' }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={shapData} layout="vertical" margin={{ left: 80, right: 20 }}>
                <defs>
                  <linearGradient id="shapValue" x1="0" y1="0" x2="1" y2="0">
                    <stop offset="0%" stopColor="var(--accent-cyan)" stopOpacity={0.8}/>
                    <stop offset="100%" stopColor="var(--accent-primary)" stopOpacity={1}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="var(--border-subtle)" />
                <XAxis type="number" tick={{fontSize: 12, fill: 'var(--text-secondary)'}} axisLine={false} tickLine={false} />
                <YAxis dataKey="feature" type="category" tick={{fontSize: 11, fill: 'var(--text-secondary)'}} axisLine={false} tickLine={false} dx={-5} />
                <Tooltip content={<CustomTooltip />} cursor={{fill: 'var(--bg-surface-hover)'}} />
                <Bar dataKey="value" fill="url(#shapValue)" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}
