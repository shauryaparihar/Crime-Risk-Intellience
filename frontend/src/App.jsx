import React, { useState, useEffect } from 'react';
import { useApi } from './hooks/useApi';
import './index.css';
import { Card, Select, Tabs, Skeleton, ErrorBanner } from './components/ui';
import {
  BarChart3, TrendingUp, AlertTriangle, Map as MapIcon,
  Shield, Siren, Info,
  Clock, ChevronRight, Server, Database
} from 'lucide-react';
import DashboardForecast from './components/DashboardForecast';
import DashboardCrimePattern from './components/DashboardCrimePattern';
import DashboardRisk from './components/DashboardRisk';
import DashboardMap from './components/DashboardMap';

const API_BASE = import.meta.env.VITE_API_URL
  ? `${import.meta.env.VITE_API_URL.replace(/\/+$/, '')}/api`
  : 'http://localhost:8001/api';

const fmt = (v, digits = 1, scale = 1) =>
  typeof v === 'number' && Number.isFinite(v) ? (v * scale).toFixed(digits) : '—';

function App() {
  const [currentState, setCurrentState] = useState('');
  const [currentTime, setCurrentTime] = useState(new Date());

  const statesRes = useApi(`${API_BASE}/states`);
  const riskRes = useApi(`${API_BASE}/risk-analysis`);
  const metricsRes = useApi(`${API_BASE}/metrics`);

  const states = statesRes.data?.states || [];
  const metrics = metricsRes.data;
  const topRisk = (riskRes.data?.rankings || [])
    .filter(r => r.Predicted_Risk_Class === 'high')
    .slice(0, 3);
  const loading = statesRes.loading;
  const apiError = statesRes.error || riskRes.error || metricsRes.error;

  // Live clock tick
  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  const tabs = [
    {
      id: 'forecast', label: 'Backtest & Analysis', icon: BarChart3,
      content: <DashboardForecast state={currentState} apiBase={API_BASE} />
    },
    {
      id: 'pattern', label: 'Crime Patterns', icon: TrendingUp,
      content: <DashboardCrimePattern state={currentState} apiBase={API_BASE} />
    },
    {
      id: 'risk', label: 'Risk Classification', icon: AlertTriangle,
      content: <DashboardRisk apiBase={API_BASE} />
    },
    {
      id: 'drilldown', label: 'State Map', icon: MapIcon,
      content: <DashboardMap state={currentState} apiBase={API_BASE} />
    }
  ];

  return (
    <div className="app-layout">
      {/* Sidebar Navigation */}
      <aside className="sidebar">
        
        {/* Logo / Header */}
        <div className="flex items-center gap-3 mb-6" style={{ marginTop: 'var(--space-2)' }}>
          <div style={{ backgroundColor: 'var(--accent-primary)', padding: '8px', borderRadius: '10px', boxShadow: 'var(--shadow-glow)' }}>
            <Shield size={22} color="#FFF" />
          </div>
          <h1 className="text-xl font-bold tracking-tight leading-tight">Risk Intelligence Dashboard</h1>
        </div>


        {/* State Selector */}
        <div>
          <Select 
            options={states} 
            value={currentState} 
            onChange={setCurrentState} 
            placeholder="Select a State"
          />
        </div>

        {/* Highest backtest risk scores */}
        <div>
          <label className="text-xs font-semibold text-error uppercase tracking-wider mb-3 flex items-center gap-2">
            <Siren size={14} /> Highest Backtest Risk Scores
          </label>
          <div className="flex flex-col gap-3">
            {riskRes.loading ? (
              <Skeleton style={{ height: '44px', marginBottom: '8px' }} />
            ) : topRisk.length > 0 ? topRisk.map(r => (
              <div key={r.state} className="flex justify-between items-center" style={{ backgroundColor: 'rgba(255,255,255,0.02)', padding: '10px 12px', borderLeft: '3px solid var(--error)', borderRadius: '0 var(--radius-md) var(--radius-md) 0' }}>
                <span className="text-sm text-main font-medium">{r.state}</span>
                <span className="text-xs text-error font-bold" title="Out-of-fold risk score, 1 (low) to 3 (high)">
                  {fmt(r.Risk_Score, 2)} / 3
                </span>
              </div>
            )) : (
              <div className="text-xs text-tertiary">No ranking available.</div>
            )}
          </div>
        </div>

        {/* About this data */}
        <div className="card" style={{ padding: '16px', backgroundColor: 'rgba(0,0,0,0.2)', border: '1px solid rgba(6, 182, 212, 0.15)' }}>
          <div className="flex items-center gap-2 mb-3">
            <Info size={16} className="text-accent-cyan" />
            <span className="text-sm font-bold text-main tracking-wide">About This Data</span>
          </div>
          <p className="text-xs text-secondary leading-relaxed">
            Historical 2023 backtest on an unverified NCRB-style sample of 12 states. Not live, not official, not for decisions.
          </p>
        </div>

        {/* Backtest metrics */}
        <div>
          <label className="text-xs font-semibold text-tertiary uppercase tracking-wider mb-2 flex items-center gap-2">
            <Server size={14} /> Backtest Metrics (Out-of-fold)
          </label>
          <div className="card" style={{ padding: '12px', backgroundColor: 'rgba(255,255,255,0.02)' }}>
            {metricsRes.loading ? (
              <Skeleton style={{ height: '80px' }} />
            ) : metrics ? (
              <div className="text-xs text-secondary flex flex-col gap-2">
                <div className="flex justify-between items-center pb-2" style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                  <span>Trained On</span>
                  <span className="font-mono text-tertiary">{metrics.training_date}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span>Sample</span>
                  <span className="font-bold text-main">{metrics.n_states} states, {metrics.n_folds} folds</span>
                </div>
                <div className="flex justify-between items-center">
                  <span>RF rate MAE</span>
                  <span className="font-bold text-main">{fmt(metrics.regression?.rf?.oof_mae_rate_per_100k, 2)} /100k</span>
                </div>
                <div className="flex justify-between items-center">
                  <span>RF rate R²</span>
                  <span className="font-bold text-main">{fmt(metrics.regression?.rf?.oof_r2, 2)}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span>RF risk-class acc.</span>
                  <span className="font-bold text-main">{fmt(metrics.risk?.rf?.oof_accuracy, 0, 100)}%</span>
                </div>
                <div className="flex justify-between items-center">
                  <span>RF dominant-crime acc.</span>
                  <span className="font-bold text-main">{fmt(metrics.classification?.rf?.oof_accuracy, 0, 100)}%</span>
                </div>
                <p className="text-[10px] text-tertiary leading-relaxed pt-1">
                  Only {metrics.n_states} states: treat these as weak evidence.
                </p>
              </div>
            ) : (
              <div className="text-xs text-tertiary">Metrics unavailable.</div>
            )}
          </div>
        </div>

      </aside>

      {/* Main Content Workspace */}
      <main className="main-content flex flex-col">
        
        <div>
          <h2 className="text-3xl font-bold mb-4 tracking-tight">Risk Intelligence Dashboard</h2>
        </div>

        {/* Top Navigation Bar */}
        <header className="flex items-center justify-between mb-8 pb-4" style={{ borderBottom: '1px solid var(--border-subtle)' }}>
          {/* Breadcrumbs */}
          <div className="flex items-center gap-2 text-sm text-secondary font-medium tracking-wide">
            <span>Terminal</span>
            <ChevronRight size={14} className="text-tertiary" />
            <span>Sample Data (12 States)</span>
            <ChevronRight size={14} className="text-tertiary" />
            <span className="text-accent-primary font-bold">{currentState || 'No state selected'}</span>
          </div>

          <div className="flex items-center gap-3">
            <div className="flex items-center gap-1.5 text-[11px] font-bold text-accent-cyan px-2 py-1 bg-[rgba(6,182,212,0.1)] border border-[rgba(6,182,212,0.2)] rounded" title="NCRB-style crime data, not verified against official tables">
              <Database size={12} /> Data: NCRB-style sample
            </div>
            {/* Live Clock */}
            <div className="flex items-center gap-2 text-sm text-secondary font-mono bg-[rgba(255,255,255,0.02)] px-3 py-1.5 rounded-md border border-[rgba(255,255,255,0.05)]">
              <Clock size={14} className="text-tertiary" />
              {currentTime.toLocaleTimeString('en-IN', { timeZone: 'Asia/Kolkata', hour12: false })} IST
            </div>
          </div>
        </header>

        {apiError && <ErrorBanner message={apiError} />}

        <div className="flex-1">
          {loading ? (
            <div className="grid grid-cols-3 gap-4">
              <Skeleton style={{ height: '160px' }} />
              <Skeleton style={{ height: '160px' }} />
              <Skeleton style={{ height: '160px' }} />
            </div>
          ) : (
            <Card style={{ minHeight: '600px', padding: 'var(--space-4)', boxShadow: 'var(--shadow-glow)' }}>
              <Tabs tabs={tabs} defaultTab="forecast" />
            </Card>
          )}
        </div>
      </main>

    </div>
  );
}

export default App;
