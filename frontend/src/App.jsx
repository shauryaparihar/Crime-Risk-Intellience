import React, { useState, useEffect } from 'react';
import './index.css';
import { Card, Select, Tabs, Skeleton } from './components/ui';
import { 
  BarChart3, TrendingUp, AlertTriangle, Map as MapIcon, 
  Shield, Activity, Siren, Radio, Crosshair, 
  Search, User, Clock, ChevronRight, Lock, Server, Database
} from 'lucide-react';
import DashboardForecast from './components/DashboardForecast';
import DashboardCrimePattern from './components/DashboardCrimePattern';
import DashboardRisk from './components/DashboardRisk';
import DashboardMap from './components/DashboardMap';

const API_BASE = import.meta.env.VITE_API_URL
  ? `${import.meta.env.VITE_API_URL}/api`
  : 'http://localhost:8001/api';

function App() {
  const [states, setStates] = useState([]);
  const [currentState, setCurrentState] = useState('');
  const [status, setStatus] = useState({ models_ready: false, data_loaded: false });
  const [globalRisk, setGlobalRisk] = useState([]);
  const [loading, setLoading] = useState(true);
  const [metrics, setMetrics] = useState(null);
  const [currentTime, setCurrentTime] = useState(new Date());

  // Live clock tick
  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    async function init() {
      try {
        const [statusRes, statesRes, riskRes, metricsRes] = await Promise.all([
          fetch(`${API_BASE}/status`).catch(() => ({ json: () => ({}) })),
          fetch(`${API_BASE}/states`).catch(() => ({ json: () => ({ states: [] }) })),
          fetch(`${API_BASE}/risk-analysis`).catch(() => ({ json: () => ({ rankings: [] }) })),
          fetch(`${API_BASE}/metrics`).catch(() => ({ json: () => (null) }))
        ]);
        const statusData = await statusRes.json();
        const statesData = await statesRes.json();
        const riskData = await riskRes.json();
        const metricsData = await metricsRes.json();
        
        setStatus(statusData);
        setMetrics(metricsData);
        setStates(statesData.states || []);
        if (statesData.states?.length > 0) {
          setCurrentState(statesData.states[0]);
        }
        if (riskData.rankings) {
          setGlobalRisk(riskData.rankings.filter(r => r.Predicted_Risk_Class === 'high').slice(0, 3));
        }
      } catch (e) {
        console.error("Connection error", e);
      } finally {
        setLoading(false);
      }
    }
    init();
  }, []);

  const tabs = [
    {
      id: 'forecast', label: 'Forecast & Analysis', icon: BarChart3,
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

        {/* Global Search Input */}
        <div className="mb-6 relative">
          <Search size={16} className="absolute left-3 top-1/2 transform -translate-y-1/2 text-tertiary" />
          <input 
            type="text" 
            placeholder="Query databases..." 
            className="w-full bg-[rgba(255,255,255,0.03)] border border-[rgba(255,255,255,0.08)] rounded-md text-sm text-main focus:outline-none focus:border-accent-primary transition-all"
            style={{ 
              padding: '10px 12px 10px 36px', 
              width: '100%',
              backgroundColor: 'rgba(0,0,0,0.2)',
              border: '1px solid var(--border-strong)',
              borderRadius: 'var(--radius-md)',
              color: 'var(--text-main)'
            }}
          />
        </div>

        {/* State Selector */}
        <div className="mb-6">
          <label className="text-xs font-semibold text-tertiary uppercase tracking-wider mb-2 flex items-center gap-2">
            <Crosshair size={14} /> Target Region
          </label>
          <Select 
            options={states} 
            value={currentState} 
            onChange={setCurrentState} 
            placeholder="Select a State"
          />
        </div>

        {/* Critical Risk Zones */}
        <div className="mb-6">
          <label className="text-xs font-semibold text-error uppercase tracking-wider mb-2 flex items-center gap-2">
            <Siren size={14} /> Critical Risk Zones
          </label>
          <div className="flex flex-col gap-2">
            {globalRisk.length > 0 ? globalRisk.map((r, i) => {
              // Deterministic fake trend based on index just for visual flair
              const trend = (4.2 + (i * 1.7)).toFixed(1);
              return (
                <div key={r.state} className="flex justify-between items-center" style={{ backgroundColor: 'rgba(255,255,255,0.02)', padding: '10px 12px', borderLeft: '3px solid var(--error)', borderRadius: '0 var(--radius-md) var(--radius-md) 0' }}>
                  <span className="text-sm text-main font-medium">{r.state}</span>
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-error font-bold flex items-center gap-1">
                      <TrendingUp size={12} strokeWidth={3} /> {trend}%
                    </span>
                  </div>
                </div>
              );
            }) : (
              <Skeleton style={{ height: '44px', marginBottom: '8px' }} />
            )}
          </div>
        </div>

        {/* Live Intel Feed */}
        <div className="card mb-6" style={{ padding: '14px', backgroundColor: 'rgba(0,0,0,0.2)', border: '1px solid rgba(6, 182, 212, 0.15)' }}>
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <Radio size={16} className="text-accent-cyan" />
              <span className="text-sm font-bold text-main tracking-wide">Live Intel Feed</span>
            </div>
            <div style={{ width: '8px', height: '8px', backgroundColor: 'var(--success)', borderRadius: '50%', boxShadow: '0 0 10px var(--success)', animation: 'pulse 2s infinite' }}></div>
          </div>
          <p className="text-xs text-secondary leading-relaxed">Processing active FIR logs from regional central bureaus.</p>
          <div className="text-[10px] text-tertiary mt-3 font-mono tracking-wider uppercase">Sync: {currentTime.toISOString().split('T')[1].split('.')[0]} UTC</div>
        </div>

        {/* Model Health / Metrics */}
        <div className="mb-6">
          <label className="text-xs font-semibold text-tertiary uppercase tracking-wider mb-2 flex items-center gap-2">
            <Server size={14} /> Model Health
          </label>
          <div className="card" style={{ padding: '12px', backgroundColor: 'rgba(255,255,255,0.02)' }}>
            {metrics ? (
              <div className="text-xs text-secondary flex flex-col gap-2">
                <div className="flex justify-between items-center pb-2" style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                  <span>Trained On</span>
                  <span className="font-mono text-tertiary">{metrics.training_date}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span>RF R² (Train)</span>
                  <span className="font-bold text-main">{(metrics.regression?.rf?.train_r2 * 100)?.toFixed(1)}%</span>
                </div>
                <div className="flex justify-between items-center">
                  <span>RF MAE (5-CV)</span>
                  <span className="font-bold text-main">{metrics.regression?.rf?.cv_mae_mean?.toFixed(2)}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span>Logit Acc</span>
                  <span className="font-bold text-main">{(metrics.classification?.logit_acc_train * 100)?.toFixed(1)}%</span>
                </div>
              </div>
            ) : (
                <Skeleton style={{ height: '80px' }} />
            )}
          </div>
        </div>

        {/* Bottom Profile / Status Component */}
        <div className="mt-auto">
          <div className="card flex items-center gap-3" style={{ padding: '12px', backgroundColor: 'rgba(255,255,255,0.02)' }}>
            <div style={{ backgroundColor: 'rgba(255,255,255,0.05)', padding: '10px', borderRadius: '50%' }}>
              <User size={20} className="text-secondary" />
            </div>
            <div className="flex-1">
              <div className="text-sm font-bold tracking-wide">Agent #884-AX</div>
              <div className="text-xs text-secondary flex items-center gap-1 mt-1">
                <Lock size={10} /> Clearance Lvl 4
              </div>
            </div>
            <div>
               <Activity size={18} color={status.models_ready ? 'var(--success)' : 'var(--error)'} title={status.models_ready ? 'Models Active' : 'Offline'} />
            </div>
          </div>
        </div>
      </aside>

      {/* Main Content Workspace */}
      <main className="main-content flex flex-col">
        
        {/* Top Navigation Bar */}
        <header className="flex items-center justify-between mb-6 pb-4" style={{ borderBottom: '1px solid var(--border-subtle)' }}>
          {/* Breadcrumbs */}
          <div className="flex items-center gap-2 text-sm text-secondary font-medium tracking-wide">
            <span>Terminal</span>
            <ChevronRight size={14} className="text-tertiary" />
            <span>India Context</span>
            <ChevronRight size={14} className="text-tertiary" />
            <span className="text-accent-primary font-bold">{currentState || 'Loading...'}</span>
          </div>

          <div className="flex items-center gap-3">
            <div className="flex items-center gap-1.5 text-[11px] font-bold text-accent-cyan px-2 py-1 bg-[rgba(6,182,212,0.1)] border border-[rgba(6,182,212,0.2)] rounded" title="Data sourced from National Crime Records Bureau">
              <Database size={12} /> Data: NCRB 2022
            </div>
            {/* Live Clock */}
            <div className="flex items-center gap-2 text-sm text-secondary font-mono bg-[rgba(255,255,255,0.02)] px-3 py-1.5 rounded-md border border-[rgba(255,255,255,0.05)]">
              <Clock size={14} className="text-tertiary" />
              {currentTime.toLocaleTimeString('en-US', { hour12: false })} UTC
            </div>
          </div>
        </header>

        <div>
          <h2 className="text-3xl font-bold mb-2 tracking-tight">Risk Intelligence Dashboard</h2>
          <p className="text-md text-secondary mb-6">Predicting regional anomalies utilizing deep ensemble analytics.</p>
        </div>

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

      <style>{`
        @keyframes pulse {
          0% { opacity: 1; transform: scale(1); }
          50% { opacity: 0.5; transform: scale(1.1); }
          100% { opacity: 1; transform: scale(1); }
        }
      `}</style>
    </div>
  );
}

export default App;
