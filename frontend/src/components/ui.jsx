import React, { useState } from 'react';
import { ChevronDown, Loader2, AlertTriangle } from 'lucide-react';

export function Card({ children, className = '', ...props }) {
  return (
    <div className={`card ${className}`} {...props}>
      {children}
    </div>
  );
}

export function Button({ 
  children, 
  variant = 'primary', 
  isLoading = false, 
  className = '', 
  disabled,
  ...props 
}) {
  const baseStyle = {
    display: 'inline-flex', alignItems: 'center', justifyContent: 'center', gap: '8px',
    padding: '8px 16px', fontSize: 'var(--font-size-sm)', fontWeight: 'var(--font-weight-medium)',
    borderRadius: 'var(--radius-md)', cursor: 'pointer', transition: 'all var(--transition-fast)',
    border: '1px solid transparent', outline: 'none'
  };
  
  const variantStyles = {
    primary: {
      backgroundColor: 'var(--accent-primary)', color: '#fff',
      boxShadow: 'var(--shadow-glow)'
    },
    outline: {
      backgroundColor: 'transparent', borderColor: 'var(--border-strong)', color: 'var(--text-main)'
    }
  };

  return (
    <button 
      style={{ ...baseStyle, ...(variantStyles[variant] || {}), opacity: (disabled || isLoading) ? 0.5 : 1, cursor: (disabled || isLoading) ? 'not-allowed' : 'pointer' }}
      disabled={disabled || isLoading}
      className={className}
      {...props}
    >
      {isLoading ? <Loader2 size={16} className="animate-spin" /> : null}
      {children}
    </button>
  );
}

export function Skeleton({ className = '', style = {} }) {
  return <div className={`skeleton ${className}`} style={{ minHeight: '20px', ...style }}></div>;
}

export function Tabs({ tabs, defaultTab, onChange }) {
  const [activeTab, setActiveTab] = useState(defaultTab || tabs[0].id);

  const handleTabClick = (tabId) => {
    setActiveTab(tabId);
    if (onChange) onChange(tabId);
  };

  return (
    <div className="tabs-container">
      <div className="flex gap-2 mb-4" style={{ borderBottom: '1px solid var(--border-subtle)', paddingBottom: '0' }}>
        {tabs.map((tab) => {
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => handleTabClick(tab.id)}
              className="font-medium text-sm flex items-center gap-2"
              style={{
                padding: 'var(--space-2) var(--space-3)',
                cursor: 'pointer', border: 'none', backgroundColor: 'transparent',
                color: isActive ? 'var(--text-main)' : 'var(--text-secondary)',
                borderBottom: isActive ? '2px solid var(--accent-primary)' : '2px solid transparent',
                transition: 'color var(--transition-fast)',
                whiteSpace: 'nowrap'
              }}
            >
              {tab.icon && <tab.icon size={16} color={isActive ? 'var(--accent-primary)' : 'currentColor'} />}
              {tab.label}
            </button>
          );
        })}
      </div>
      <div className="tab-content" style={{ animation: 'fadeIn var(--transition-smooth)' }}>
        {tabs.find(t => t.id === activeTab)?.content}
      </div>
    </div>
  );
}

export function Select({ options, value, onChange, placeholder = 'Select Item...' }) {
  return (
    <div style={{ position: 'relative', width: '100%' }}>
      <select 
        value={value} 
        onChange={e => onChange(e.target.value)}
        style={{
          width: '100%', padding: 'var(--space-2) var(--space-3)',
          fontSize: 'var(--font-size-sm)', backgroundColor: 'var(--bg-surface)',
          color: 'var(--text-main)', border: '1px solid var(--border-strong)',
          borderRadius: 'var(--radius-md)', appearance: 'none', cursor: 'pointer',
          outline: 'none', transition: 'border-color var(--transition-fast)'
        }}
        onFocus={(e) => e.target.style.borderColor = 'var(--accent-primary)'}
        onBlur={(e) => e.target.style.borderColor = 'var(--border-strong)'}
      >
        <option value="" disabled>{placeholder}</option>
        {options.map(opt => (
          <option key={opt.value || opt} value={opt.value || opt}>
            {opt.label || opt}
          </option>
        ))}
      </select>
      <ChevronDown 
        size={16} 
        color="var(--text-secondary)" 
        style={{ position: 'absolute', right: '12px', top: '50%', transform: 'translateY(-50%)', pointerEvents: 'none' }} 
      />
    </div>
  );
}

export function ErrorBanner({ message = "Backend offline — showing cached data" }) {
  return (
    <div style={{ backgroundColor: 'rgba(248, 113, 113, 0.1)', color: 'var(--error)', padding: 'var(--space-3) var(--space-4)', borderRadius: 'var(--radius-md)', border: '1px solid var(--error)', display: 'flex', alignItems: 'center', gap: '8px', marginBottom: 'var(--space-4)', fontSize: 'var(--font-size-sm)', fontWeight: 'var(--font-weight-medium)' }}>
      <AlertTriangle size={18} />
      {message}
    </div>
  );
}
