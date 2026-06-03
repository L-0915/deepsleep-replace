import React, { useState, useRef, useEffect } from 'react';
import { Menu, Settings, ChevronDown, Moon } from 'lucide-react';
import useChatStore from '../hooks/useChat';
import useModelStore from '../hooks/useModel';

export default function Header() {
  const { sidebarOpen, setSidebarOpen, setSettingsOpen, settings, setSettings } = useChatStore();
  const { currentArch, currentBeta, setArch, setBeta } = useModelStore();
  const archLabels = { deepsleep: 'DeepSleep', qwen: 'Qwen', qwen_mt: 'Qwen 多轮' };
  const [archDropdownOpen, setArchDropdownOpen] = useState(false);
  const [betaDropdownOpen, setBetaDropdownOpen] = useState(false);
  const archRef = useRef(null);
  const betaRef = useRef(null);

  useEffect(() => {
    function handleClickOutside(e) {
      if (archRef.current && !archRef.current.contains(e.target)) {
        setArchDropdownOpen(false);
      }
      if (betaRef.current && !betaRef.current.contains(e.target)) {
        setBetaDropdownOpen(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', settings.theme);
  }, [settings.theme]);

  const toggleTheme = () => {
    setSettings({ theme: settings.theme === 'dark' ? 'light' : 'dark' });
  };

  return (
    <header
      style={{
        height: 56,
        background: settings.theme === 'dark'
          ? 'rgba(20, 20, 20, 0.8)'
          : 'rgba(255, 255, 255, 0.8)',
        backdropFilter: 'blur(12px)',
        borderBottom: `1px solid var(--border)`,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '0 16px',
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        zIndex: 100,
      }}
    >
      {/* Left section */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
        <button
          onClick={() => setSidebarOpen(!sidebarOpen)}
          style={{
            background: 'none',
            border: 'none',
            color: 'var(--text-primary)',
            cursor: 'pointer',
            padding: 6,
            borderRadius: 6,
            display: 'flex',
            alignItems: 'center',
          }}
          aria-label="切换侧边栏"
        >
          <Menu size={20} />
        </button>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <Moon size={22} style={{ color: 'var(--accent)' }} />
          <span style={{ fontWeight: 700, fontSize: 18, letterSpacing: '-0.3px' }}>
            DeepSleep
          </span>
        </div>
      </div>

      {/* Center - Model selectors */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        {/* Architecture selector */}
        <div ref={archRef} style={{ position: 'relative' }}>
          <button
            onClick={() => { setArchDropdownOpen(!archDropdownOpen); setBetaDropdownOpen(false); }}
            style={{
              background: 'var(--bg-card)',
              border: '1px solid var(--border)',
              color: 'var(--text-primary)',
              padding: '6px 12px',
              borderRadius: 8,
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: 6,
              fontSize: 14,
            }}
          >
            {archLabels[currentArch] || 'Qwen'}
            <ChevronDown size={14} />
          </button>
          {archDropdownOpen && (
            <div
              style={{
                position: 'absolute',
                top: '100%',
                left: 0,
                marginTop: 4,
                background: 'var(--bg-card)',
                border: '1px solid var(--border)',
                borderRadius: 8,
                overflow: 'hidden',
                minWidth: 140,
                zIndex: 200,
                boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
              }}
            >
              {['deepsleep', 'qwen', 'qwen_mt'].map(arch => (
                <button
                  key={arch}
                  onClick={() => { setArch(arch); setArchDropdownOpen(false); }}
                  style={{
                    display: 'block',
                    width: '100%',
                    padding: '8px 12px',
                    background: currentArch === arch ? 'var(--accent)' : 'transparent',
                    border: 'none',
                    color: currentArch === arch ? '#fff' : 'var(--text-primary)',
                    cursor: 'pointer',
                    textAlign: 'left',
                    fontSize: 14,
                  }}
                >
                  {archLabels[arch]}
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Beta selector */}
        <div ref={betaRef} style={{ position: 'relative' }}>
          <button
            onClick={() => { setBetaDropdownOpen(!betaDropdownOpen); setArchDropdownOpen(false); }}
            style={{
              background: 'var(--bg-card)',
              border: '1px solid var(--border)',
              color: 'var(--text-primary)',
              padding: '6px 12px',
              borderRadius: 8,
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: 6,
              fontSize: 14,
            }}
          >
            beta={currentBeta}
            <ChevronDown size={14} />
          </button>
          {betaDropdownOpen && (
            <div
              style={{
                position: 'absolute',
                top: '100%',
                left: 0,
                marginTop: 4,
                background: 'var(--bg-card)',
                border: '1px solid var(--border)',
                borderRadius: 8,
                overflow: 'hidden',
                minWidth: 120,
                zIndex: 200,
                boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
              }}
            >
              {['0.1', '0.5'].map(beta => (
                <button
                  key={beta}
                  onClick={() => { setBeta(beta); setBetaDropdownOpen(false); }}
                  style={{
                    display: 'block',
                    width: '100%',
                    padding: '8px 12px',
                    background: currentBeta === beta ? 'var(--accent)' : 'transparent',
                    border: 'none',
                    color: currentBeta === beta ? '#fff' : 'var(--text-primary)',
                    cursor: 'pointer',
                    textAlign: 'left',
                    fontSize: 14,
                  }}
                >
                  beta={beta}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Right section */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <button
          onClick={toggleTheme}
          style={{
            background: 'none',
            border: 'none',
            color: 'var(--text-secondary)',
            cursor: 'pointer',
            padding: 6,
            borderRadius: 6,
            display: 'flex',
            alignItems: 'center',
            fontSize: 18,
          }}
          aria-label="切换主题"
        >
          {settings.theme === 'dark' ? '☀️' : '🌙'}
        </button>
        <button
          onClick={() => setSettingsOpen(true)}
          style={{
            background: 'none',
            border: 'none',
            color: 'var(--text-secondary)',
            cursor: 'pointer',
            padding: 6,
            borderRadius: 6,
            display: 'flex',
            alignItems: 'center',
          }}
          aria-label="设置"
        >
          <Settings size={20} />
        </button>
      </div>
    </header>
  );
}
