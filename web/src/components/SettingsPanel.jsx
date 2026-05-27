import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X } from 'lucide-react';
import useChatStore from '../hooks/useChat';

export default function SettingsPanel() {
  const { settingsOpen, setSettingsOpen, settings, setSettings } = useChatStore();

  return (
    <AnimatePresence>
      {settingsOpen && (
        <>
          {/* Overlay */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            onClick={() => setSettingsOpen(false)}
            style={{
              position: 'fixed',
              inset: 0,
              background: 'rgba(0,0,0,0.5)',
              zIndex: 300,
            }}
          />
          {/* Panel */}
          <motion.div
            initial={{ x: '100%' }}
            animate={{ x: 0 }}
            exit={{ x: '100%' }}
            transition={{ duration: 0.25, ease: 'easeInOut' }}
            style={{
              position: 'fixed',
              top: 0,
              right: 0,
              width: 360,
              maxWidth: '90vw',
              height: '100vh',
              background: 'var(--bg-secondary)',
              borderLeft: '1px solid var(--border)',
              zIndex: 400,
              display: 'flex',
              flexDirection: 'column',
            }}
          >
            {/* Header */}
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                padding: '16px 20px',
                borderBottom: '1px solid var(--border)',
              }}
            >
              <h2 style={{ fontSize: 18, fontWeight: 600 }}>设置</h2>
              <button
                onClick={() => setSettingsOpen(false)}
                style={{
                  background: 'none',
                  border: 'none',
                  color: 'var(--text-primary)',
                  cursor: 'pointer',
                  padding: 4,
                  borderRadius: 6,
                  display: 'flex',
                  alignItems: 'center',
                }}
              >
                <X size={20} />
              </button>
            </div>

            {/* Content */}
            <div style={{ flex: 1, overflowY: 'auto', padding: '20px' }}>
              {/* Temperature */}
              <div style={{ marginBottom: 24 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                  <label style={{ fontSize: 14, fontWeight: 500 }}>Temperature</label>
                  <span style={{ fontSize: 14, color: 'var(--accent)' }}>{settings.temperature}</span>
                </div>
                <input
                  type="range"
                  min="0.1"
                  max="1.5"
                  step="0.1"
                  value={settings.temperature}
                  onChange={e => setSettings({ temperature: parseFloat(e.target.value) })}
                  style={{
                    width: '100%',
                    accentColor: 'var(--accent)',
                    height: 6,
                    cursor: 'pointer',
                  }}
                />
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: 'var(--text-secondary)', marginTop: 4 }}>
                  <span>精确 0.1</span>
                  <span>创意 1.5</span>
                </div>
              </div>

              {/* Top P */}
              <div style={{ marginBottom: 24 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                  <label style={{ fontSize: 14, fontWeight: 500 }}>Top-P</label>
                  <span style={{ fontSize: 14, color: 'var(--accent)' }}>{settings.top_p}</span>
                </div>
                <input
                  type="range"
                  min="0.5"
                  max="1.0"
                  step="0.05"
                  value={settings.top_p}
                  onChange={e => setSettings({ top_p: parseFloat(e.target.value) })}
                  style={{
                    width: '100%',
                    accentColor: 'var(--accent)',
                    height: 6,
                    cursor: 'pointer',
                  }}
                />
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: 'var(--text-secondary)', marginTop: 4 }}>
                  <span>0.5</span>
                  <span>1.0</span>
                </div>
              </div>

              {/* Max Tokens */}
              <div style={{ marginBottom: 24 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                  <label style={{ fontSize: 14, fontWeight: 500 }}>最大生成长度</label>
                  <span style={{ fontSize: 14, color: 'var(--accent)' }}>{settings.max_tokens}</span>
                </div>
                <input
                  type="range"
                  min="64"
                  max="1024"
                  step="64"
                  value={settings.max_tokens}
                  onChange={e => setSettings({ max_tokens: parseInt(e.target.value) })}
                  style={{
                    width: '100%',
                    accentColor: 'var(--accent)',
                    height: 6,
                    cursor: 'pointer',
                  }}
                />
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: 'var(--text-secondary)', marginTop: 4 }}>
                  <span>64</span>
                  <span>1024</span>
                </div>
              </div>

              {/* Theme toggle */}
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  padding: '12px 0',
                  borderTop: '1px solid var(--border)',
                }}
              >
                <div>
                  <div style={{ fontSize: 14, fontWeight: 500 }}>主题模式</div>
                  <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginTop: 2 }}>
                    {settings.theme === 'dark' ? '当前: 深色主题' : '当前: 浅色主题'}
                  </div>
                </div>
                <button
                  onClick={() => setSettings({ theme: settings.theme === 'dark' ? 'light' : 'dark' })}
                  style={{
                    width: 48,
                    height: 26,
                    borderRadius: 13,
                    border: 'none',
                    background: settings.theme === 'dark' ? 'var(--accent)' : 'var(--border)',
                    cursor: 'pointer',
                    position: 'relative',
                    transition: 'background 0.2s',
                  }}
                >
                  <div
                    style={{
                      width: 20,
                      height: 20,
                      borderRadius: '50%',
                      background: '#fff',
                      position: 'absolute',
                      top: 3,
                      left: settings.theme === 'dark' ? 25 : 3,
                      transition: 'left 0.2s ease',
                    }}
                  />
                </button>
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}
