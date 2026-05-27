import React from 'react';
import { Plus, MessageSquare, Trash2, Activity, GitCompare } from 'lucide-react';
import useChatStore from '../hooks/useChat';

function getDateGroup(timestamp) {
  const now = new Date();
  const date = new Date(timestamp);
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const yesterday = new Date(today - 86400000);
  const dateDay = new Date(date.getFullYear(), date.getMonth(), date.getDate());

  if (dateDay.getTime() === today.getTime()) return 'today';
  if (dateDay.getTime() === yesterday.getTime()) return 'yesterday';
  return 'earlier';
}

export default function Sidebar() {
  const {
    conversations,
    currentConversationId,
    newConversation,
    deleteConversation,
    switchConversation,
    sidebarOpen,
    setSidebarOpen,
    setAssessmentOpen,
    setCompareOpen,
  } = useChatStore();

  const grouped = { today: [], yesterday: [], earlier: [] };
  conversations.forEach(c => {
    const group = getDateGroup(c.createdAt);
    grouped[group].push(c);
  });

  const handleSelect = (id) => {
    switchConversation(id);
    // close sidebar on mobile
    if (window.innerWidth < 768) {
      setSidebarOpen(false);
    }
  };

  const handleNew = () => {
    newConversation();
    if (window.innerWidth < 768) {
      setSidebarOpen(false);
    }
  };

  return (
    <>
      {/* Mobile overlay */}
      {sidebarOpen && (
        <div
          onClick={() => setSidebarOpen(false)}
          style={{
            position: 'fixed',
            inset: 0,
            background: 'rgba(0,0,0,0.5)',
            zIndex: 150,
          }}
        />
      )}

      <aside
        style={{
          width: sidebarOpen ? 260 : 0,
          minWidth: sidebarOpen ? 260 : 0,
          background: 'var(--bg-secondary)',
          borderRight: '1px solid var(--border)',
          height: '100vh',
          overflow: 'hidden',
          transition: 'width 0.2s ease, min-width 0.2s ease',
          position: 'relative',
          zIndex: 200,
          display: 'flex',
          flexDirection: 'column',
        }}
      >
        <div style={{ width: 260, height: '100%', display: 'flex', flexDirection: 'column' }}>
          {/* New conversation button */}
          <div style={{ padding: '12px 12px 8px' }}>
            <button
              onClick={handleNew}
              style={{
                width: '100%',
                padding: '10px 16px',
                background: 'var(--accent)',
                color: '#fff',
                border: 'none',
                borderRadius: 10,
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: 8,
                fontSize: 14,
                fontWeight: 600,
                transition: 'background 0.15s',
              }}
              onMouseEnter={e => e.currentTarget.style.background = 'var(--accent-hover)'}
              onMouseLeave={e => e.currentTarget.style.background = 'var(--accent)'}
            >
              <Plus size={18} />
              新对话
            </button>
          </div>

          {/* Conversation list */}
          <div style={{ flex: 1, overflowY: 'auto', padding: '0 8px' }}>
            {['today', 'yesterday', 'earlier'].map(group => {
              const items = grouped[group];
              if (items.length === 0) return null;

              return (
                <div key={group} style={{ marginBottom: 8 }}>
                  <div
                    style={{
                      padding: '8px 8px 4px',
                      fontSize: 12,
                      color: 'var(--text-secondary)',
                      fontWeight: 500,
                    }}
                  >
                    {group === 'today' ? '今天' : group === 'yesterday' ? '昨天' : '更早'}
                  </div>
                  {items.map(conv => (
                    <div
                      key={conv.id}
                      className="conv-item"
                      onClick={() => handleSelect(conv.id)}
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        padding: '8px 10px',
                        borderRadius: 8,
                        cursor: 'pointer',
                        background: conv.id === currentConversationId
                          ? 'var(--bg-card)'
                          : 'transparent',
                        transition: 'background 0.15s',
                        gap: 8,
                        marginBottom: 2,
                      }}
                      onMouseEnter={e => {
                        if (conv.id !== currentConversationId) {
                          e.currentTarget.style.background = 'var(--bg-card)';
                        }
                      }}
                      onMouseLeave={e => {
                        if (conv.id !== currentConversationId) {
                          e.currentTarget.style.background = 'transparent';
                        }
                      }}
                    >
                      <MessageSquare size={16} style={{ color: 'var(--text-secondary)', flexShrink: 0 }} />
                      <span
                        style={{
                          flex: 1,
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          whiteSpace: 'nowrap',
                          fontSize: 14,
                        }}
                      >
                        {conv.title}
                      </span>
                      <button
                        onClick={e => {
                          e.stopPropagation();
                          deleteConversation(conv.id);
                        }}
                        style={{
                          background: 'none',
                          border: 'none',
                          color: 'var(--text-secondary)',
                          cursor: 'pointer',
                          padding: 2,
                          borderRadius: 4,
                          display: 'flex',
                          alignItems: 'center',
                          opacity: 0,
                          transition: 'opacity 0.15s',
                          flexShrink: 0,
                        }}
                        className="delete-btn"
                        onMouseEnter={e => { e.currentTarget.style.color = '#ef4444'; e.currentTarget.style.opacity = 1; }}
                        onMouseLeave={e => { e.currentTarget.style.color = 'var(--text-secondary)'; e.currentTarget.style.opacity = 0; }}
                      >
                        <Trash2 size={14} />
                      </button>
                    </div>
                  ))}
                </div>
              );
            })}
          </div>

          {/* Bottom tools */}
          <div style={{ padding: 12, borderTop: '1px solid var(--border)' }}>
            <button
              onClick={() => { setAssessmentOpen(true); if (window.innerWidth < 768) setSidebarOpen(false); }}
              style={{
                width: '100%',
                padding: '10px 12px',
                background: 'none',
                border: '1px solid var(--border)',
                borderRadius: 8,
                color: 'var(--text-primary)',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: 8,
                fontSize: 13,
                marginBottom: 8,
                transition: 'background 0.15s',
              }}
              onMouseEnter={e => e.currentTarget.style.background = 'var(--bg-card)'}
              onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
            >
              <Activity size={16} style={{ color: 'var(--accent)' }} />
              睡眠评估
            </button>
            <button
              onClick={() => { setCompareOpen(true); if (window.innerWidth < 768) setSidebarOpen(false); }}
              style={{
                width: '100%',
                padding: '10px 12px',
                background: 'none',
                border: '1px solid var(--border)',
                borderRadius: 8,
                color: 'var(--text-primary)',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: 8,
                fontSize: 13,
                transition: 'background 0.15s',
              }}
              onMouseEnter={e => e.currentTarget.style.background = 'var(--bg-card)'}
              onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
            >
              <GitCompare size={16} style={{ color: 'var(--accent)' }} />
              模型对比
            </button>
          </div>
        </div>
      </aside>
    </>
  );
}
