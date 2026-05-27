import React from 'react';
import { Moon } from 'lucide-react';

const quickQuestions = [
  { icon: '💤', text: '失眠了怎么办？' },
  { icon: '😤', text: '睡眠呼吸暂停有哪些症状？' },
  { icon: '💊', text: '褪黑素可以长期服用吗？' },
  { icon: '⏰', text: '每天睡多久才算健康？' },
  { icon: '👶', text: '宝宝晚上睡觉出汗正常吗？' },
  { icon: '🌟', text: '如何改善睡眠质量？' },
];

export default function WelcomeScreen({ onSend }) {
  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        height: '100%',
        padding: '40px 20px',
        maxWidth: 720,
        margin: '0 auto',
      }}
    >
      <div style={{ textAlign: 'center', marginBottom: 48 }}>
        <div
          style={{
            fontSize: 48,
            marginBottom: 16,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: 12,
          }}
        >
          <Moon size={44} style={{ color: 'var(--accent)' }} />
        </div>
        <h1
          style={{
            fontSize: 32,
            fontWeight: 700,
            marginBottom: 8,
            background: 'linear-gradient(135deg, var(--accent) 0%, #a78bfa 100%)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
          }}
        >
          你好，我是小曦
        </h1>
        <p style={{ fontSize: 16, color: 'var(--text-secondary)', fontWeight: 400 }}>
          你的 AI 睡眠健康助手
        </p>
      </div>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
          gap: 12,
          width: '100%',
        }}
      >
        {quickQuestions.map((q, i) => (
          <button
            key={i}
            onClick={() => onSend(q.text)}
            style={{
              padding: '14px 16px',
              background: 'var(--bg-card)',
              border: '1px solid var(--border)',
              borderRadius: 12,
              color: 'var(--text-primary)',
              cursor: 'pointer',
              textAlign: 'left',
              fontSize: 14,
              display: 'flex',
              alignItems: 'center',
              gap: 10,
              transition: 'all 0.2s ease',
              lineHeight: 1.4,
            }}
            onMouseEnter={e => {
              e.currentTarget.style.borderColor = 'var(--accent)';
              e.currentTarget.style.transform = 'translateY(-2px)';
              e.currentTarget.style.boxShadow = '0 4px 12px rgba(99, 102, 241, 0.15)';
            }}
            onMouseLeave={e => {
              e.currentTarget.style.borderColor = 'var(--border)';
              e.currentTarget.style.transform = 'translateY(0)';
              e.currentTarget.style.boxShadow = 'none';
            }}
          >
            <span style={{ fontSize: 20, flexShrink: 0 }}>{q.icon}</span>
            <span>{q.text}</span>
          </button>
        ))}
      </div>

      <p
        style={{
          marginTop: 32,
          fontSize: 12,
          color: 'var(--text-secondary)',
          textAlign: 'center',
          lineHeight: 1.6,
        }}
      >
        小曦可以回答睡眠健康相关问题，但不能替代专业医疗建议。
      </p>
    </div>
  );
}
