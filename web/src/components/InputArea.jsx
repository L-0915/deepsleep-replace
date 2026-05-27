import React, { useState, useRef, useEffect } from 'react';
import { Send, Square, Lightbulb } from 'lucide-react';
import useChatStore from '../hooks/useChat';

export default function InputArea({ onSend }) {
  const [input, setInput] = useState('');
  const textareaRef = useRef(null);
  const { isGenerating, settings, setSettings, stopGeneration } = useChatStore();
  const thinking = settings.thinking;

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      const scrollHeight = textareaRef.current.scrollHeight;
      const maxH = 6 * 24; // ~6 lines
      textareaRef.current.style.height = Math.min(scrollHeight, maxH) + 'px';
    }
  }, [input]);

  const handleSend = () => {
    const trimmed = input.trim();
    if (!trimmed || isGenerating) return;
    onSend(trimmed);
    setInput('');
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div
      style={{
        padding: '0 16px 16px',
        maxWidth: 800,
        margin: '0 auto',
        width: '100%',
      }}
    >
      <div
        style={{
          display: 'flex',
          alignItems: 'flex-end',
          gap: 8,
          background: 'var(--bg-input)',
          border: '1px solid var(--border)',
          borderRadius: 16,
          padding: '8px 12px',
          transition: 'border-color 0.2s',
        }}
        onFocus={e => { e.currentTarget.style.borderColor = 'var(--accent)'; }}
        onBlur={e => { e.currentTarget.style.borderColor = 'var(--border)'; }}
      >
        <textarea
          ref={textareaRef}
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="输入你的睡眠健康问题..."
          rows={1}
          style={{
            flex: 1,
            background: 'none',
            border: 'none',
            outline: 'none',
            color: 'var(--text-primary)',
            fontSize: 14,
            lineHeight: '24px',
            resize: 'none',
            maxHeight: 144,
            fontFamily: 'inherit',
            padding: 0,
          }}
        />

        {/* Thinking toggle */}
        <button
          onClick={() => setSettings({ thinking: !thinking })}
          style={{
            background: thinking ? 'rgba(99, 102, 241, 0.15)' : 'none',
            border: 'none',
            color: thinking ? 'var(--accent)' : 'var(--text-secondary)',
            cursor: 'pointer',
            padding: 6,
            borderRadius: 8,
            display: 'flex',
            alignItems: 'center',
            transition: 'all 0.15s',
            flexShrink: 0,
          }}
          title={thinking ? '关闭思考模式' : '开启思考模式'}
        >
          <Lightbulb size={20} />
        </button>

        {/* Send / Stop button */}
        {isGenerating ? (
          <button
            onClick={stopGeneration}
            style={{
              background: 'var(--accent)',
              border: 'none',
              color: '#fff',
              cursor: 'pointer',
              padding: 6,
              borderRadius: 8,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              width: 36,
              height: 36,
              flexShrink: 0,
              transition: 'background 0.15s',
            }}
          >
            <Square size={16} />
          </button>
        ) : (
          <button
            onClick={handleSend}
            disabled={!input.trim()}
            style={{
              background: input.trim() ? 'var(--accent)' : 'var(--border)',
              border: 'none',
              color: input.trim() ? '#fff' : 'var(--text-secondary)',
              cursor: input.trim() ? 'pointer' : 'default',
              padding: 6,
              borderRadius: 8,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              width: 36,
              height: 36,
              flexShrink: 0,
              transition: 'all 0.15s',
            }}
          >
            <Send size={16} />
          </button>
        )}
      </div>
    </div>
  );
}
