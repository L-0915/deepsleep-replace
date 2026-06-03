import React, { useState, useRef, useEffect, useMemo } from 'react';
import { Send, Square, Lightbulb, AlertTriangle } from 'lucide-react';
import useChatStore, { estimateTokens } from '../hooks/useChat';

export default function InputArea({ onSend, contextUsage, maxContext }) {
  const [input, setInput] = useState('');
  const textareaRef = useRef(null);
  const { isGenerating, settings, setSettings, stopGeneration } = useChatStore();
  const thinking = settings.thinking;

  // 判断是否已超出上下文窗口
  const isExceeded = useMemo(() => {
    const baseTokens = (contextUsage && contextUsage.promptTokens) || 0;
    const draftTokens = estimateTokens(input);
    const max = maxContext || 2048;
    return (baseTokens + draftTokens) >= max;
  }, [contextUsage, maxContext, input]);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      const scrollHeight = textareaRef.current.scrollHeight;
      const maxH = 6 * 24;
      textareaRef.current.style.height = Math.min(scrollHeight, maxH) + 'px';
    }
  }, [input]);

  const handleSend = () => {
    const trimmed = input.trim();
    if (!trimmed || isGenerating || isExceeded) return;
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
      {/* 上下文已满警告 */}
      {isExceeded && !isGenerating && (
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 6,
            background: 'rgba(239, 68, 68, 0.1)',
            border: '1px solid rgba(239, 68, 68, 0.3)',
            borderRadius: 10,
            padding: '8px 12px',
            marginBottom: 8,
            fontSize: 13,
            color: '#fca5a5',
          }}
        >
          <AlertTriangle size={15} />
          <span>已达最大上下文窗口，请<button
            onClick={() => useChatStore.getState().newConversation()}
            style={{
              background: 'none',
              border: 'none',
              color: '#f87171',
              cursor: 'pointer',
              textDecoration: 'underline',
              padding: 0,
              font: 'inherit',
            }}
          >新建对话</button></span>
        </div>
      )}

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
          placeholder={isExceeded ? "上下文已满，请新建对话..." : "输入你的睡眠健康问题..."}
          rows={1}
          disabled={isExceeded && !isGenerating}
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
            opacity: (isExceeded && !isGenerating) ? 0.5 : 1,
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
            disabled={!input.trim() || isExceeded}
            style={{
              background: (input.trim() && !isExceeded) ? 'var(--accent)' : 'var(--border)',
              border: 'none',
              color: (input.trim() && !isExceeded) ? '#fff' : 'var(--text-secondary)',
              cursor: (input.trim() && !isExceeded) ? 'pointer' : 'default',
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
