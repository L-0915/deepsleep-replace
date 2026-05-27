import React, { useState, useRef, useCallback, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Send, Square } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight';
import useChatStore from '../hooks/useChat';
import useModelStore from '../hooks/useModel';
import { streamChat } from '../utils/api';

function ComparePanel({ modelId, modelLabel, messages, thinking, isStreaming, isStreamingThinking, usage, genTime }) {
  return (
    <div
      style={{
        flex: 1,
        display: 'flex',
        flexDirection: 'column',
        background: 'var(--bg-secondary)',
        borderRadius: 12,
        border: '1px solid var(--border)',
        overflow: 'hidden',
        minWidth: 0,
      }}
    >
      {/* Panel header */}
      <div
        style={{
          padding: '10px 14px',
          borderBottom: '1px solid var(--border)',
          fontSize: 13,
          fontWeight: 600,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}
      >
        <span>{modelLabel}</span>
        {genTime > 0 && (
          <span style={{ fontSize: 11, color: 'var(--text-secondary)' }}>
            {genTime.toFixed(1)}s
          </span>
        )}
      </div>

      {/* Messages */}
      <div style={{ flex: 1, overflowY: 'auto', padding: 12 }}>
        {messages.map((msg, i) => (
          <div key={i} style={{ marginBottom: 12 }}>
            {msg.role === 'user' ? (
              <div style={{ textAlign: 'right' }}>
                <span
                  style={{
                    display: 'inline-block',
                    background: 'var(--user-bubble)',
                    color: '#fff',
                    padding: '8px 12px',
                    borderRadius: '12px 12px 4px 12px',
                    fontSize: 13,
                    maxWidth: '80%',
                    textAlign: 'left',
                  }}
                >
                  {msg.content}
                </span>
              </div>
            ) : (
              <div>
                {msg.thinking && (
                  <details style={{ marginBottom: 8 }}>
                    <summary
                      style={{
                        fontSize: 12,
                        color: 'var(--accent)',
                        cursor: 'pointer',
                        padding: '4px 0',
                      }}
                    >
                      💭 思考过程
                      {isStreamingThinking && (
                        <span className="thinking-dots" style={{ marginLeft: 6 }}>
                          <span /><span /><span />
                        </span>
                      )}
                    </summary>
                    <div
                      style={{
                        fontSize: 12,
                        color: 'var(--text-secondary)',
                        padding: '6px 8px',
                        background: 'var(--thinking-bg)',
                        borderRadius: 6,
                        whiteSpace: 'pre-wrap',
                      }}
                    >
                      {msg.thinking}
                    </div>
                  </details>
                )}
                <div
                  className="markdown-content"
                  style={{
                    fontSize: 13,
                    lineHeight: 1.6,
                    background: 'var(--ai-bubble)',
                    border: '1px solid var(--border)',
                    padding: '10px 12px',
                    borderRadius: '0 12px 12px 12px',
                  }}
                >
                  <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeHighlight]}>
                    {msg.content || ' '}
                  </ReactMarkdown>
                </div>
              </div>
            )}
          </div>
        ))}
        {isStreaming && !messages.some(m => m.role === 'assistant') && (
          <div className="thinking-dots" style={{ padding: 8 }}>
            <span /><span /><span />
          </div>
        )}
      </div>

      {/* Usage info */}
      {usage && (
        <div
          style={{
            padding: '8px 14px',
            borderTop: '1px solid var(--border)',
            fontSize: 11,
            color: 'var(--text-secondary)',
            display: 'flex',
            justifyContent: 'space-between',
          }}
        >
          <span>prompt: {usage.prompt_tokens}</span>
          <span>completion: {usage.completion_tokens}</span>
        </div>
      )}
    </div>
  );
}

export default function CompareMode() {
  const { compareOpen, setCompareOpen, settings } = useChatStore();
  const { models } = useModelStore();
  const [leftModel, setLeftModel] = useState('ds_b0.1');
  const [rightModel, setRightModel] = useState('qwen_b0.1');
  const [input, setInput] = useState('');
  const [leftState, setLeftState] = useState({ messages: [], usage: null, time: 0, streaming: false, streamingThinking: false });
  const [rightState, setRightState] = useState({ messages: [], usage: null, time: 0, streaming: false, streamingThinking: false });
  const [isGenerating, setIsGenerating] = useState(false);
  const leftAbortRef = useRef(null);
  const rightAbortRef = useRef(null);

  const runStream = useCallback(async (modelId, messages, setState, abortRef) => {
    const controller = new AbortController();
    abortRef.current = controller;
    const startTime = Date.now();

    setState(prev => ({ ...prev, streaming: true, streamingThinking: false }));

    let thinkingContent = '';
    let content = '';

    // Add empty assistant message
    setState(prev => ({
      ...prev,
      messages: [...prev.messages, { role: 'assistant', content: '', thinking: '' }],
    }));

    try {
      for await (const event of streamChat(messages, modelId, settings.thinking, {
        temperature: settings.temperature,
        top_p: settings.top_p,
        max_tokens: settings.max_tokens,
      })) {
        if (controller.signal.aborted) break;

        if (event.type === 'thinking') {
          thinkingContent += event.content;
          setState(prev => ({
            ...prev,
            streamingThinking: true,
            messages: prev.messages.map((m, i) =>
              i === prev.messages.length - 1 && m.role === 'assistant'
                ? { ...m, thinking: thinkingContent }
                : m
            ),
          }));
        } else if (event.type === 'content') {
          content += event.content;
          setState(prev => ({
            ...prev,
            streamingThinking: false,
            messages: prev.messages.map((m, i) =>
              i === prev.messages.length - 1 && m.role === 'assistant'
                ? { ...m, content }
                : m
            ),
          }));
        } else if (event.type === 'done') {
          setState(prev => ({
            ...prev,
            usage: event.usage,
            time: (Date.now() - startTime) / 1000,
            streaming: false,
            streamingThinking: false,
          }));
        } else if (event.type === 'error') {
          setState(prev => ({
            ...prev,
            messages: prev.messages.map((m, i) =>
              i === prev.messages.length - 1 && m.role === 'assistant'
                ? { ...m, content: prev.messages[prev.messages.length - 1].content + `\n\n⚠️ ${event.content}` }
                : m
            ),
            streaming: false,
            streamingThinking: false,
          }));
        }
      }
    } catch (e) {
      if (e.name !== 'AbortError') {
        console.error('Stream error:', e);
      }
      setState(prev => ({ ...prev, streaming: false, streamingThinking: false }));
    }
  }, [settings]);

  const handleSend = () => {
    const trimmed = input.trim();
    if (!trimmed || isGenerating) return;

    const userMsg = { role: 'user', content: trimmed };

    setLeftState(prev => ({ ...prev, messages: [...prev.messages, userMsg] }));
    setRightState(prev => ({ ...prev, messages: [...prev.messages, userMsg] }));
    setInput('');
    setIsGenerating(true);

    const leftMessages = [...leftState.messages, userMsg].map(m => ({ role: m.role, content: m.content }));
    const rightMessages = [...rightState.messages, userMsg].map(m => ({ role: m.role, content: m.content }));

    Promise.all([
      runStream(leftModel, leftMessages, setLeftState, leftAbortRef),
      runStream(rightModel, rightMessages, setRightState, rightAbortRef),
    ]).finally(() => setIsGenerating(false));
  };

  const stopGeneration = () => {
    if (leftAbortRef.current) leftAbortRef.current.abort();
    if (rightAbortRef.current) rightAbortRef.current.abort();
    setIsGenerating(false);
    setLeftState(prev => ({ ...prev, streaming: false }));
    setRightState(prev => ({ ...prev, streaming: false }));
  };

  const handleClose = () => {
    stopGeneration();
    setLeftState({ messages: [], usage: null, time: 0, streaming: false, streamingThinking: false });
    setRightState({ messages: [], usage: null, time: 0, streaming: false, streamingThinking: false });
    setCompareOpen(false);
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <AnimatePresence>
      {compareOpen && (
        <>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={handleClose}
            style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.5)', zIndex: 300 }}
          />
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            transition={{ duration: 0.2 }}
            style={{
              position: 'fixed',
              inset: 16,
              background: 'var(--bg-primary)',
              borderRadius: 16,
              border: '1px solid var(--border)',
              zIndex: 400,
              display: 'flex',
              flexDirection: 'column',
              overflow: 'hidden',
            }}
          >
            {/* Header */}
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                padding: '12px 16px',
                borderBottom: '1px solid var(--border)',
              }}
            >
              <h2 style={{ fontSize: 16, fontWeight: 600 }}>模型对比</h2>
              <button
                onClick={handleClose}
                style={{
                  background: 'none',
                  border: 'none',
                  color: 'var(--text-primary)',
                  cursor: 'pointer',
                  padding: 4,
                  borderRadius: 6,
                  display: 'flex',
                }}
              >
                <X size={20} />
              </button>
            </div>

            {/* Model selectors */}
            <div
              style={{
                display: 'flex',
                gap: 12,
                padding: '8px 16px',
                borderBottom: '1px solid var(--border)',
              }}
            >
              <select
                value={leftModel}
                onChange={e => setLeftModel(e.target.value)}
                disabled={isGenerating}
                style={{
                  flex: 1,
                  padding: '6px 10px',
                  background: 'var(--bg-card)',
                  border: '1px solid var(--border)',
                  borderRadius: 8,
                  color: 'var(--text-primary)',
                  fontSize: 13,
                  cursor: 'pointer',
                }}
              >
                {models.map(m => (
                  <option key={m.id} value={m.id}>{m.name}</option>
                ))}
              </select>
              <span style={{ color: 'var(--text-secondary)', alignSelf: 'center', fontSize: 13 }}>VS</span>
              <select
                value={rightModel}
                onChange={e => setRightModel(e.target.value)}
                disabled={isGenerating}
                style={{
                  flex: 1,
                  padding: '6px 10px',
                  background: 'var(--bg-card)',
                  border: '1px solid var(--border)',
                  borderRadius: 8,
                  color: 'var(--text-primary)',
                  fontSize: 13,
                  cursor: 'pointer',
                }}
              >
                {models.map(m => (
                  <option key={m.id} value={m.id}>{m.name}</option>
                ))}
              </select>
            </div>

            {/* Compare panels */}
            <div style={{ flex: 1, display: 'flex', gap: 12, padding: 12, overflow: 'hidden' }}>
              <ComparePanel
                modelId={leftModel}
                modelLabel={models.find(m => m.id === leftModel)?.name || leftModel}
                messages={leftState.messages}
                thinking={leftState.thinking}
                isStreaming={leftState.streaming}
                isStreamingThinking={leftState.streamingThinking}
                usage={leftState.usage}
                genTime={leftState.time}
              />
              <ComparePanel
                modelId={rightModel}
                modelLabel={models.find(m => m.id === rightModel)?.name || rightModel}
                messages={rightState.messages}
                thinking={rightState.thinking}
                isStreaming={rightState.streaming}
                isStreamingThinking={rightState.streamingThinking}
                usage={rightState.usage}
                genTime={rightState.time}
              />
            </div>

            {/* Shared input */}
            <div style={{ padding: '0 16px 12px' }}>
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 8,
                  background: 'var(--bg-input)',
                  border: '1px solid var(--border)',
                  borderRadius: 12,
                  padding: '8px 12px',
                }}
              >
                <input
                  value={input}
                  onChange={e => setInput(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder="输入问题，同时向两个模型提问..."
                  disabled={isGenerating}
                  style={{
                    flex: 1,
                    background: 'none',
                    border: 'none',
                    outline: 'none',
                    color: 'var(--text-primary)',
                    fontSize: 14,
                    fontFamily: 'inherit',
                  }}
                />
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
                    }}
                  >
                    <Send size={16} />
                  </button>
                )}
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}
