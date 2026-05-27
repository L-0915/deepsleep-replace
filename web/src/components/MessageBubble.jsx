import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight';
import { Moon, User } from 'lucide-react';
import ThinkingBlock from './ThinkingBlock';

export default function MessageBubble({ message, isStreamingThinking }) {
  const isUser = message.role === 'user';
  const showThinking = message.thinking || isStreamingThinking;

  return (
    <div
      style={{
        display: 'flex',
        justifyContent: isUser ? 'flex-end' : 'flex-start',
        padding: '4px 0',
        marginBottom: 4,
      }}
    >
      <div
        style={{
          maxWidth: '85%',
          display: 'flex',
          gap: 10,
          flexDirection: isUser ? 'row-reverse' : 'row',
        }}
      >
        {/* Avatar */}
        <div
          style={{
            width: 32,
            height: 32,
            borderRadius: '50%',
            background: isUser ? 'var(--user-bubble)' : 'var(--bg-card)',
            border: isUser ? 'none' : '1px solid var(--border)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            flexShrink: 0,
            marginTop: 2,
          }}
        >
          {isUser ? (
            <User size={16} color="#fff" />
          ) : (
            <Moon size={16} style={{ color: 'var(--accent)' }} />
          )}
        </div>

        {/* Bubble content */}
        <div style={{ flex: 1, minWidth: 0 }}>
          {/* Thinking block (AI only) */}
          {!isUser && showThinking && (
            <ThinkingBlock
              thinking={message.thinking}
              isStreaming={isStreamingThinking}
            />
          )}

          {/* Message content */}
          <div
            style={{
              background: isUser ? 'var(--user-bubble)' : 'var(--ai-bubble)',
              border: isUser ? 'none' : '1px solid var(--border)',
              color: isUser ? '#fff' : 'var(--text-primary)',
              padding: '12px 16px',
              borderRadius: isUser ? '16px 16px 4px 16px' : '16px 16px 16px 4px',
              fontSize: 14,
              lineHeight: 1.7,
              wordBreak: 'break-word',
            }}
          >
            {isUser ? (
              <span style={{ whiteSpace: 'pre-wrap' }}>{message.content}</span>
            ) : (
              <div className="markdown-content">
                <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeHighlight]}>
                  {message.content || ' '}
                </ReactMarkdown>
              </div>
            )}
          </div>

          {/* Token info */}
          {message.usage && (
            <div
              style={{
                fontSize: 11,
                color: 'var(--text-secondary)',
                marginTop: 4,
                textAlign: isUser ? 'right' : 'left',
                padding: '0 4px',
              }}
            >
              prompt: {message.usage.prompt_tokens} | completion: {message.usage.completion_tokens}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
