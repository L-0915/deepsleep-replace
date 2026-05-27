import React, { useRef, useEffect } from 'react';
import MessageBubble from './MessageBubble';
import WelcomeScreen from './WelcomeScreen';

export default function ChatArea({ messages, streamingMessage, streamingThinking, onQuickSend }) {
  const scrollRef = useRef(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, streamingMessage, streamingThinking]);

  const allMessages = [...messages];
  if (streamingMessage) {
    const lastIdx = allMessages.length - 1;
    if (lastIdx >= 0 && allMessages[lastIdx].role === 'assistant' && streamingMessage) {
      allMessages[lastIdx] = { ...allMessages[lastIdx], ...streamingMessage };
    } else {
      allMessages.push(streamingMessage);
    }
  }

  if (allMessages.length === 0) {
    return (
      <div style={{ flex: 1, overflow: 'auto' }}>
        <WelcomeScreen onSend={onQuickSend} />
      </div>
    );
  }

  return (
    <div
      ref={scrollRef}
      style={{
        flex: 1,
        overflowY: 'auto',
        padding: '20px 16px',
        paddingBottom: 40,
      }}
    >
      <div style={{ maxWidth: 800, margin: '0 auto' }}>
        {allMessages.map((msg, i) => (
          <MessageBubble
            key={msg.id || i}
            message={msg}
            isStreamingThinking={
              i === allMessages.length - 1 && msg.role === 'assistant' && streamingThinking
            }
          />
        ))}
      </div>
    </div>
  );
}
