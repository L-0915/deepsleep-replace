import React, { useState, useCallback, useRef, useEffect } from 'react';
import Header from './components/Header';
import Sidebar from './components/Sidebar';
import ChatArea from './components/ChatArea';
import InputArea from './components/InputArea';
import SettingsPanel from './components/SettingsPanel';
import CompareMode from './components/CompareMode';
import SleepAssessment from './components/SleepAssessment';
import useChatStore from './hooks/useChat';
import useModelStore from './hooks/useModel';
import { streamChat } from './utils/api';

export default function App() {
  const {
    conversations,
    currentConversationId,
    isGenerating,
    setIsGenerating,
    setAbortController,
    newConversation,
    addMessage,
    updateLastMessage,
    settings,
  } = useChatStore();

  const { currentModel } = useModelStore();
  const [streamingMessage, setStreamingMessage] = useState(null);
  const [streamingThinking, setStreamingThinking] = useState(false);
  const abortRef = useRef(null);

  const currentConv = conversations.find(c => c.id === currentConversationId) || null;
  const messages = currentConv ? currentConv.messages : [];

  // Keep a ref to handleSend so the event listener always has the latest version
  const handleSendRef = useRef(handleSend);
  handleSendRef.current = handleSend;

  // Listen for custom event from SleepAssessment
  useEffect(() => {
    const handler = (e) => {
      handleSendRef.current(e.detail);
    };
    window.addEventListener('deepsleep-send-message', handler);
    return () => window.removeEventListener('deepsleep-send-message', handler);
  }, []);

  const handleSend = useCallback(async (text) => {
    // Ensure we have a conversation
    let convId = currentConversationId;
    if (!convId) {
      newConversation();
      convId = useChatStore.getState().currentConversationId;
    }

    // Add user message
    addMessage('user', text);

    // Prepare messages for API
    const conv = useChatStore.getState().conversations.find(c => c.id === convId);
    const apiMessages = conv
      ? conv.messages.map(m => ({ role: m.role, content: m.content }))
      : [{ role: 'user', content: text }];

    // If conversation was just created, the user message is already there
    const finalMessages = apiMessages.length > 0 ? apiMessages : [{ role: 'user', content: text }];

    // Start streaming
    setIsGenerating(true);
    const controller = new AbortController();
    abortRef.current = controller;
    setAbortController(controller);

    let thinkingContent = '';
    let content = '';
    let usage = null;

    // Initialize streaming state
    setStreamingMessage({ id: 'streaming', role: 'assistant', content: '', thinking: '', usage: null });
    setStreamingThinking(false);

    try {
      const model = useModelStore.getState().currentModel;
      for await (const event of streamChat(finalMessages, model, settings.thinking, {
        temperature: settings.temperature,
        top_p: settings.top_p,
        max_tokens: settings.max_tokens,
      })) {
        if (controller.signal.aborted) break;

        if (event.type === 'thinking') {
          thinkingContent += event.content;
          setStreamingThinking(true);
          setStreamingMessage(prev => prev ? { ...prev, thinking: thinkingContent } : prev);
        } else if (event.type === 'content') {
          content += event.content;
          setStreamingThinking(false);
          setStreamingMessage(prev => prev ? { ...prev, content } : prev);
        } else if (event.type === 'done') {
          usage = event.usage;
        } else if (event.type === 'error') {
          content += `\n\n⚠️ ${event.content}`;
          setStreamingMessage(prev => prev ? { ...prev, content } : prev);
        }
      }
    } catch (e) {
      if (e.name !== 'AbortError') {
        console.error('Stream error:', e);
        content += '\n\n⚠️ 连接出错，请重试。';
        setStreamingMessage(prev => prev ? { ...prev, content } : prev);
      }
    }

    // Commit the assistant message to the store
    addMessage('assistant', content, thinkingContent);
    if (usage) {
      updateLastMessage('assistant', { usage });
    }

    setStreamingMessage(null);
    setStreamingThinking(false);
    setIsGenerating(false);
    setAbortController(null);
  }, [currentConversationId, settings, addMessage, updateLastMessage, setIsGenerating, setAbortController, newConversation]);

  return (
    <div style={{ display: 'flex', height: '100vh', overflow: 'hidden', background: 'var(--bg-primary)' }}>
      <Sidebar />

      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden', paddingTop: 56 }}>
        <Header />
        <ChatArea
          messages={messages}
          streamingMessage={streamingMessage}
          streamingThinking={streamingThinking}
          onQuickSend={handleSend}
        />
        <InputArea onSend={handleSend} />
      </div>

      <SettingsPanel />
      <CompareMode />
      <SleepAssessment />
    </div>
  );
}
