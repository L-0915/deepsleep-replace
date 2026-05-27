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
import { saveConversations } from './utils/storage';

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

  const handleSend = useCallback(async (text) => {
    // Ensure we have a conversation
    let convId = useChatStore.getState().currentConversationId;
    if (!convId) {
      useChatStore.getState().newConversation();
      convId = useChatStore.getState().currentConversationId;
    }

    // Add user message using the confirmed convId
    const { conversations: convs } = useChatStore.getState();
    const convIdx = convs.findIndex(c => c.id === convId);
    if (convIdx === -1) return;

    const updated = [...convs];
    updated[convIdx] = {
      ...updated[convIdx],
      messages: [
        ...updated[convIdx].messages,
        { id: Date.now().toString(36) + Math.random().toString(36).slice(2, 8), role: 'user', content: text, thinking: '', timestamp: Date.now() },
      ],
      title: updated[convIdx].messages.length === 0 ? text.slice(0, 24) + (text.length > 24 ? '...' : '') : updated[convIdx].title,
    };
    saveConversations(updated);
    useChatStore.setState({ conversations: updated });

    // Prepare messages for API (from the updated conversation)
    const apiMessages = updated[convIdx].messages.map(m => ({ role: m.role, content: m.content }));

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
      for await (const event of streamChat(apiMessages, model, settings.thinking, {
        signal: controller.signal,
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
  }, [settings, addMessage, updateLastMessage, setIsGenerating, setAbortController, newConversation]);

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
