import { create } from 'zustand';
import { saveConversations, loadConversations, saveSettings, loadSettings } from '../utils/storage';

const defaultSettings = {
  temperature: 0.7,
  top_p: 0.9,
  max_tokens: 512,
  thinking: true,
  theme: 'dark',
};

function generateId() {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
}

function getTitle(content) {
  if (!content) return '新对话';
  const text = typeof content === 'string' ? content : '';
  return text.slice(0, 24) + (text.length > 24 ? '...' : '') || '新对话';
}

const useChatStore = create((set, get) => {
  const savedConversations = loadConversations();
  const savedSettings = loadSettings();

  return {
    conversations: savedConversations,
    currentConversationId: savedConversations.length > 0 ? savedConversations[0].id : null,
    settings: { ...defaultSettings, ...savedSettings },
    isGenerating: false,
    abortController: null,
    sidebarOpen: true,
    settingsOpen: false,
    assessmentOpen: false,
    compareOpen: false,

    setSidebarOpen: (open) => set({ sidebarOpen: open }),
    setSettingsOpen: (open) => set({ settingsOpen: open }),
    setAssessmentOpen: (open) => set({ assessmentOpen: open }),
    setCompareOpen: (open) => set({ compareOpen: open }),

    newConversation: () => {
      const id = generateId();
      const conv = {
        id,
        title: '新对话',
        messages: [],
        model: get().settings.currentModel || 'ds_b0.1',
        createdAt: Date.now(),
      };
      const conversations = [conv, ...get().conversations];
      saveConversations(conversations);
      set({ conversations, currentConversationId: id });
    },

    deleteConversation: (id) => {
      const conversations = get().conversations.filter(c => c.id !== id);
      saveConversations(conversations);
      const currentId = get().currentConversationId;
      set({
        conversations,
        currentConversationId: currentId === id
          ? (conversations.length > 0 ? conversations[0].id : null)
          : currentId,
      });
    },

    switchConversation: (id) => {
      set({ currentConversationId: id });
    },

    getCurrentConversation: () => {
      const { conversations, currentConversationId } = get();
      return conversations.find(c => c.id === currentConversationId) || null;
    },

    addMessage: (role, content, thinking = '') => {
      const { conversations, currentConversationId } = get();
      const idx = conversations.findIndex(c => c.id === currentConversationId);
      if (idx === -1) return;

      const msg = { id: generateId(), role, content, thinking, timestamp: Date.now() };
      const updated = [...conversations];
      updated[idx] = {
        ...updated[idx],
        messages: [...updated[idx].messages, msg],
        title: updated[idx].messages.length === 0 && role === 'user'
          ? getTitle(content)
          : updated[idx].title,
      };
      saveConversations(updated);
      set({ conversations: updated });
    },

    updateLastMessage: (role, updates) => {
      const { conversations, currentConversationId } = get();
      const idx = conversations.findIndex(c => c.id === currentConversationId);
      if (idx === -1) return;

      const updated = [...conversations];
      const msgs = [...updated[idx].messages];
      const lastIdx = msgs.length - 1;
      if (lastIdx >= 0 && msgs[lastIdx].role === role) {
        msgs[lastIdx] = { ...msgs[lastIdx], ...updates };
        updated[idx] = { ...updated[idx], messages: msgs };
        saveConversations(updated);
        set({ conversations: updated });
      }
    },

    setSettings: (partial) => {
      const settings = { ...get().settings, ...partial };
      saveSettings(settings);
      set({ settings });
    },

    setIsGenerating: (val) => set({ isGenerating: val }),
    setAbortController: (ctrl) => set({ abortController: ctrl }),

    stopGeneration: () => {
      const { abortController } = get();
      if (abortController) {
        abortController.abort();
        set({ abortController: null, isGenerating: false });
      }
    },
  };
});

export default useChatStore;
