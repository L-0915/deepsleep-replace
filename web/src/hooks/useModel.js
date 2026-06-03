import { create } from 'zustand';
import { fetchModels as fetchModelsApi } from '../utils/api';

export const MODEL_MAX_CONTEXT = { deepsleep: 1800, qwen: 32000, qwen_mt: 32000 };

const useModelStore = create((set, get) => ({
  models: [
    { id: 'ds_b0.1', name: 'DeepSleep beta=0.1', arch: 'deepsleep', loaded: true },
    { id: 'ds_b0.5', name: 'DeepSleep beta=0.5', arch: 'deepsleep', loaded: true },
    { id: 'qwen_b0.1', name: 'Qwen beta=0.1', arch: 'qwen', loaded: true },
    { id: 'qwen_b0.5', name: 'Qwen beta=0.5', arch: 'qwen', loaded: true },
    { id: 'qwen_mt_b0.1', name: 'Qwen多轮 beta=0.1', arch: 'qwen_mt', loaded: true },
    { id: 'qwen_mt_b0.5', name: 'Qwen多轮 beta=0.5', arch: 'qwen_mt', loaded: true },
  ],
  currentModel: 'ds_b0.1',
  currentArch: 'deepsleep',
  currentBeta: '0.1',
  isLoading: false,

  getMaxContext: () => MODEL_MAX_CONTEXT[get().currentArch] || 2048,

  setModel: (modelId) => {
    const { models } = get();
    const model = models.find(m => m.id === modelId);
    if (model) {
      set({
        currentModel: modelId,
        currentArch: model.arch,
        currentBeta: model.id.split('_b')[1] || '0.1',
      });
    }
  },

  setArch: (arch) => {
    const { models, currentBeta } = get();
    const model = models.find(m => m.arch === arch && m.id.includes(`_b${currentBeta}`));
    if (model) {
      set({ currentModel: model.id, currentArch: arch });
    }
  },

  setBeta: (beta) => {
    const { models, currentArch } = get();
    const model = models.find(m => m.arch === currentArch && m.id.includes(`_b${beta}`));
    if (model) {
      set({ currentModel: model.id, currentBeta: beta });
    }
  },

  fetchModels: async () => {
    set({ isLoading: true });
    try {
      const models = await fetchModelsApi();
      if (models && models.length > 0) {
        set({ models });
      }
    } catch (e) {
      console.error('Failed to fetch models:', e);
    } finally {
      set({ isLoading: false });
    }
  },
}));

export default useModelStore;
