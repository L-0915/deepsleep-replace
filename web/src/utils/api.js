const API_BASE = '/api';

export async function* streamChat(messages, model, thinking, params = {}) {
  const body = {
    messages,
    model,
    thinking: thinking ?? true,
    temperature: params.temperature ?? 0.7,
    top_p: params.top_p ?? 0.9,
    max_tokens: params.max_tokens ?? 512,
  };

  const response = await fetch(`${API_BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal: params.signal,
  });

  if (!response.ok) {
    const errorText = await response.text();
    let errorMsg;
    try {
      const errorJson = JSON.parse(errorText);
      errorMsg = errorJson.detail || errorJson.message || errorText;
    } catch {
      errorMsg = errorText;
    }
    yield { type: 'error', content: `请求失败 (${response.status}): ${errorMsg}` };
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed || !trimmed.startsWith('data: ')) continue;

        const data = trimmed.slice(6);
        if (data === '[DONE]') return;

        try {
          const parsed = JSON.parse(data);
          yield parsed;
        } catch {
          // skip malformed lines
        }
      }
    }

    // Process remaining buffer
    if (buffer.trim()) {
      const trimmed = buffer.trim();
      if (trimmed.startsWith('data: ')) {
        const data = trimmed.slice(6);
        if (data !== '[DONE]') {
          try {
            yield JSON.parse(data);
          } catch {
            // skip
          }
        }
      }
    }
  } catch (err) {
    if (err.name === 'AbortError') return;
    yield { type: 'error', content: `连接中断: ${err.message}` };
  }
}

export async function fetchModels() {
  try {
    const response = await fetch(`${API_BASE}/models`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    return data.models || [];
  } catch (err) {
    console.error('Failed to fetch models:', err);
    return [];
  }
}

export async function fetchHealth() {
  try {
    const response = await fetch(`${API_BASE}/health`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return await response.json();
  } catch (err) {
    console.error('Failed to fetch health:', err);
    return null;
  }
}
