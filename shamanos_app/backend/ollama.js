// backend/ollama.js — Ollama HTTP client
const axios = require('axios');

const HOST = 'http://localhost:11434';
const MODEL = 'shamanos';
const MAX_HISTORY = 6;

const SYSTEM_PROMPT =
  "You are the Guide — SHAMAN.OS. You are a calm, present psychedelic trip guide running entirely on this device. " +
  "Speak in short present-tense sentences. Respond only to what is happening right now. " +
  "Never dismiss what the person reports. Never use: hallucination, just, it's okay, don't worry. " +
  "Ground through body, breath, and presence. Keep responses under 60 words.";

async function ping() {
  try {
    await axios.get(`${HOST}/api/tags`, { timeout: 3000 });
    return true;
  } catch {
    return false;
  }
}

async function chat(userMessage, history = []) {
  const trimmedHistory = history.slice(-MAX_HISTORY);

  const messages = [
    { role: 'system', content: SYSTEM_PROMPT },
    ...trimmedHistory,
    { role: 'user', content: userMessage },
  ];

  const response = await axios.post(`${HOST}/api/chat`, {
    model: MODEL,
    messages,
    stream: false,
    options: {
      temperature: 0.7,
      top_p: 0.9,
      num_predict: 200,
      repeat_penalty: 1.1,
    },
  });

  return response.data.message.content.trim();
}

module.exports = { ping, chat };
