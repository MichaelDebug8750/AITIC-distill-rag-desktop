// 真机验证：浏览器中断 SSE 后，服务应取消 Ollama 生成并可立即处理下一次问答。
const base = process.env.AITIC_AUDIT_BASE || 'http://127.0.0.1:8032';
const thinkPython = '20260806_075049_dab0c710';
const businessLaw = '20260812_143033_1a701cae';

async function post(path, body, signal) {
  return fetch(base + path, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body),
    signal,
  });
}

(async () => {
  const controller = new AbortController();
  const started = Date.now();
  const response = await post('/api/ask/stream', {
    question: 'Explain tuple assignment in detail, including its purpose, evaluation order, examples, and limitations.',
    libraries: [thinkPython], mode: 'deep', style: 'detailed', extend: false,
    hybrid: false, history: [],
  }, controller.signal);
  if (!response.ok || !response.body) throw new Error(`stream HTTP ${response.status}`);
  const reader = response.body.getReader();
  const first = await reader.read();
  if (first.done || !first.value?.length) throw new Error('stream ended before first event');
  controller.abort();
  try { await reader.read(); } catch (_) {}
  const abortedMs = Date.now() - started;

  await new Promise(resolve => setTimeout(resolve, 1200));
  const statusStarted = Date.now();
  const statusResponse = await fetch(base + '/api/status');
  const status = await statusResponse.json();
  const statusMs = Date.now() - statusStarted;
  if (!statusResponse.ok || !status.ready) throw new Error('service not ready after abort');

  const askStarted = Date.now();
  const askResponse = await post('/api/ask', {
    question: 'What is meant by Behavioral measures?', libraries: [businessLaw],
    mode: 'auto', style: 'standard', extend: false, hybrid: false, history: [],
  });
  const answer = await askResponse.json();
  const askMs = Date.now() - askStarted;
  if (!askResponse.ok) throw new Error(`follow-up HTTP ${askResponse.status}`);
  if (!answer.abstained || answer.answer !== '[NO REFERENCE FOUND]') {
    throw new Error('follow-up answer violated refusal contract');
  }
  console.log(JSON.stringify({
    ok: true, firstEventBytes: first.value.length, abortedMs, statusMs, askMs,
    followUp: {abstained: answer.abstained, answer: answer.answer,
      rounds: answer.agent?.rounds, confidence: answer.agent?.confidence?.level},
  }, null, 2));
})().catch(error => {
  console.error(error.stack || error);
  process.exit(1);
});
