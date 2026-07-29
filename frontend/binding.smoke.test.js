const fs = require('node:fs');
const vm = require('node:vm');

class MockClassList {
  constructor() { this.values = new Set(); }
  add(...names) { names.forEach(name => this.values.add(name)); }
  remove(...names) { names.forEach(name => this.values.delete(name)); }
  toggle(name, force) {
    if (force === true) this.values.add(name);
    else if (force === false) this.values.delete(name);
    else if (this.values.has(name)) this.values.delete(name);
    else this.values.add(name);
    return this.values.has(name);
  }
  contains(name) { return this.values.has(name); }
}

class MockElement {
  constructor() {
    this.textContent = '';
    this.innerHTML = '';
    this.value = '';
    this.checked = true;
    this.disabled = false;
    this.dataset = {};
    this.className = '';
    this.classList = new MockClassList();
    this.style = { values: {}, setProperty: (name, value) => { this.style.values[name] = value; } };
    this.scrollHeight = 0;
    this.scrollTop = 0;
  }
  addEventListener() {}
  showModal() {}
  close() {}
  click() {}
}

const elements = new Map();
const getElement = selector => {
  if (!elements.has(selector)) elements.set(selector, new MockElement());
  return elements.get(selector);
};

getElement('#questionInput').value = '测试病例';
getElement('#endpointInput').value = 'http://127.0.0.1:50042/chat/stream';
getElement('#difficultySelect').value = 'hard';
getElement('#difficultyAgentToggle').checked = false;
getElement('#webSearchToggle').checked = false;

const localStore = new Map();
const sandbox = {
  console,
  Date,
  Intl,
  JSON,
  Math,
  Map,
  Set,
  Blob,
  AbortController,
  TextDecoder,
  URL: { createObjectURL: () => 'blob:test', revokeObjectURL: () => {} },
  setTimeout,
  clearTimeout,
  fetch: () => { throw new Error('Smoke test must not call the network'); },
  navigator: { clipboard: { writeText: async () => {} } },
  localStorage: {
    getItem: key => localStore.get(key) || null,
    setItem: (key, value) => localStore.set(key, value)
  },
  document: {
    querySelector: getElement,
    querySelectorAll: () => [],
    addEventListener: () => {},
    createElement: () => new MockElement()
  },
  window: {}
};

vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(__dirname + '/app.js', 'utf8'), sandbox, { filename: 'app.js' });

const state = sandbox.window.MedScope.getState();
const votes = Number(getElement('#supportCount').textContent) + Number(getElement('#reserveCount').textContent);
const manualPayload = sandbox.window.MedScope.buildRequestPayload('测试问题');
getElement('#difficultyAgentToggle').checked = true;
getElement('#webSearchToggle').checked = true;
const automaticPayload = sandbox.window.MedScope.buildRequestPayload('测试问题');
sandbox.window.MedScope.handleStreamEvent({
  type: 'final_result',
  timestamp: Date.now(),
  content: '<think>这部分思考过程不应显示。</think>\n答案：\n建议：患者应由皮肤科医生或感染科医生评估后制定个体化治疗方案。\n完整正文应当全部显示，不能在三百个字符处截断。'
});
const longDecisionState = sandbox.window.MedScope.getState();
const assertions = [
  [state.experts.length === 4, 'expert recruitment binding'],
  [state.events.length === 8, 'event stream binding'],
  [state.stages.every(stage => stage.status === 'complete'), 'stage status binding'],
  [state.decision.finalized === true, 'final decision binding'],
  [state.decision.evidence.length === 4, 'evidence extraction binding'],
  [state.decision.confidence > 0, 'confidence binding'],
  [votes === 4, 'vote aggregation binding'],
  [getElement('#expertCount').textContent === '4 位专家', 'expert count DOM binding'],
  [manualPayload.enableDifficultyAgent === false && manualPayload.difficulty === 'hard', 'manual difficulty binding'],
  [manualPayload.enableWebSearch === false, 'web search off binding'],
  [automaticPayload.enableDifficultyAgent === true && automaticPayload.difficulty === null, 'agent difficulty binding'],
  [automaticPayload.enableWebSearch === true, 'web search on binding'],
  [longDecisionState.decision.title === '建议：患者应由皮肤科医生或感染科医生评估后制定个体化治疗方案。', 'full decision title binding'],
  [!longDecisionState.decision.summary.includes('思考过程') && longDecisionState.decision.summary.includes('完整正文应当全部显示'), 'visible answer cleanup binding']
];

const failed = assertions.filter(([passed]) => !passed);
if (failed.length) {
  failed.forEach(([, label]) => console.error(`FAIL: ${label}`));
  process.exit(1);
}

assertions.forEach(([, label]) => console.log(`PASS: ${label}`));
