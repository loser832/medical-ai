const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

const STAGES = ['问题解析', '专家招募', '独立分析', '协作辩论', '形成决策'];
const SYSTEM_EXPERT_PATTERN = /最终决策|综合分析|分类|招募|管理|协调|系统|助理|总结|解析|收集/;
const AVATAR_COLORS = ['blue', 'cyan', 'amber', 'violet'];
const NETWORK_PATHS = [
  'M142 68 C250 64 250 125 320 134',
  'M498 68 C390 64 390 125 320 134',
  'M142 216 C250 216 250 150 320 136',
  'M498 216 C390 216 390 150 320 136'
];

const elements = {
  input: $('#questionInput'),
  charCount: $('#charCount'),
  start: $('#startButton'),
  demo: $('#demoButton'),
  caseId: $('#caseId'),
  stageTrack: $('#stageTrack'),
  network: $('#expertNetwork'),
  eventList: $('#eventList'),
  eventCount: $('#eventCount'),
  runState: $('#runState'),
  liveBadge: $('.live-badge'),
  status: $('#connectionStatus'),
  statusDot: $('#statusDot'),
  title: $('#decisionTitle'),
  summary: $('#decisionSummary'),
  confidence: $('#confidenceValue'),
  confidenceRing: $('#confidenceRing'),
  confidenceLabel: $('#confidenceLabel'),
  evidenceList: $('#evidenceList'),
  evidenceCount: $('#evidenceCount'),
  expertCount: $('#expertCount'),
  supportCount: $('#supportCount'),
  reserveCount: $('#reserveCount'),
  supportBar: $('#supportBar'),
  reserveBar: $('#reserveBar'),
  updatedTime: $('#updatedTime'),
  modelName: $('#modelName'),
  ragStatus: $('#ragStatus'),
  difficultySelect: $('#difficultySelect'),
  difficultyAgentToggle: $('#difficultyAgentToggle'),
  difficultyModeStatus: $('#difficultyModeStatus'),
  dialog: $('#settingsDialog'),
  endpoint: $('#endpointInput'),
  toast: $('#toast')
};

const appState = {
  endpoint: localStorage.getItem('medscope-endpoint') || 'http://127.0.0.1:50042/chat/stream',
  controller: null,
  running: false,
  sessionId: '',
  caseId: '',
  model: 'Qwen3-32B',
  currentStage: -1,
  stages: [],
  experts: [],
  opinions: {},
  events: [],
  decision: {}
};

function escapeHtml(value = '') {
  return String(value).replace(/[&<>'"]/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[char]));
}

function stripThinkContent(value = '') {
  return String(value)
    .replace(/<think\b[^>]*>[\s\S]*?<\/think>/gi, '')
    .replace(/<think\b[^>]*>[\s\S]*?(?=(?:答案|最终答案)[：:])/gi, '')
    .replace(/<\/?think\b[^>]*>/gi, '');
}

function cleanMarkdown(value = '') {
  return stripThinkContent(value)
    .replace(/^#{1,6}\s*/gm, '')
    .replace(/\*\*(.*?)\*\*/g, '$1')
    .replace(/[`*_]/g, '')
    .replace(/\r/g, '')
    .trim();
}

function cleanInline(value = '') {
  return cleanMarkdown(value).replace(/\s+/g, ' ').trim();
}

function nowTime(timestamp = Date.now()) {
  return new Intl.DateTimeFormat('zh-CN', {
    hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false
  }).format(new Date(timestamp));
}

function makeCaseId(timestamp = Date.now()) {
  const date = new Date(timestamp);
  const month = String(date.getMonth() + 1).padStart(2, '0');
  const day = String(date.getDate()).padStart(2, '0');
  const suffix = String(timestamp).slice(-3);
  return `MS-${month}${day}-${suffix}`;
}

function showToast(message) {
  elements.toast.textContent = message;
  elements.toast.classList.add('show');
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => elements.toast.classList.remove('show'), 1800);
}

function setConnection(label, type = 'ok') {
  elements.status.textContent = label;
  elements.statusDot.className = `status-dot${type === 'error' ? ' error' : type === 'loading' ? ' loading' : ''}`;
}

function setRunning(running, label = running ? '正在分析' : '分析完成') {
  appState.running = running;
  elements.start.disabled = running;
  elements.demo.disabled = running;
  elements.runState.textContent = label;
  elements.liveBadge.classList.toggle('is-running', running);
}

function resetAnalysis(sessionId = `web-${Date.now()}`) {
  const timestamp = Date.now();
  appState.sessionId = sessionId;
  appState.caseId = makeCaseId(timestamp);
  appState.currentStage = -1;
  appState.stages = STAGES.map(label => ({ label, status: 'pending', startedAt: null, duration: null }));
  appState.experts = [];
  appState.opinions = {};
  appState.events = [];
  appState.decision = {
    finalized: false,
    title: '等待专家共识',
    summary: '多智能体正在分析病例信息，最终决策将在共识形成后展示。',
    evidence: [],
    option: null,
    confidence: 0,
    updatedAt: null
  };
  renderAll();
}

function renderAll() {
  elements.caseId.textContent = appState.caseId;
  elements.modelName.textContent = appState.model;
  renderStages();
  renderExperts();
  renderEvents();
  renderDecision();
  renderVotes();
}

function renderStages() {
  elements.stageTrack.innerHTML = appState.stages.map((stage, index) => {
    const className = stage.status === 'complete' ? ' is-complete' : stage.status === 'active' ? ' is-active' : '';
    const icon = stage.status === 'complete' ? '✓' : index + 1;
    const detail = stage.status === 'active'
      ? '进行中'
      : stage.status === 'complete'
        ? (stage.duration == null ? '已完成' : `${(stage.duration / 1000).toFixed(1)}s`)
        : '等待';
    return `<button class="stage${className}" data-stage="${index}" type="button"><i>${icon}</i><span>${stage.label}<small>${detail}</small></span></button>`;
  }).join('');
}

function activateStage(index, timestamp = Date.now()) {
  index = Math.max(0, Math.min(index, appState.stages.length - 1));
  if (index < appState.currentStage) return;

  if (appState.currentStage >= 0 && index > appState.currentStage) {
    const previous = appState.stages[appState.currentStage];
    previous.status = 'complete';
    previous.duration = Math.max(0, timestamp - (previous.startedAt || timestamp));
    for (let i = appState.currentStage + 1; i < index; i += 1) {
      appState.stages[i].status = 'complete';
    }
  }

  if (appState.stages[index].status !== 'complete') {
    appState.stages[index].status = 'active';
    appState.stages[index].startedAt ||= timestamp;
  }
  appState.currentStage = index;
  renderStages();
}

function finishStages(timestamp = Date.now()) {
  appState.stages.forEach((stage, index) => {
    if (stage.status === 'active') stage.duration = Math.max(0, timestamp - (stage.startedAt || timestamp));
    stage.status = 'complete';
    if (index < appState.currentStage && stage.duration == null) stage.duration = 0;
  });
  appState.currentStage = appState.stages.length - 1;
  renderStages();
}

function parseExperts(content) {
  const experts = [];
  cleanMarkdown(content).split('\n').forEach(line => {
    const match = line.match(/^\s*\d+[.、]\s*(.+?)\s*[-—]\s*(.+?)\s*[-—]\s*层级结构[：:]\s*(.+?)\s*$/);
    if (!match) return;
    const role = match[1].trim();
    if (!experts.some(item => item.role === role)) {
      experts.push({ role, description: match[2].trim(), hierarchy: match[3].trim() });
    }
  });
  return experts;
}

function isMedicalExpert(name = '') {
  return /(专家|医生)$/.test(name.trim()) && !SYSTEM_EXPERT_PATTERN.test(name);
}

function ensureExpert(name, description = '正在形成专业意见') {
  const role = cleanInline(name);
  if (!role || !isMedicalExpert(role)) return null;
  let expert = appState.experts.find(item => item.role === role || item.role.includes(role) || role.includes(item.role));
  if (!expert) {
    expert = { role, description, hierarchy: '独立' };
    appState.experts.push(expert);
  }
  return expert;
}

function inferStance(content = '') {
  const text = cleanInline(content);
  if (/不建议|不推荐|禁忌|反对|不适合|不可进行|风险大于获益/.test(text)) return 'oppose';
  if (/有条件|谨慎|前提|需复核|需监测|密切监测|若.*则|在.*情况下/.test(text)) return 'conditional';
  return 'support';
}

function extractOption(content = '') {
  const match = String(content).match(/\\boxed\{?([A-E])\}?/i);
  return match ? match[1].toUpperCase() : null;
}

function avatarText(role = '') {
  const simplified = role.replace(/医学|临床|内科|外科|专家|医生|科/g, '');
  return simplified.charAt(0) || role.charAt(0) || '医';
}

function renderExperts() {
  const experts = appState.experts.slice(0, 4);
  if (!experts.length) {
    elements.network.innerHTML = '<div class="network-empty"><div><span>＋</span>等待专家招募结果</div></div>';
    return;
  }

  const opinionCount = experts.filter(expert => appState.opinions[expert.role]).length;
  const paths = experts.map((expert, index) => {
    const stance = appState.opinions[expert.role]?.stance || 'pending';
    const lineClass = stance === 'support' ? 'support' : stance === 'conditional' ? 'clarify' : 'question';
    return `<path class="line ${lineClass}" d="${NETWORK_PATHS[index]}"/>`;
  }).join('');

  const nodes = experts.map((expert, index) => {
    const opinion = appState.opinions[expert.role];
    const stance = opinion?.stance || 'pending';
    const stanceLabel = stance === 'support' ? '支持' : stance === 'conditional' ? '补充' : stance === 'oppose' ? '反对' : '分析中';
    return `<article class="expert-node node-${index + 1}" data-expert="${escapeHtml(expert.role)}">
      <span class="expert-avatar ${AVATAR_COLORS[index]}">${escapeHtml(avatarText(expert.role))}</span>
      <div><strong title="${escapeHtml(expert.role)}">${escapeHtml(expert.role)}</strong><small title="${escapeHtml(expert.description)}">${escapeHtml(expert.description)}</small></div>
      <b class="${stance}">${stanceLabel}</b>
    </article>`;
  }).join('');

  elements.network.innerHTML = `
    <svg class="network-lines" viewBox="0 0 640 270" preserveAspectRatio="none" aria-hidden="true">${paths}</svg>
    ${nodes}
    <div class="consensus-node">
      <span><svg viewBox="0 0 24 24"><path d="M12 3v18M3 12h18"/></svg></span>
      <strong>共识引擎</strong><small>${opinionCount} / ${experts.length} 意见已汇总</small>
    </div>`;
}

function highlightExpert(agent = '') {
  $$('.expert-node').forEach(node => node.classList.remove('is-speaking'));
  const node = $$('.expert-node').find(item => {
    const role = item.dataset.expert || '';
    return agent.includes(role) || role.includes(agent.replace('专家', ''));
  });
  if (node) {
    node.classList.add('is-speaking');
    setTimeout(() => node.classList.remove('is-speaking'), 900);
  }
}

function addEvent(agent, content, timestamp = Date.now(), type = 'output') {
  if (!content) return;
  appState.events.push({ agent: agent || '系统', content, timestamp, type });
  if (appState.events.length > 60) appState.events.shift();
  renderEvents();
  highlightExpert(agent || '');
}

function renderEvents() {
  elements.eventCount.textContent = `${appState.events.length} 个事件`;
  elements.eventList.innerHTML = appState.events.map((event, index) => `
    <article class="event-item">
      <span class="event-icon">${String(index + 1).padStart(2, '0')}</span>
      <div class="event-body"><strong>${escapeHtml(event.agent)}</strong><p title="${escapeHtml(cleanInline(event.content))}">${escapeHtml(cleanInline(event.content))}</p></div>
      <time class="event-time">${nowTime(event.timestamp)}</time>
    </article>`).join('');
  elements.eventList.scrollTop = elements.eventList.scrollHeight;
}

function extractEvidence(content) {
  const cleaned = cleanMarkdown(content);
  let candidates = cleaned.split('\n')
    .map(line => line.trim())
    .filter(line => /^(?:[-•]|\d+[.、）)])\s*/.test(line))
    .map(line => line.replace(/^(?:[-•]|\d+[.、）)])\s*/, '').trim())
    .filter(line => line.length >= 8);

  if (candidates.length < 2) {
    candidates = cleaned.split(/[。；;]/)
      .map(line => line.replace(/^[\s:：-]+/, '').trim())
      .filter(line => line.length >= 12);
  }

  return [...new Set(candidates)].slice(0, 4).map(text => {
    const parts = text.split(/[：:]/);
    const firstClause = parts[0].split(/[，,]/)[0];
    const title = (parts.length > 1 && parts[0].length <= 14 ? parts[0] : firstClause).slice(0, 14);
    return { title: title || '决策依据', detail: text };
  });
}

function extractDecisionTitle(content, option) {
  if (option) return `最终选择：${option}`;
  const markdown = cleanMarkdown(content);
  const text = cleanInline(content).replace(/^答案[：:]\s*/, '');
  const explicitAction = markdown.match(/(?:^|\n)\s*((?:不建议|不推荐|不适合|建议|推荐|适合|可以|不宜)[：:]?[^\n。；]*[。]?)/m);
  if (explicitAction) return explicitAction[1].trim();
  const conclusion = text.match(/(?:最终决策|最终结论|总结|结论)[：:]?\s*([^。；]+[。]?)/);
  if (conclusion) return conclusion[1].trim();
  const firstLine = markdown.split('\n')
    .map(line => line.replace(/^答案[：:]\s*/, '').trim())
    .find(line => line && !/^(答案|最终医疗决策)$/.test(line));
  return firstLine || '综合决策已形成';
}

function updateDecision(content, timestamp = Date.now()) {
  const summary = cleanMarkdown(content).replace(/^\s*答案[：:]\s*/, '').trim();
  const option = extractOption(content);
  appState.decision = {
    finalized: true,
    title: extractDecisionTitle(content, option),
    summary,
    evidence: extractEvidence(content),
    option,
    confidence: 0,
    updatedAt: timestamp,
    raw: content
  };
  renderDecision();
  renderVotes();
}

function calculateVotes() {
  let support = 0;
  let reserve = 0;
  const finalOption = appState.decision.option;

  appState.experts.forEach(expert => {
    const opinion = appState.opinions[expert.role];
    if (!opinion) return;
    if (finalOption && opinion.option) {
      if (finalOption === opinion.option) support += 1;
      else reserve += 1;
    } else if (opinion.stance === 'support') {
      support += 1;
    } else {
      reserve += 1;
    }
  });
  return { support, reserve, responded: support + reserve };
}

function calculateConfidence(votes) {
  if (!appState.decision.finalized) return 0;
  const agreement = votes.responded ? votes.support / votes.responded : 0.65;
  const evidenceBonus = Math.min(appState.decision.evidence.length * 2, 8);
  return Math.min(97, Math.round(58 + agreement * 31 + evidenceBonus));
}

function renderDecision() {
  const decision = appState.decision;
  elements.title.textContent = decision.title;
  elements.summary.textContent = decision.summary;
  elements.evidenceCount.textContent = `${decision.evidence.length} 项`;
  elements.evidenceList.innerHTML = decision.evidence.length
    ? decision.evidence.map((item, index) => `<li><i>${String(index + 1).padStart(2, '0')}</i><p><strong>${escapeHtml(item.title)}</strong><span>${escapeHtml(item.detail)}</span></p></li>`).join('')
    : '<li class="empty-copy">最终结论形成后，将自动提取关键决策依据。</li>';
  elements.updatedTime.textContent = decision.updatedAt ? nowTime(decision.updatedAt) : '尚未';
}

function renderVotes() {
  const votes = calculateVotes();
  const confidence = calculateConfidence(votes);
  const supportWidth = votes.responded ? Math.round(votes.support / votes.responded * 100) : 0;
  elements.expertCount.textContent = `${appState.experts.length} 位专家`;
  elements.supportCount.textContent = votes.support;
  elements.reserveCount.textContent = votes.reserve;
  elements.supportBar.style.setProperty('--width', `${supportWidth}%`);
  elements.reserveBar.style.setProperty('--width', `${votes.responded ? 100 - supportWidth : 100}%`);
  elements.confidence.textContent = confidence;
  elements.confidenceRing.style.setProperty('--value', confidence);
  elements.confidenceLabel.textContent = confidence >= 90 ? '高度一致' : confidence >= 80 ? '较高一致' : confidence > 0 ? '存在分歧' : '等待评估';
  appState.decision.confidence = confidence;
}

function inferStage(message) {
  const text = cleanInline([
    message?.step?.agent,
    message?.step?.description,
    message?.output?.agentName,
    message?.output?.content,
    message?.type
  ].filter(Boolean).join(' '));
  if (message.type === 'final_result' || /最终决策|形成决策|决策阶段/.test(text)) return 4;
  if (/辩论|交互|总结|参与|协作/.test(text)) return 3;
  if (/初步意见|专业分析|独立分析|咨询专业|最终观点|答案收集/.test(text)) return 2;
  if (/招募|团队组建|初始化专家/.test(text)) return 1;
  return 0;
}

function eventTimestamp(message) {
  return message?.step?.timestamp || message?.output?.timestamp || message?.timestamp || Date.now();
}

function processAgentOutput(agent, content) {
  const recruited = parseExperts(content);
  if (recruited.length) appState.experts = recruited;

  const expert = ensureExpert(agent);
  if (expert) {
    appState.opinions[expert.role] = {
      content,
      stance: inferStance(content),
      option: extractOption(content)
    };
  }
  renderExperts();
  renderVotes();
}

function handleStreamEvent(message) {
  if (!message || typeof message !== 'object') return;
  const timestamp = eventTimestamp(message);
  activateStage(inferStage(message), timestamp);
  if (message.model) {
    appState.model = message.model;
    elements.modelName.textContent = appState.model;
  }

  if (message.type === 'agent_step') {
    const details = message.step?.details?.length ? ` · ${message.step.details.join('、')}` : '';
    addEvent(message.step?.agent, `${message.step?.description || ''}${details}`, timestamp, 'step');
  } else if (message.type === 'agent_output') {
    processAgentOutput(message.output?.agentName || '系统', message.output?.content || '');
    addEvent(message.output?.agentName, message.output?.content, timestamp, 'output');
  } else if (message.type === 'final_result') {
    updateDecision(message.content || '', timestamp);
    addEvent('最终决策者', message.content, timestamp, 'final');
    finishStages(timestamp);
  } else if (message.type === 'complete') {
    finishStages(timestamp);
    setRunning(false);
    setConnection('推理服务连接正常');
  } else if (message.type === 'error') {
    throw new Error(message.error || '推理服务返回错误');
  }
}

function parseSseChunk(chunk) {
  const payload = chunk.split(/\r?\n/)
    .filter(line => line.startsWith('data:'))
    .map(line => line.slice(5).trim())
    .join('\n');
  if (payload) handleStreamEvent(JSON.parse(payload));
}

function buildRequestPayload(query) {
  const useDifficultyAgent = elements.difficultyAgentToggle.checked;
  return {
    query,
    id: appState.sessionId,
    enableMultiAgent: true,
    enableDifficultyAgent: useDifficultyAgent,
    difficulty: useDifficultyAgent ? null : elements.difficultySelect.value,
    needRag: $('#ragToggle').checked
  };
}

function updateDifficultyMode() {
  const useDifficultyAgent = elements.difficultyAgentToggle.checked;
  elements.difficultySelect.disabled = useDifficultyAgent;
  elements.difficultyModeStatus.textContent = useDifficultyAgent
    ? '由智能体自动判断'
    : '由用户指定分析深度';
}

async function startAnalysis() {
  const query = elements.input.value.trim();
  if (!query) return showToast('请先输入临床问题');
  if (appState.controller) appState.controller.abort();
  appState.controller = new AbortController();
  resetAnalysis();
  setRunning(true);
  setConnection('正在连接推理服务', 'loading');
  activateStage(0);

  try {
    const response = await fetch(appState.endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
      body: JSON.stringify(buildRequestPayload(query)),
      signal: appState.controller.signal
    });
    if (!response.ok || !response.body) throw new Error(`服务响应异常（HTTP ${response.status}）`);

    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';
    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
      const chunks = buffer.split(/\r?\n\r?\n/);
      buffer = chunks.pop() || '';
      chunks.forEach(parseSseChunk);
      if (done) {
        if (buffer.trim()) parseSseChunk(buffer);
        break;
      }
    }
    if (appState.running) {
      finishStages();
      setRunning(false);
      setConnection('推理服务连接正常');
    }
  } catch (error) {
    if (error.name === 'AbortError') return;
    setRunning(false, '连接失败');
    setConnection('推理服务未连接', 'error');
    addEvent('系统', `${error.message}。可点击“回放演示决策”体验动态绑定。`);
    showToast('后端未连接，可使用演示回放');
  }
}

function buildDemoStream(base = Date.now(), useDifficultyAgent = false) {
  const at = seconds => base + seconds * 1000;
  const stream = [
    { type: 'agent_step', step: { agent: '问题解析器', description: '识别急性缺血性脑卒中、抗凝用药与静脉溶栓评估任务', timestamp: at(0) } },
    { type: 'agent_output', output: { agentName: '专家招募系统', content: '## 专家招募结果\n\n1. 神经内科专家 - 负责卒中诊断、时间窗和神经功能评估。 - 层级结构：独立\n2. 急诊医学专家 - 负责急性期处置与溶栓流程。 - 层级结构：独立\n3. 临床药理专家 - 负责抗凝药物与出血风险评估。 - 层级结构：独立\n4. 循证医学专家 - 负责指南证据核验。 - 层级结构：独立', timestamp: at(1) } },
    { type: 'agent_output', output: { agentName: '神经内科专家', content: '发病 2 小时且 CT 排除出血，支持在无其他禁忌证时进行静脉溶栓。', timestamp: at(3) } },
    { type: 'agent_output', output: { agentName: '急诊医学专家', content: '患者处于治疗时间窗，建议溶栓；治疗前需复核血压、血糖和血小板计数。', timestamp: at(4) } },
    { type: 'agent_output', output: { agentName: '临床药理专家', content: '华法林患者 INR 1.6 未超过 1.7 的排除阈值，但需严密监测出血风险。', timestamp: at(5) } },
    { type: 'agent_output', output: { agentName: '循证医学专家', content: '指南证据支持在 INR ≤ 1.7 且无其他禁忌证时实施阿替普酶静脉溶栓。', timestamp: at(6) } },
    { type: 'agent_step', step: { agent: '辩论协调器', description: '第 2 轮专家辩论完成，开始汇总共识', timestamp: at(8) } },
    { type: 'final_result', model: 'Qwen3-32B', timestamp: at(10), content: '## 最终医疗决策\n\n建议静脉溶栓，同时执行严密出血监测。\n\n1. 时间窗符合：症状出现至今 2 小时，处于 4.5 小时静脉溶栓窗内。\n2. 影像学排除出血：头颅 CT 未发现颅内出血征象。\n3. INR 未超阈值：当前 INR 1.6，低于华法林患者排除阈值 1.7。\n4. 风险控制：治疗前复核血压、血小板与用药史，溶栓后监测出血风险。' },
    { type: 'complete', model: 'Qwen3-32B', timestamp: at(10.2) }
  ];
  if (useDifficultyAgent) {
    stream.splice(
      1,
      0,
      { type: 'agent_step', step: { agent: '难度评估智能体', description: '正在判断问题难度 · 解析问题、评估推理复杂度、选择分析路径', timestamp: at(0.3) } },
      { type: 'agent_output', output: { agentName: '难度评估智能体', content: '## 难度评估结果\n\n智能体判定当前问题为：**困难**', timestamp: at(0.7) } }
    );
  }
  return stream;
}

async function runDemo() {
  if (appState.controller) appState.controller.abort();
  resetAnalysis(`demo-${Date.now()}`);
  setRunning(true, '演示回放中');
  setConnection('正在回放演示数据', 'loading');
  const stream = buildDemoStream(Date.now(), elements.difficultyAgentToggle.checked);
  for (const message of stream) {
    if (!appState.running && message.type !== 'complete') break;
    handleStreamEvent(message);
    await new Promise(resolve => setTimeout(resolve, 360));
  }
  setConnection('演示数据已就绪');
}

function loadDemoSnapshot() {
  resetAnalysis('demo-initial');
  buildDemoStream(Date.now() - 11000).forEach(handleStreamEvent);
  setConnection('演示数据已就绪');
}

function exportDecision() {
  const votes = calculateVotes();
  const payload = {
    caseId: appState.caseId,
    sessionId: appState.sessionId,
    question: elements.input.value,
    generatedAt: new Date().toISOString(),
    model: appState.model,
    stages: appState.stages,
    experts: appState.experts.map(expert => ({ ...expert, opinion: appState.opinions[expert.role] || null })),
    votes,
    decision: appState.decision,
    events: appState.events
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `${payload.caseId}-decision-log.json`;
  link.click();
  URL.revokeObjectURL(url);
  showToast('决策记录已导出');
}

function updateCount() {
  elements.charCount.textContent = elements.input.value.length;
}

elements.input.addEventListener('input', updateCount);
elements.start.addEventListener('click', startAnalysis);
elements.demo.addEventListener('click', runDemo);
$('#exportButton').addEventListener('click', exportDecision);
$('#ragToggle').addEventListener('change', event => {
  elements.ragStatus.textContent = event.target.checked ? '已启用' : '已关闭';
});
elements.difficultyAgentToggle.addEventListener('change', updateDifficultyMode);
$('#settingsButton').addEventListener('click', () => {
  elements.endpoint.value = appState.endpoint;
  elements.dialog.showModal();
});
$('#saveSettings').addEventListener('click', event => {
  event.preventDefault();
  const endpoint = elements.endpoint.value.trim();
  if (!endpoint) return;
  appState.endpoint = endpoint;
  localStorage.setItem('medscope-endpoint', endpoint);
  elements.dialog.close();
  showToast('接口设置已保存');
});
$('#copyCaseId').addEventListener('click', async () => {
  await navigator.clipboard?.writeText(appState.caseId);
  showToast('病例编号已复制');
});
$('#collapseEvents').addEventListener('click', event => {
  elements.eventList.classList.toggle('is-collapsed');
  event.currentTarget.innerHTML = elements.eventList.classList.contains('is-collapsed') ? '展开详情 <span>⌄</span>' : '收起详情 <span>⌃</span>';
});
document.addEventListener('keydown', event => {
  if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') startAnalysis();
});

window.MedScope = {
  getState: () => JSON.parse(JSON.stringify(appState)),
  handleStreamEvent,
  reset: resetAnalysis,
  buildRequestPayload
};

elements.endpoint.value = appState.endpoint;
updateDifficultyMode();
updateCount();
loadDemoSnapshot();
