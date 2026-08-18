const state = {
  sessionId: localStorage.getItem("livingTabletopSession"),
  view: null,
  busy: false,
  developerOpen: false,
  scenarios: [],
  selectedScenarioId: null,
  llmConfig: null,
  modelBusy: false,
  narrative: {
    sequenceId: null,
    stateVersion: null,
    beats: [],
    cursor: 0,
    status: "ready",
    skipped: false,
    interrupted: false,
    decisionUnlocked: false,
    renderedBeatId: null,
    pollToken: 0,
    pollTimer: null,
    endWaitTimer: null,
  },
};

const $ = (selector) => document.querySelector(selector);
const narrativeProgressKey = "livingTabletopNarrativeProgress";
const elements = {
  modal: $("#start-modal"),
  start: $("#start-game"),
  startName: $("#start-name"),
  scenarioPicker: $("#scenario-picker"),
  scenarioSource: $("#scenario-source"),
  startHeadline: $("#start-headline"),
  startDescription: $("#start-description"),
  sessionNote: $("#session-note"),
  caseLabel: $("#case-label"),
  scenarioTitle: $("#scenario-title"),
  newGame: $("#new-game"),
  narrative: $("#narrative"),
  narrativeControls: $("#narrative-controls"),
  narrativeProgress: $("#narrative-progress"),
  narrativePending: $("#narrative-pending"),
  narrativeContinue: $("#continue-narrative"),
  narrativeSkip: $("#skip-narrative"),
  narrativeInterrupt: $("#interrupt-performance"),
  sceneName: $("#scene-name"),
  sceneStage: $("#scene-stage"),
  sceneDescription: $("#scene-description"),
  sceneMode: $("#scene-mode"),
  sceneExits: $("#scene-exits"),
  sceneObjects: $("#scene-objects"),
  sceneActors: $("#scene-actors"),
  scenePlayer: $("#scene-player"),
  time: $("#world-time"),
  actions: $("#suggested-actions"),
  decisionDivider: $("#decision-divider"),
  dialogueBlock: $("#dialogue-block"),
  dialogueOptions: $("#dialogue-options"),
  form: $("#free-action-form"),
  formLabel: $("#free-action-label"),
  formNote: $("#free-action-note"),
  input: $("#free-action-input"),
  submit: $("#submit-action"),
  clarification: $("#clarification"),
  check: $("#check-result"),
  ending: $("#ending-banner"),
  save: $("#save-state"),
  playerName: $("#player-name"),
  status: $("#session-status"),
  hpText: $("#hp-text"),
  hpMeter: $("#hp-meter"),
  sanityText: $("#sanity-text"),
  sanityMeter: $("#sanity-meter"),
  luckText: $("#luck-text"),
  luckMeter: $("#luck-meter"),
  conditions: $("#investigator-conditions"),
  characteristics: $("#characteristics"),
  skills: $("#skills"),
  ruleChoice: $("#rule-choice"),
  npcs: $("#present-npcs"),
  inventory: $("#inventory"),
  clueCount: $("#clue-count"),
  clues: $("#clue-list"),
  devToggle: $("#developer-toggle"),
  devClose: $("#developer-close"),
  devPanel: $("#developer-console"),
  modelToggle: $("#model-toggle"),
  modelPanel: $("#model-panel"),
  modelDot: $("#model-dot"),
  modelSummary: $("#model-summary"),
  modelMode: $("#model-mode"),
  localModel: $("#local-model-select"),
  remoteModel: $("#remote-model-select"),
  localModelStatus: $("#local-model-status"),
  remoteModelStatus: $("#remote-model-status"),
  modelLastUsed: $("#model-last-used"),
  modelMessage: $("#model-message"),
  modelRefresh: $("#model-refresh"),
  modelProbe: $("#model-probe"),
  modelSave: $("#model-save"),
  toast: $("#toast"),
};

const actionIcons = {
  investigate: "⌕",
  social: "◌",
  risk: "△",
  move: "→",
  other: "·",
};

const outcomeLabels = {
  CRITICAL: "大成功",
  EXTREME: "极难成功",
  HARD: "困难成功",
  SUCCESS: "成功",
  FAILURE: "失败",
  FUMBLE: "大失败",
  AUTOMATIC: "自动成功",
  INTERRUPTED: "被中断",
};

const difficultyLabels = { regular: "常规", hard: "困难", extreme: "极难" };
const characteristicLabels = { str: "力量", con: "体质", siz: "体型", dex: "敏捷", app: "外貌", int: "智力", pow: "意志", edu: "教育" };
const skillLabels = {
  observation: "侦查", research: "图书馆使用", charm: "魅惑", law: "法律", agility: "敏捷",
  force: "力量", fight: "格斗", dodge: "闪避", first_aid: "急救", psychology: "心理学",
  medicine: "医学", persuasion: "说服", deception: "话术", athletics: "运动", stealth: "潜行",
  occult: "神秘学", fighting: "格斗",
};

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
  })[char]);
}

class APIError extends Error {
  constructor(message, { kind = "http", status = null } = {}) {
    super(message);
    this.name = "APIError";
    this.kind = kind;
    this.status = status;
  }
}

async function api(path, options = {}) {
  let response;
  try {
    response = await fetch(path, {
      headers: { "Content-Type": "application/json", ...(options.headers || {}) },
      ...options,
    });
  } catch (error) {
    const offline = typeof navigator !== "undefined" && navigator.onLine === false;
    const message = offline
      ? "当前设备处于离线状态。请恢复网络后重新提交；刚才的操作尚未发送。"
      : "无法连接游戏服务。请检查网络，或确认本地游戏服务仍在运行；刚才的操作尚未发送。";
    throw new APIError(message, { kind: "network" });
  }
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    const fallback = response.status >= 500
      ? `游戏服务暂时不可用 (${response.status})，请稍后重试。`
      : `请求失败 (${response.status})`;
    throw new APIError(body.detail || fallback, { status: response.status });
  }
  return response.json();
}

function clearInlineNotice() {
  elements.clarification.textContent = "";
  elements.clarification.classList.add("hidden");
  elements.clarification.classList.remove("request-error");
}

function showInlineNotice(message, { error = false } = {}) {
  elements.clarification.textContent = message;
  elements.clarification.classList.remove("hidden");
  elements.clarification.classList.toggle("request-error", error);
}

function showStartError(message) {
  elements.sessionNote.textContent = message;
  elements.sessionNote.classList.add("request-error");
}

function modelShortName(model) {
  if (!model) return "未配置";
  return model
    .replace("qwen3.5:9b-q4_K_M", "Qwen3.5 9B")
    .replace("gpt-5.6-luna-openai-compact", "GPT-5.6 Luna");
}

function fillModelSelect(select, provider) {
  if (!select || !provider) return;
  const models = [...new Set([provider.model, ...(provider.models || [])].filter(Boolean))];
  select.innerHTML = models.length
    ? models.map((model) => `<option value="${escapeHtml(model)}">${escapeHtml(model)}</option>`).join("")
    : '<option value="">没有发现可用模型</option>';
  select.value = provider.model || "";
  select.disabled = !models.length;
}

function providerStatus(provider) {
  if (!provider) return { label: "未配置", className: "error" };
  if (provider.last_error) return { label: provider.last_error, className: "error" };
  if (provider.last_success_at) return { label: "生成可用", className: "online" };
  if (provider.state === "online") return { label: "模型目录可达", className: "online" };
  if (provider.state === "cooldown") return { label: "短暂冷却中", className: "error" };
  if (provider.state === "disabled") return { label: "未启用", className: "error" };
  return { label: "尚未确认", className: "" };
}

function renderLLMConfiguration(config) {
  state.llmConfig = config;
  const local = config.providers?.local;
  const remote = config.providers?.remote;
  elements.modelMode.value = config.mode;
  fillModelSelect(elements.localModel, local);
  fillModelSelect(elements.remoteModel, remote);

  const localStatus = providerStatus(local);
  const remoteStatus = providerStatus(remote);
  elements.localModelStatus.textContent = localStatus.label;
  elements.localModelStatus.className = localStatus.className;
  elements.remoteModelStatus.textContent = remoteStatus.label;
  elements.remoteModelStatus.className = remoteStatus.className;

  const last = config.last_used;
  elements.modelLastUsed.textContent = last
    ? `上次命中：${last.provider === "local" ? "本地" : "远程"} · ${last.model} · ${last.latency_ms} ms`
    : "尚无本轮调用记录";

  const modeLabels = { auto: "自动", local: "本地", remote: "远程" };
  const selectedProvider = config.mode === "remote" ? remote : local;
  const selectedHasError = selectedProvider?.last_error;
  const anyOnline = [local, remote].some((item) => item?.state === "online" && !item.last_error);
  const visualState = selectedHasError
    ? (config.mode === "auto" && anyOnline ? "degraded" : "offline")
    : (anyOnline ? "online" : "offline");
  elements.modelDot.dataset.state = visualState;
  const summaryModel = last?.model || selectedProvider?.model;
  elements.modelSummary.textContent = `${modeLabels[config.mode] || "模型"} · ${modelShortName(summaryModel)}`;
}

function setModelBusy(busy) {
  state.modelBusy = busy;
  elements.modelRefresh.disabled = busy;
  elements.modelProbe.disabled = busy;
  elements.modelSave.disabled = busy;
}

async function loadLLMConfiguration(refresh = false) {
  const config = await api(`/api/llm/config${refresh ? "?refresh=true" : ""}`);
  renderLLMConfiguration(config);
  return config;
}

async function saveLLMConfiguration({ notify = true } = {}) {
  if (state.modelBusy) return null;
  setModelBusy(true);
  elements.modelMessage.classList.remove("error");
  elements.modelMessage.textContent = "正在应用模型设置……";
  try {
    const config = await api("/api/llm/config", {
      method: "PUT",
      body: JSON.stringify({
        mode: elements.modelMode.value,
        local_model: elements.localModel.value || null,
        remote_model: elements.remoteModel.value || null,
      }),
    });
    renderLLMConfiguration(config);
    elements.modelMessage.textContent = "设置已应用，下一次调用立即生效。";
    if (notify) toast("模型设置已应用");
    return config;
  } catch (error) {
    elements.modelMessage.textContent = error.message;
    elements.modelMessage.classList.add("error");
    return null;
  } finally {
    setModelBusy(false);
  }
}

async function probeSelectedModel() {
  const saved = await saveLLMConfiguration({ notify: false });
  if (!saved) return;
  setModelBusy(true);
  elements.modelMessage.classList.remove("error");
  elements.modelMessage.textContent = "正在进行真实生成测试……";
  try {
    const provider = elements.modelMode.value === "auto" ? null : elements.modelMode.value;
    const result = await api("/api/llm/probe", {
      method: "POST",
      body: JSON.stringify({ provider }),
    });
    elements.modelMessage.textContent = `测试成功：${result.provider === "local" ? "本地" : "远程"} · ${result.model} · ${result.latency_ms} ms`;
    await loadLLMConfiguration(false);
  } catch (error) {
    elements.modelMessage.textContent = error.message;
    elements.modelMessage.classList.add("error");
    await loadLLMConfiguration(false).catch(() => {});
  } finally {
    setModelBusy(false);
  }
}

function setModelPanel(open) {
  elements.modelPanel.classList.toggle("hidden", !open);
  elements.modelToggle.setAttribute("aria-expanded", String(open));
  if (open && !state.llmConfig) {
    setModelBusy(true);
    loadLLMConfiguration(true)
      .catch((error) => {
        elements.modelMessage.textContent = error.message;
        elements.modelMessage.classList.add("error");
      })
      .finally(() => setModelBusy(false));
  }
}

function setBusy(busy) {
  state.busy = busy;
  document.body.classList.toggle("loading", busy);
  elements.submit.disabled = busy;
  document.querySelectorAll(".action-button").forEach((button) => { button.disabled = busy; });
  document.querySelectorAll(".dialogue-button").forEach((button) => { button.disabled = busy; });
  document.querySelectorAll(".rule-choice-button").forEach((button) => { button.disabled = busy; });
  document.querySelectorAll(".scene-action").forEach((button) => { button.disabled = busy || performanceActive(); });
  elements.narrativeContinue.disabled = busy;
  elements.narrativeSkip.disabled = busy;
  elements.narrativeInterrupt.disabled = busy;
  elements.save.textContent = busy ? "世界正在运行…" : "世界已保存";
  if (!busy) paintNarrativeBeat();
}

function stopNarrativePolling() {
  state.narrative.pollToken += 1;
  if (state.narrative.pollTimer) window.clearTimeout(state.narrative.pollTimer);
  state.narrative.pollTimer = null;
  if (state.narrative.endWaitTimer) window.clearTimeout(state.narrative.endWaitTimer);
  state.narrative.endWaitTimer = null;
}

function readNarrativeProgress(sequence) {
  try {
    const saved = JSON.parse(sessionStorage.getItem(narrativeProgressKey) || "null");
    if (!saved || saved.sequenceId !== sequence.id || saved.stateVersion !== sequence.state_version) return null;
    return saved;
  } catch (_error) {
    return null;
  }
}

function saveNarrativeProgress() {
  const playback = state.narrative;
  if (!playback.sequenceId) return;
  try {
    sessionStorage.setItem(narrativeProgressKey, JSON.stringify({
      sequenceId: playback.sequenceId,
      stateVersion: playback.stateVersion,
      cursor: playback.cursor,
      skipped: playback.skipped,
      interrupted: playback.interrupted,
      decisionUnlocked: playback.decisionUnlocked,
    }));
  } catch (_error) {
    // Storage may be unavailable in privacy-restricted browser contexts.
  }
}

function scheduleNarrativeEndUnlock() {
  const playback = state.narrative;
  const waitingAtEnd = playback.status === "pending"
    && playback.cursor >= playback.beats.length - 1
    && !playback.skipped
    && !playback.interrupted
    && !playback.decisionUnlocked;
  if (!waitingAtEnd || playback.endWaitTimer) return;
  playback.endWaitTimer = window.setTimeout(() => {
    playback.endWaitTimer = null;
    if (playback.status !== "pending" || playback.cursor < playback.beats.length - 1) return;
    playback.decisionUnlocked = true;
    stopNarrativePolling();
    paintNarrativeBeat();
  }, 1200);
}

function performanceActive() {
  const playback = state.narrative;
  if (playback.skipped || playback.interrupted) return false;
  if (playback.decisionUnlocked) return false;
  return playback.status === "pending" || playback.cursor < playback.beats.length - 1;
}

function syncDecisionVisibility() {
  if (!state.view) return;
  const active = performanceActive();
  const ended = state.view.status !== "ACTIVE";
  const hasPrompt = Boolean(state.view.rule_prompt);
  const interrupting = state.narrative.interrupted && !ended && !hasPrompt;
  const dialogueOptions = state.view.dialogue_options || [];
  const showDecisions = !active && !ended && !hasPrompt && !interrupting;

  elements.decisionDivider.classList.toggle("hidden", !showDecisions);
  elements.dialogueBlock.classList.toggle("hidden", !showDecisions || dialogueOptions.length === 0);
  elements.actions.classList.toggle("hidden", !showDecisions);
  elements.form.classList.toggle("hidden", active || ended || hasPrompt);
  elements.ruleChoice.classList.toggle("hidden", active || !hasPrompt);
  elements.ending.classList.toggle("hidden", active || !ended);
  elements.narrativeInterrupt.classList.toggle("hidden", !active || hasPrompt);
  elements.sceneStage.classList.toggle("performance-active", active);
  elements.sceneMode.textContent = active ? "剧情演出中" : showDecisions ? "可交互场景" : "场景观察";
  document.querySelectorAll(".scene-action").forEach((button) => {
    button.disabled = state.busy || active || ended || hasPrompt;
  });

  if (interrupting) {
    elements.formLabel.textContent = "你要如何打断当前演出？";
    elements.formNote.textContent = "直接描述你现在说的话或采取的动作；尚未发生的后续剧情不会被预先展示。";
    elements.input.placeholder = "例如：我立刻打断她：“等等，你刚才说门是从里面锁上的？”";
  } else if (dialogueOptions.length) {
    elements.formLabel.textContent = "或者，用自己的话开口或行动";
    elements.formNote.textContent = "你不必照着选项说，也可以离开现场或完全偏离主线。";
    elements.input.placeholder = "例如：我直视对方说：“先别隐瞒，把你亲眼看见的都告诉我。”";
  } else {
    elements.formLabel.textContent = "或者，做任何你想做的事";
    elements.formNote.textContent = "你可以离开现场或偏离主线；KP 会判断结果，世界仍会继续运转。";
    elements.input.placeholder = state.view.scenario.presentation.free_action_placeholder;
  }
}

function scenePosition(position) {
  return `left:${Number(position?.x || 50)}%;top:${Number(position?.y || 50)}%`;
}

function scenePiece({ className, name, position, interaction, icon = "·", sprite = false }) {
  const tag = interaction ? "button" : "div";
  const action = interaction ? ` data-scene-action-id="${escapeHtml(interaction.action_id)}"` : "";
  const aria = interaction ? ` aria-label="${escapeHtml(interaction.label)}" title="${escapeHtml(interaction.label)}"` : "";
  const body = sprite
    ? '<span class="sprite" aria-hidden="true"></span>'
    : `<span class="piece-icon" aria-hidden="true"><i>${escapeHtml(icon)}</i></span>`;
  return `<${tag} class="scene-piece ${escapeHtml(className)}${interaction ? " scene-action" : ""}" style="${scenePosition(position)}"${action}${aria}>
    ${body}<span class="piece-label">${escapeHtml(name)}</span>
  </${tag}>`;
}

function sceneIcon(item) {
  const type = item.interaction?.type;
  if (type === "SEARCH" || type === "EXAMINE") return "⌕";
  if (type === "TAKE") return "◇";
  if (type === "FORCE" || type === "CONFRONT") return "!";
  if (item.kind === "item") return "◆";
  return "▣";
}

function renderScene(visual) {
  const safe = visual || { archetype: "void", actors: [], objects: [], hotspots: [], exits: [], player: null };
  elements.sceneStage.dataset.archetype = safe.archetype || "interior";
  elements.sceneStage.classList.toggle("danger", Boolean(safe.danger));
  elements.sceneDescription.textContent = safe.description || state.view?.scene?.description || "眼前的场景仍在形成。";
  elements.sceneActors.innerHTML = (safe.actors || []).map((actor) => scenePiece({
    className: `${actor.kind === "creature" ? "creature-piece" : "actor-piece"}`,
    name: actor.name,
    position: actor.position,
    interaction: actor.interaction,
    sprite: true,
  })).join("");
  const objects = (safe.objects || []).map((item) => scenePiece({
    className: "object-piece",
    name: item.name,
    position: item.position,
    interaction: item.interaction,
    icon: sceneIcon(item),
  }));
  const hotspots = (safe.hotspots || []).map((item) => scenePiece({
    className: "scene-hotspot",
    name: item.name,
    position: item.position,
    interaction: item.interaction,
    icon: sceneIcon(item),
  }));
  elements.sceneObjects.innerHTML = [...objects, ...hotspots].join("");
  elements.sceneExits.innerHTML = (safe.exits || []).map((exit) => scenePiece({
    className: `scene-exit edge-${exit.edge}${exit.available ? "" : " unavailable"}`,
    name: exit.label,
    position: exit.position,
    interaction: exit.interaction,
    icon: exit.edge === "north" ? "↑" : exit.edge === "east" ? "→" : exit.edge === "south" ? "↓" : "←",
  })).join("");
  elements.scenePlayer.innerHTML = safe.player ? scenePiece({
    className: "player-piece",
    name: safe.player.name,
    position: safe.player.position,
    interaction: null,
    sprite: true,
  }) : "";
  document.querySelectorAll("[data-scene-action-id]").forEach((button) => {
    button.addEventListener("click", () => act({ action_id: button.dataset.sceneActionId }));
  });
}

function paintNarrativeBeat() {
  const playback = state.narrative;
  const beat = playback.beats[playback.cursor];
  if (beat && playback.renderedBeatId !== beat.id) {
    elements.narrative.textContent = beat.text;
    elements.narrative.style.animation = "none";
    void elements.narrative.offsetWidth;
    elements.narrative.style.animation = "";
    playback.renderedBeatId = beat.id;
  }
  const hasNext = playback.cursor < playback.beats.length - 1;
  const pendingAtEnd = playback.status === "pending" && !hasNext;
  if (pendingAtEnd) scheduleNarrativeEndUnlock();
  else if (playback.endWaitTimer) {
    window.clearTimeout(playback.endWaitTimer);
    playback.endWaitTimer = null;
  }
  const showControls = performanceActive();
  elements.narrativeControls.classList.toggle("hidden", !showControls);
  elements.narrativeProgress.textContent = playback.beats.length
    ? `${Math.min(playback.cursor + 1, playback.beats.length)} / ${playback.beats.length}`
    : "";
  elements.narrativePending.classList.toggle("hidden", playback.status !== "pending" || playback.decisionUnlocked);
  elements.narrativeContinue.disabled = state.busy || !hasNext;
  elements.narrativeContinue.textContent = pendingAtEnd ? "生成中…" : hasNext ? "继续" : "已读完";
  syncDecisionVisibility();
  saveNarrativeProgress();
}

function scheduleNarrativePoll(token, attempt = 0) {
  if (!state.sessionId || state.narrative.status !== "pending" || state.narrative.skipped || state.narrative.interrupted || state.narrative.decisionUnlocked) return;
  state.narrative.pollTimer = window.setTimeout(async () => {
    if (token !== state.narrative.pollToken) return;
    try {
      const sequenceId = state.narrative.sequenceId;
      const data = await api(`/api/sessions/${state.sessionId}/narrative/${encodeURIComponent(sequenceId)}`);
      if (token !== state.narrative.pollToken || data.superseded || data.id !== sequenceId) return;
      state.narrative.beats = data.beats || state.narrative.beats;
      state.narrative.status = data.status;
      state.narrative.cursor = Math.min(state.narrative.cursor, Math.max(0, state.narrative.beats.length - 1));
      paintNarrativeBeat();
      if (data.status === "pending") scheduleNarrativePoll(token, 0);
    } catch (error) {
      if (token !== state.narrative.pollToken) return;
      if (attempt < 2) {
        scheduleNarrativePoll(token, attempt + 1);
      } else {
        state.narrative.status = "fallback";
        paintNarrativeBeat();
        showInlineNotice("世界状态已经保存，但后台叙事暂时无法更新；你仍可继续行动。", { error: true });
      }
    }
  }, attempt ? 1400 : 700);
}

function installNarrativeSequence(sequence, fallbackText = "") {
  stopNarrativePolling();
  const safeSequence = sequence || {
    id: `legacy-${Date.now()}`,
    state_version: state.view?.version,
    status: "ready",
    beats: [{ id: "legacy-beat", text: fallbackText, source: "authored", skippable: false }],
  };
  const sameSequence = state.narrative.sequenceId === safeSequence.id
    && state.narrative.stateVersion === safeSequence.state_version;
  if (sameSequence) {
    state.narrative.beats = safeSequence.beats || state.narrative.beats;
    state.narrative.status = safeSequence.status || state.narrative.status;
    state.narrative.cursor = Math.min(
      state.narrative.cursor,
      Math.max(0, state.narrative.beats.length - 1),
    );
    paintNarrativeBeat();
    if (state.narrative.status === "pending" && !state.narrative.decisionUnlocked) {
      const token = state.narrative.pollToken;
      scheduleNarrativePoll(token);
    }
    return;
  }
  state.narrative.sequenceId = safeSequence.id;
  state.narrative.stateVersion = safeSequence.state_version;
  state.narrative.beats = safeSequence.beats || [];
  const savedProgress = readNarrativeProgress(safeSequence);
  state.narrative.cursor = Math.min(
    savedProgress?.cursor || 0,
    Math.max(0, state.narrative.beats.length - 1),
  );
  state.narrative.status = safeSequence.status || "ready";
  state.narrative.skipped = Boolean(savedProgress?.skipped);
  state.narrative.interrupted = Boolean(savedProgress?.interrupted);
  state.narrative.decisionUnlocked = Boolean(savedProgress?.decisionUnlocked);
  state.narrative.renderedBeatId = null;
  paintNarrativeBeat();
  if (state.narrative.status === "pending") {
    const token = state.narrative.pollToken;
    scheduleNarrativePoll(token);
  }
}

function toast(message) {
  elements.toast.textContent = message;
  elements.toast.classList.add("show");
  window.setTimeout(() => elements.toast.classList.remove("show"), 2200);
}

function render(view) {
  state.view = view;
  const presentation = view.scenario.presentation;
  elements.caseLabel.textContent = presentation.case_label;
  elements.scenarioTitle.textContent = view.scenario.title;
  document.title = `Living Tabletop · ${view.scenario.title}`;
  elements.sceneName.textContent = view.scene.name;
  renderScene(view.scene.visual);
  elements.time.textContent = view.time_label;
  installNarrativeSequence(view.narrative_sequence, view.narrative);
  elements.playerName.textContent = view.player.name;
  elements.hpText.textContent = `${view.player.hp} / ${view.player.max_hp}`;
  elements.hpMeter.style.width = `${(view.player.hp / view.player.max_hp) * 100}%`;
  elements.sanityText.textContent = `${view.player.sanity} / ${view.player.max_sanity}`;
  elements.sanityMeter.style.width = `${(view.player.sanity / view.player.max_sanity) * 100}%`;
  elements.luckText.textContent = `${view.player.luck} / ${view.player.max_luck}`;
  elements.luckMeter.style.width = `${(view.player.luck / view.player.max_luck) * 100}%`;
  elements.conditions.innerHTML = view.player.conditions.length
    ? view.player.conditions.map((condition) => `<span>${escapeHtml(condition)}</span>`).join("")
    : '<span class="condition-ok">状态稳定</span>';
  elements.characteristics.innerHTML = Object.entries(view.player.characteristics)
    .map(([name, value]) => `<div><span>${escapeHtml(characteristicLabels[name] || name.toUpperCase())}</span><strong>${value}</strong></div>`)
    .join("");
  elements.skills.innerHTML = Object.entries(view.player.skills)
    .sort((left, right) => right[1] - left[1])
    .map(([name, value]) => `<span>${escapeHtml(skillLabels[name] || name)} <strong>${value}</strong></span>`)
    .join("");

  const statusLabels = { ACTIVE: "调查中", WON: "已完成", LOST: "失败", ESCAPED: "已撤离" };
  elements.status.textContent = statusLabels[view.status] || view.status;

  elements.npcs.innerHTML = view.scene.present_npcs.length
    ? view.scene.present_npcs.map((npc) => `<span class="tag">${escapeHtml(npc.name)} · ${escapeHtml(npc.role || "")}</span>`).join("")
    : '<span class="empty-copy">此处没有其他人</span>';
  elements.inventory.innerHTML = view.player.inventory.length
    ? view.player.inventory.map((item) => `<span class="tag">${escapeHtml(item)}</span>`).join("")
    : '<span class="empty-copy">空</span>';

  elements.clueCount.textContent = view.clues.length;
  elements.clues.innerHTML = view.clues.length
    ? view.clues.map((clue) => `<article class="clue-card"><h4>${escapeHtml(clue.title)}</h4><p>${escapeHtml(clue.description)}</p></article>`).join("")
    : '<p class="empty-copy">可靠的线索会记录在这里。</p>';

  elements.actions.innerHTML = view.suggested_actions.map((action) => `
    <button class="action-button" type="button" data-action-id="${escapeHtml(action.id)}" data-risk="${escapeHtml(action.risk)}">
      <span class="action-icon">${actionIcons[action.category] || "·"}</span>
      <span class="action-label">${escapeHtml(action.label)}</span>
      <span class="action-meta">${action.duration_minutes} MIN</span>
    </button>`).join("");
  document.querySelectorAll(".action-button").forEach((button) => {
    button.addEventListener("click", () => act({ action_id: button.dataset.actionId }));
  });

  const dialogueOptions = view.dialogue_options || [];
  elements.dialogueOptions.innerHTML = dialogueOptions.map((option) => `
    <button class="dialogue-button" type="button" data-action-id="${escapeHtml(option.action_id)}" data-utterance="${escapeHtml(option.text)}">
      <span>${escapeHtml(option.text)}</span>
      <small>${option.duration_minutes} MIN</small>
    </button>`).join("");
  document.querySelectorAll(".dialogue-button").forEach((button) => {
    button.addEventListener("click", () => act({ action_id: button.dataset.actionId, utterance: button.dataset.utterance }));
  });

  clearInlineNotice();
  const resolution = view.last_resolution;
  if (resolution?.needs_clarification) {
    showInlineNotice(resolution.clarification);
  }
  const displayedCheck = resolution?.check || view.rule_prompt?.check;
  if (displayedCheck?.required) {
    const check = displayedCheck;
    const dice = check.bonus_dice > 0 ? `奖励骰 ×${check.bonus_dice}` : check.bonus_dice < 0 ? `惩罚骰 ×${Math.abs(check.bonus_dice)}` : "";
    const candidates = check.candidates?.length > 1 ? ` [${check.candidates.join(" / ")}]` : "";
    const opponent = check.opponent
      ? `<span class="opposed-roll">对抗 ${escapeHtml(check.opponent.label)}：${check.opponent.roll} · ${escapeHtml(outcomeLabels[check.opponent.outcome] || check.opponent.outcome)}</span>`
      : "";
    const outcomeText = check.opponent && !check.succeeded
      ? `对抗失败（自身${outcomeLabels[check.outcome] || check.outcome}）`
      : outcomeLabels[check.outcome] || check.outcome;
    elements.check.innerHTML = `
      <span>${escapeHtml(skillLabels[check.skill] || check.skill.toUpperCase())} · ${escapeHtml(difficultyLabels[check.difficulty] || check.difficulty)}</span>
      <strong>${check.roll} / ${check.target}${escapeHtml(candidates)}</strong>
      <span>${escapeHtml(outcomeText)}${dice ? ` · ${escapeHtml(dice)}` : ""}${check.pushed ? " · 孤注一掷" : ""}${check.luck_spent ? ` · 幸运 -${check.luck_spent}` : ""}</span>
      ${opponent}`;
    elements.check.classList.remove("hidden");
  } else if (resolution?.interrupted) {
    elements.check.innerHTML = "<strong>行动被世界事件中断</strong>";
    elements.check.classList.remove("hidden");
  } else {
    elements.check.innerHTML = "";
    elements.check.classList.add("hidden");
  }

  if (resolution?.sanity_check) {
    const sanity = resolution.sanity_check;
    elements.check.insertAdjacentHTML(
      "beforeend",
      `<span class="sanity-roll">SAN ${sanity.roll} / ${sanity.target} · ${sanity.succeeded ? "成功" : "失败"} · 损失 ${sanity.loss}</span>`,
    );
    elements.check.classList.remove("hidden");
  }

  const prompt = view.rule_prompt;
  if (prompt) {
    const choiceLabels = {
      accept_failure: "接受失败",
      spend_luck: `消耗 ${prompt.luck_cost} 点幸运`,
      push_roll: "孤注一掷",
    };
    elements.ruleChoice.innerHTML = `
      <div><strong>检定失败，结果尚未落定</strong><span>孤注一掷会再次耗时；若仍失败，后果将升级。</span></div>
      <div class="rule-choice-actions">${prompt.choices.map((choice) => `
        <button class="rule-choice-button" type="button" data-rule-choice="${escapeHtml(choice)}">${escapeHtml(choiceLabels[choice] || choice)}</button>`).join("")}</div>`;
    elements.ruleChoice.classList.remove("hidden");
    document.querySelectorAll(".rule-choice-button").forEach((button) => {
      button.addEventListener("click", () => act({ rule_choice: button.dataset.ruleChoice }));
    });
  } else {
    elements.ruleChoice.classList.add("hidden");
    elements.ruleChoice.innerHTML = "";
  }

  const ended = view.status !== "ACTIVE";
  elements.ending.textContent = ended ? `CASE CLOSED · ${statusLabels[view.status]}` : "";
  elements.ending.classList.toggle("hidden", !ended);
  if (ended) elements.actions.innerHTML = "";
  syncDecisionVisibility();
}

async function createGame() {
  const playerName = elements.startName.value.trim() || "调查员";
  if (!state.selectedScenarioId) {
    toast("请先选择一个调查模组");
    return;
  }
  setBusy(true);
  try {
    const view = await api("/api/sessions", {
      method: "POST",
      body: JSON.stringify({
        player_name: playerName,
        seed: 1927,
        scenario_id: state.selectedScenarioId,
      }),
    });
    state.sessionId = view.session_id;
    localStorage.setItem("livingTabletopSession", state.sessionId);
    elements.modal.classList.add("hidden");
    render(view);
  } catch (error) {
    showStartError(error.message);
    toast("无法开始调查");
  } finally {
    setBusy(false);
  }
}

function selectScenario(scenarioId) {
  const scenario = state.scenarios.find((item) => item.id === scenarioId);
  if (!scenario) return;
  state.selectedScenarioId = scenario.id;
  document.querySelectorAll(".scenario-option").forEach((button) => {
    button.setAttribute("aria-pressed", String(button.dataset.scenarioId === scenario.id));
  });
  const presentation = scenario.presentation;
  elements.startHeadline.textContent = presentation.headline;
  elements.startDescription.textContent = presentation.description;
  elements.start.textContent = presentation.start_button;
  elements.sessionNote.textContent = presentation.session_note;
  elements.sessionNote.classList.remove("request-error");
  if (scenario.source) {
    elements.scenarioSource.innerHTML = `剧情结构改编自 <a href="${escapeHtml(scenario.source.url)}" target="_blank" rel="noreferrer">${escapeHtml(scenario.source.title)}</a> · ${escapeHtml(scenario.source.rights_note)}`;
    elements.scenarioSource.classList.remove("hidden");
  } else {
    elements.scenarioSource.classList.add("hidden");
    elements.scenarioSource.textContent = "";
  }
}

async function loadScenarios() {
  const data = await api("/api/scenarios");
  state.scenarios = data.scenarios;
  elements.scenarioPicker.innerHTML = state.scenarios.map((scenario) => `
    <button class="scenario-option" type="button" data-scenario-id="${escapeHtml(scenario.id)}" aria-pressed="false">
      <span>${escapeHtml(scenario.title)}</span>
      <small>${escapeHtml(scenario.subtitle)}</small>
    </button>`).join("");
  document.querySelectorAll(".scenario-option").forEach((button) => {
    button.addEventListener("click", () => selectScenario(button.dataset.scenarioId));
  });
  const defaultScenario = state.scenarios.find((scenario) => scenario.default) || state.scenarios[0];
  if (defaultScenario) selectScenario(defaultScenario.id);
}

async function loadGame() {
  if (!state.sessionId) return;
  try {
    const view = await api(`/api/sessions/${state.sessionId}`);
    elements.modal.classList.add("hidden");
    render(view);
  } catch (error) {
    if (error.status === 404) {
      localStorage.removeItem("livingTabletopSession");
      state.sessionId = null;
    } else {
      showStartError(error.message);
    }
    elements.modal.classList.remove("hidden");
  }
}

async function act(payload) {
  if (!state.sessionId || state.busy) return;
  stopNarrativePolling();
  clearInlineNotice();
  setBusy(true);
  try {
    const requestPayload = payload.rule_choice ? payload : { ...payload, interactive_rules: true };
    const view = await api(`/api/sessions/${state.sessionId}/actions`, {
      method: "POST",
      body: JSON.stringify(requestPayload),
    });
    render(view);
    elements.input.value = "";
    loadLLMConfiguration(false).catch(() => {});
    if (state.developerOpen) await refreshDeveloper();
  } catch (error) {
    showInlineNotice(error.message, { error: true });
    if (error.status === 503) loadLLMConfiguration(false).catch(() => {});
    toast("行动未提交");
  } finally {
    setBusy(false);
  }
}

function line(label, value) {
  return `<div class="console-item"><span>${escapeHtml(label)}</span><em>${escapeHtml(value)}</em></div>`;
}

async function refreshDeveloper() {
  if (!state.sessionId) return;
  try {
    const data = await api(`/api/sessions/${state.sessionId}/developer`);
    const experience = data.director.experience;
    $("#experience-metrics").innerHTML = Object.entries(experience).map(([name, value]) => `
      <div class="metric-tile"><span>${escapeHtml(name)}</span><strong>${value}</strong></div>`).join("");
    const latest = data.director.interventions.at(-1);
    $("#director-decision").textContent = latest
      ? `${latest.action}\n\n原因：${latest.reason}\n\n世界依据：${latest.world_justification}\n\n预期：${latest.expected_experience_effect}`
      : `Phase: ${data.director.phase}\n尚未触发干预。`;
    $("#threat-clocks").innerHTML = data.threats.map((threat) => line(threat.name, `${threat.progress}%`)).join("");
    $("#event-queue").innerHTML = data.event_queue.length
      ? data.event_queue.map((event) => line(event.type, event.time.slice(11, 16))).join("")
      : line("queue", "empty");
    $("#npc-locations").innerHTML = data.npc_locations.map((npc) => line(npc.name, `${npc.location}${npc.active ? "" : " · inactive"}`)).join("");
    $("#world-map-summary").innerHTML = (data.world_map?.locations || []).map((location) => `
      <span class="world-node${location.player_here ? " player-here" : ""}">${escapeHtml(location.name)}</span>`).join("");
    $("#event-log").innerHTML = data.event_log.slice(-25).reverse().map((event) => line(`#${event.seq} ${event.type}`, event.time.slice(11, 19))).join("");
  } catch (error) {
    toast(error.message);
  }
}

function setDeveloper(open) {
  state.developerOpen = open;
  elements.devPanel.classList.toggle("open", open);
  elements.devPanel.setAttribute("aria-hidden", String(!open));
  elements.devToggle.setAttribute("aria-pressed", String(open));
  if (open) refreshDeveloper();
}

elements.start.addEventListener("click", createGame);
elements.startName.addEventListener("keydown", (event) => { if (event.key === "Enter") createGame(); });
elements.form.addEventListener("submit", (event) => {
  event.preventDefault();
  const text = elements.input.value.trim();
  if (text) act({ text });
});
elements.narrativeContinue.addEventListener("click", () => {
  if (state.busy || state.narrative.cursor >= state.narrative.beats.length - 1) return;
  state.narrative.cursor += 1;
  paintNarrativeBeat();
});
elements.narrativeSkip.addEventListener("click", () => {
  if (state.busy) return;
  state.narrative.cursor = Math.max(0, state.narrative.beats.length - 1);
  state.narrative.skipped = true;
  stopNarrativePolling();
  paintNarrativeBeat();
  toast("已跳过剩余描写，你可以继续行动");
});
elements.narrativeInterrupt.addEventListener("click", () => {
  if (state.busy || !performanceActive()) return;
  state.narrative.interrupted = true;
  stopNarrativePolling();
  paintNarrativeBeat();
  elements.input.focus();
  toast("演出已暂停，请描述你现在如何介入");
});
elements.input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    elements.form.requestSubmit();
  }
});
elements.newGame.addEventListener("click", () => {
  if (!window.confirm("开始新游戏？当前存档仍会保留，但浏览器将切换到新局。")) return;
  localStorage.removeItem("livingTabletopSession");
  sessionStorage.removeItem(narrativeProgressKey);
  stopNarrativePolling();
  state.sessionId = null;
  elements.modal.classList.remove("hidden");
  setDeveloper(false);
});
elements.devToggle.addEventListener("click", () => setDeveloper(!state.developerOpen));
elements.devClose.addEventListener("click", () => setDeveloper(false));
elements.modelToggle.addEventListener("click", (event) => {
  event.stopPropagation();
  setModelPanel(elements.modelPanel.classList.contains("hidden"));
});
elements.modelPanel.addEventListener("click", (event) => event.stopPropagation());
elements.modelRefresh.addEventListener("click", async () => {
  if (state.modelBusy) return;
  setModelBusy(true);
  elements.modelMessage.textContent = "正在刷新模型列表……";
  elements.modelMessage.classList.remove("error");
  try {
    await loadLLMConfiguration(true);
    elements.modelMessage.textContent = "模型列表已刷新。";
  } catch (error) {
    elements.modelMessage.textContent = error.message;
    elements.modelMessage.classList.add("error");
  } finally {
    setModelBusy(false);
  }
});
elements.modelSave.addEventListener("click", () => saveLLMConfiguration());
elements.modelProbe.addEventListener("click", probeSelectedModel);
document.addEventListener("click", () => setModelPanel(false));

window.addEventListener("offline", () => {
  const message = "当前设备已离线。恢复网络后可重新提交，输入内容会为你保留。";
  if (elements.modal.classList.contains("hidden")) showInlineNotice(message, { error: true });
  else showStartError(message);
});
window.addEventListener("online", () => {
  const message = "网络连接已恢复，可以重新提交刚才的操作。";
  if (elements.modal.classList.contains("hidden")) showInlineNotice(message);
  else {
    const scenario = state.scenarios.find((item) => item.id === state.selectedScenarioId);
    elements.sessionNote.textContent = scenario?.presentation.session_note || message;
    elements.sessionNote.classList.remove("request-error");
  }
  toast("网络连接已恢复");
});

async function boot() {
  try {
    await loadScenarios();
    await loadGame();
    loadLLMConfiguration(true).catch(() => {
      elements.modelDot.dataset.state = "offline";
      elements.modelSummary.textContent = "模型 · 无法检查";
    });
  } catch (error) {
    showStartError(error.message);
    toast("无法连接游戏服务");
  }
}

boot();
