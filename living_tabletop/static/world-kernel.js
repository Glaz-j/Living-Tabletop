const state = {
  sessionId: localStorage.getItem("vwk-session-id"),
  projection: null,
  busy: false,
};

const $ = (id) => document.getElementById(id);
const escapeHtml = (value) => String(value ?? "")
  .replaceAll("&", "&amp;").replaceAll("<", "&lt;")
  .replaceAll(">", "&gt;").replaceAll('"', "&quot;");

const eventLabels = {
  MovementAttempted: "尝试移动",
  EntityMoved: "位置改变",
  MovementAttemptFailed: "移动未能完成",
  TimeAdvanced: "世界时间推进",
  InspectionAttempted: "开始调查",
  EntityInspected: "完成调查",
  InspectionAttemptFailed: "调查未能完成",
  KnowledgeLearned: "获得新知识",
  ConnectionDiscovered: "发现隐藏通路",
  InteractionAttempted: "开始互动",
  InteractionCompleted: "互动完成",
  InteractionAttemptFailed: "互动未能完成",
  EntityPlaced: "物品位置改变",
  ConnectionStateChanged: "通路状态改变",
  WaitAttempted: "开始等待",
  WaitCompleted: "等待结束",
  HouseStirred: "宅邸异动",
  ActionInterrupted: "行动被世界事件打断",
};

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    const detail = data.detail?.message || data.detail || `HTTP ${response.status}`;
    const error = new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    error.status = response.status;
    throw error;
  }
  return response.json();
}

async function createWorld() {
  const data = await api("/api/world-kernel/sessions", {
    method: "POST",
    body: JSON.stringify({ viewer_id: "player" }),
  });
  state.sessionId = data.session_id;
  state.projection = data.projection;
  localStorage.setItem("vwk-session-id", state.sessionId);
  render();
}

async function loadWorld() {
  if (!state.sessionId) return createWorld();
  try {
    state.projection = await api(
      `/api/world-kernel/sessions/${state.sessionId}/projection?viewer_id=player`
    );
    render();
  } catch (error) {
    if (error.status === 404) return createWorld();
    showStatus(`无法加载世界：${error.message}`, true);
  }
}

function showStatus(message, error = false) {
  const element = $("request-status");
  element.textContent = message;
  element.classList.toggle("error", error);
}

async function dispatch(kind, payload) {
  if (state.busy || !state.projection) return;
  state.busy = true;
  renderAffordances();
  showStatus("Command 已提交，等待世界事务完成……");
  try {
    const data = await api(`/api/world-kernel/sessions/${state.sessionId}/commands`, {
      method: "POST",
      body: JSON.stringify({
        kind,
        payload,
        expected_state_version: state.projection.world_version,
        issuer_id: "player",
        actor_id: "player",
        command_id: `cmd_${crypto.randomUUID()}`,
        idempotency_key: `ui_${crypto.randomUUID()}`,
      }),
    });
    state.projection = data.projection;
    const receipt = data.receipt;
    const label = receipt.outcome === "succeeded" ? "行动已写入世界" :
      receipt.outcome === "interrupted" ? "行动被世界事件中断，你仍可继续尝试" :
      receipt.outcome === "failed" ? `这是一次有效尝试，但结果未成功：${receipt.reason}` :
      "重复命令未再次执行";
    showStatus(label, false);
    render();
  } catch (error) {
    if (error.status === 409) await loadWorld();
    showStatus(`行动未提交：${error.message}`, true);
  } finally {
    state.busy = false;
    renderAffordances();
  }
}

function locationById(id) {
  return state.projection.map.locations.find((item) => item.id === id);
}

function renderMap() {
  const projection = state.projection;
  const locations = Object.fromEntries(projection.map.locations.map((item) => [item.id, item]));
  $("map-edges").innerHTML = projection.map.connections.map((connection) => {
    const from = locations[connection.from_location];
    const to = locations[connection.to_location];
    if (!from || !to) return "";
    const locked = connection.state.locked || !connection.state.open;
    return `<line class="map-edge ${locked ? "locked" : ""}" x1="${from.x}" y1="${from.y}" x2="${to.x}" y2="${to.y}"><title>${escapeHtml(connection.name)}</title></line>`;
  }).join("");

  $("map-nodes").innerHTML = projection.map.locations.map((location) => {
    const classes = ["map-node", location.current ? "current" : "", location.observed ? "" : "unobserved"].join(" ");
    return `<g class="${classes}" data-location-id="${escapeHtml(location.id)}" transform="translate(${location.x} ${location.y})">
      <circle r="4.5"></circle>
      <text y="7.8">${escapeHtml(location.name)}</text>
      <text class="node-meta" y="10.4">${location.current ? "YOU ARE HERE" : location.observed ? "OBSERVED" : "UNOBSERVED"}</text>
      <title>${escapeHtml(location.description)}</title>
    </g>`;
  }).join("");

  document.querySelectorAll(".map-node").forEach((node) => {
    node.addEventListener("click", () => {
      const destination = node.dataset.locationId;
      const affordance = projection.affordances.find(
        (item) => item.kind === "move" && item.payload.destination_id === destination
      );
      if (affordance) dispatch(affordance.kind, affordance.payload);
    });
  });
}

function renderAffordances() {
  if (!state.projection) return;
  $("affordances").innerHTML = state.projection.affordances.map((item, index) =>
    `<button type="button" data-affordance="${index}" ${state.busy ? "disabled" : ""}>${escapeHtml(item.label)}</button>`
  ).join("");
  document.querySelectorAll("[data-affordance]").forEach((button) => {
    button.addEventListener("click", () => {
      const item = state.projection.affordances[Number(button.dataset.affordance)];
      dispatch(item.kind, item.payload);
    });
  });
}

function renderEvents() {
  const events = [...state.projection.recent_events].reverse();
  $("event-log").innerHTML = events.length ? events.map((event) => {
    const time = new Date(event.world_time).toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" });
    const detail = event.payload.description || event.payload.reason ||
      (event.payload.destination ? `抵达 ${locationById(event.payload.destination)?.name || event.payload.destination}` : "");
    return `<li><time>#${event.sequence} · ${time}</time><strong>${escapeHtml(eventLabels[event.type] || event.type)}</strong>${detail ? `<p>${escapeHtml(detail)}</p>` : ""}</li>`;
  }).join("") : `<li class="empty">世界刚刚建立，尚无已发生事件。</li>`;
}

function render() {
  if (!state.projection) return;
  const projection = state.projection;
  $("world-title").textContent = projection.world.title;
  $("world-time").textContent = new Date(projection.world.time).toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" });
  $("world-version").textContent = projection.world_version;
  $("observer-name").textContent = projection.observer.name;
  $("location-name").textContent = locationById(projection.observer.location_id)?.name || "未知位置";
  $("visible-entities").innerHTML = projection.visible_entities.length ? projection.visible_entities.map((entity) =>
    `<article class="entity-card"><strong>${escapeHtml(entity.name)}</strong><p>${escapeHtml(entity.description)}</p></article>`
  ).join("") : `<span class="empty">此处没有其他可观察实体。</span>`;
  $("inventory").innerHTML = projection.observer.inventory.length ? projection.observer.inventory.map((item) =>
    `<span>${escapeHtml(item.name)}</span>`
  ).join("") : `<span class="empty">没有随身物品。</span>`;
  $("knowledge").innerHTML = projection.knowledge.length ? projection.knowledge.map((item) =>
    `<div class="knowledge-item">${escapeHtml(item.subject)} · ${escapeHtml(item.predicate)} · ${escapeHtml(item.claimed_value)}<small>${escapeHtml(item.stance)} / ${Math.round(item.confidence * 100)}% / ${escapeHtml(item.source)}</small></div>`
  ).join("") : `<span class="empty">尚未形成可用知识。</span>`;
  renderMap();
  renderAffordances();
  renderEvents();
}

$("new-world").addEventListener("click", async () => {
  localStorage.removeItem("vwk-session-id");
  state.sessionId = null;
  await createWorld();
  showStatus("已建立一个新的、独立的世界分支。");
});

$("replay-check").addEventListener("click", async () => {
  if (!state.sessionId) return;
  try {
    const report = await api(`/api/world-kernel/sessions/${state.sessionId}/replay`);
    showStatus(report.verified
      ? `重放验证通过：${report.event_count} 个事件，从零重建结果一致。`
      : "重放验证失败：Snapshot 与事件重建结果不一致。", !report.verified);
  } catch (error) {
    showStatus(`重放验证失败：${error.message}`, true);
  }
});

$("dev-toggle").addEventListener("click", async () => {
  if (!state.sessionId) return;
  try {
    const projection = await api(`/api/world-kernel/sessions/${state.sessionId}/projection?view=dev`);
    $("dev-json").textContent = JSON.stringify(projection, null, 2);
    $("dev-dialog").showModal();
  } catch (error) {
    showStatus(`开发者投影加载失败：${error.message}`, true);
  }
});

$("dev-close").addEventListener("click", () => $("dev-dialog").close());
window.addEventListener("offline", () => showStatus("网络已断开；Command 没有提交，世界状态不会改变。", true));
loadWorld();
