const PLAYER_COUNT = 8;

const state = {
  round: 1,
  phase: "第1夜",
  speakingIdx: 0,
  socket: null,
  players: Array.from({ length: PLAYER_COUNT }, (_, i) => ({
    name: `player${i}`,
    role: "未知",
    alive: true
  })),
  logs: []
};

function addLog(title, text) {
  state.logs.unshift({ title, text, ts: new Date().toLocaleTimeString() });
  state.logs = state.logs.slice(0, 100);
  renderLogs();
}

function renderSeats() {
  const seats = document.getElementById("seats");
  seats.innerHTML = "";

  const cx = seats.clientWidth / 2;
  const cy = seats.clientHeight / 2;
  const radius = Math.min(cx, cy) - 80;

  state.players.forEach((p, i) => {
    const angle = ((Math.PI * 2) / PLAYER_COUNT) * i - Math.PI / 2;
    const x = cx + radius * Math.cos(angle) - 45;
    const y = cy + radius * Math.sin(angle) - 32;

    const div = document.createElement("div");
    div.className = `seat ${i === state.speakingIdx ? "speaking" : ""} ${p.alive ? "" : "dead"}`;
    div.style.left = `${x}px`;
    div.style.top = `${y}px`;
    div.innerHTML = `<div class="name">${p.name}</div><div class="role">${p.role}</div>`;
    seats.appendChild(div);
  });
}

function renderLogs() {
  const stream = document.getElementById("event-stream");
  stream.innerHTML = "";
  state.logs.forEach((l) => {
    const div = document.createElement("div");
    div.className = "log-item";
    div.innerHTML = `<strong>[${l.ts}] ${l.title}</strong><div>${l.text}</div>`;
    stream.appendChild(div);
  });
}

function renderMeta() {
  document.getElementById("phase-text").textContent = `${state.phase} / 第${state.round}轮`;
}

function render() {
  renderMeta();
  renderSeats();
}

function setConnStatus(online, text) {
  const el = document.getElementById("conn-status");
  el.textContent = text;
  el.className = `status ${online ? "on" : "off"}`;
}

function onBackendEvent(payload) {
  const type = payload.type;

  if (type === "phase") {
    if (payload.round) state.round = payload.round;
    if (payload.phase) state.phase = payload.phase;
    addLog("阶段切换", `${state.phase}`);
  }

  if (type === "speech") {
    const idx = state.players.findIndex((x) => x.name === payload.speaker);
    if (idx >= 0) state.speakingIdx = idx;
    addLog(`发言 ${payload.speaker || "未知"}`, payload.text || "");
  }

  if (type === "wolf_kill") {
    document.getElementById("wolf-kill").textContent = `${payload.target || "-"}`;
    addLog("夜晚刀人", `目标：${payload.target || "-"}`);
  }

  if (type === "seer_check") {
    document.getElementById("seer-check").textContent = `${payload.target || "-"} => ${payload.result || "-"}`;
    addLog("预言家验人", `${payload.target || "-"}：${payload.result || "-"}`);
  }

  if (type === "witch_action") {
    document.getElementById("witch-action").textContent = payload.action || "-";
    addLog("女巫行动", payload.action || "-");
  }

  if (type === "vote") {
    document.getElementById("vote-info").textContent = `${payload.from || "-"} -> ${payload.to || "-"}`;
    addLog("投票", `${payload.from || "-"} 投给 ${payload.to || "-"}`);
  }

  if (type === "player_status") {
    const player = state.players.find((x) => x.name === payload.player);
    if (player) {
      if (typeof payload.alive === "boolean") player.alive = payload.alive;
      if (payload.role) player.role = payload.role;
    }
    addLog("玩家状态更新", `${payload.player}: ${payload.alive === false ? "出局" : "存活"}`);
  }

  render();
}

function connectWS() {
  const wsUrl = document.getElementById("ws-url").value.trim();
  if (!wsUrl) return;

  if (state.socket) {
    state.socket.close();
  }

  const socket = new WebSocket(wsUrl);
  state.socket = socket;
  setConnStatus(false, "连接中...");

  socket.onopen = () => {
    setConnStatus(true, "已连接");
    addLog("系统", `已连接 ${wsUrl}`);
  };

  socket.onmessage = (event) => {
    try {
      const payload = JSON.parse(event.data);
      onBackendEvent(payload);
    } catch {
      addLog("原始消息", event.data);
    }
  };

  socket.onclose = () => {
    setConnStatus(false, "连接断开");
    addLog("系统", "连接断开");
  };

  socket.onerror = () => {
    setConnStatus(false, "连接异常");
    addLog("系统", "WebSocket 连接异常");
  };
}

window.addEventListener("resize", renderSeats);

document.getElementById("connect-btn").addEventListener("click", connectWS);
document.getElementById("next-speaker").addEventListener("click", () => {
  state.speakingIdx = (state.speakingIdx + 1) % PLAYER_COUNT;
  renderSeats();
});
document.getElementById("prev-speaker").addEventListener("click", () => {
  state.speakingIdx = (state.speakingIdx - 1 + PLAYER_COUNT) % PLAYER_COUNT;
  renderSeats();
});

render();
addLog("系统", "页面已加载，可连接后端 WebSocket。\n如需联调，请让后端按协议持续推送 JSON 事件。");
