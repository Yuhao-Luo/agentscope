const state = {
  phaseIndex: 0,
  phases: ["第1夜", "第1天", "第2夜", "第2天"],
  timeline: ["开局：身份分发完成", "第1夜：狼人行动", "第1天：公开讨论与投票"],
  wolfKill: [{ wolf: "player3", target: "player7", note: "优先刀高发言位" }],
  nightDiscussion: [{ speaker: "player3", text: "先刀player7，明天甩锅给player2" }],
  seerChecks: [{ seer: "player2", target: "player5", result: "好人" }],
  witchActions: [{ witch: "player6", action: "未使用解药", poison: "未使用毒药" }],
  dayDiscussion: [
    { speaker: "player1", text: "我觉得player3发言很可疑" },
    { speaker: "player4", text: "先听听player2有没有验人信息" }
  ],
  votes: [
    { from: "player1", to: "player3", reason: "发言冲突" },
    { from: "player2", to: "player3", reason: "逻辑断层" },
    { from: "player4", to: "player7", reason: "跟票" }
  ]
};

function renderList(containerId, items, mapper) {
  const container = document.getElementById(containerId);
  container.innerHTML = "";
  items.forEach((it) => {
    const div = document.createElement("div");
    div.className = `item ${mapper.className ? mapper.className(it) : ""}`;
    div.innerHTML = mapper.text(it);
    container.appendChild(div);
  });
}

function render() {
  document.getElementById("phase-pill").textContent = `阶段：${state.phases[state.phaseIndex]}`;

  const timeline = document.getElementById("timeline");
  timeline.innerHTML = "";
  state.timeline.forEach((t) => {
    const li = document.createElement("li");
    li.textContent = t;
    timeline.appendChild(li);
  });

  renderList("wolf-kill", state.wolfKill, {
    className: () => "bad",
    text: (x) => `<strong>${x.wolf}</strong> 刀了 <strong>${x.target}</strong>（${x.note}）`
  });

  renderList("night-discussion", state.nightDiscussion, {
    text: (x) => `<strong>${x.speaker}</strong>：${x.text}`
  });

  renderList("seer-checks", state.seerChecks, {
    className: (x) => (x.result === "好人" ? "good" : "bad"),
    text: (x) => `<strong>${x.seer}</strong> 查验 <strong>${x.target}</strong>：${x.result}`
  });

  renderList("witch-actions", state.witchActions, {
    text: (x) => `<strong>${x.witch}</strong>：${x.action} / ${x.poison}`
  });

  renderList("day-discussion", state.dayDiscussion, {
    text: (x) => `<strong>${x.speaker}</strong>：${x.text}`
  });

  const voteTable = document.getElementById("vote-table");
  voteTable.innerHTML = "";
  state.votes.forEach((v) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${v.from}</td><td>${v.to}</td><td>${v.reason}</td>`;
    voteTable.appendChild(tr);
  });
}

document.getElementById("add-day-talk").addEventListener("click", () => {
  const speaker = document.getElementById("speaker").value.trim();
  const speech = document.getElementById("speech").value.trim();
  if (!speaker || !speech) return;
  state.dayDiscussion.push({ speaker, text: speech });
  render();
});

document.getElementById("next-phase").addEventListener("click", () => {
  state.phaseIndex = (state.phaseIndex + 1) % state.phases.length;
  state.timeline.push(`阶段推进：${state.phases[state.phaseIndex]}`);
  render();
});

render();
