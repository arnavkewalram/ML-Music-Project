"use strict";

const state = { sample: null, file: null, instrument: "piano" };

const el = (id) => document.getElementById(id);

// ---------- Load config (samples + instruments) ----------
async function init() {
  const cfg = await fetch("/api/config").then((r) => r.json());
  renderSamples(cfg.samples);
  renderInstruments(cfg.instruments);
  el("go").addEventListener("click", run);
  el("file").addEventListener("change", onFile);
}

function renderSamples(samples) {
  const wrap = el("samples");
  wrap.innerHTML = "";
  samples.forEach((s) => {
    const card = document.createElement("button");
    card.className = "sample";
    card.type = "button";
    card.setAttribute("role", "radio");
    card.setAttribute("aria-checked", "false");
    card.innerHTML = `
      <span class="s-title">${s.title}</span>
      <span class="s-hint">${s.hint}</span>
      <audio controls preload="none" src="/samples/${s.id}"></audio>`;
    // Clicking the card selects it; clicking the audio shouldn't toggle selection
    card.addEventListener("click", (e) => {
      if (e.target.tagName === "AUDIO") return;
      selectSample(s.id);
    });
    card.querySelector("audio").addEventListener("click", (e) => e.stopPropagation());
    wrap.appendChild(card);
  });
}

function selectSample(id) {
  state.sample = id;
  state.file = null;
  el("file").value = "";
  el("upload-face").classList.remove("has-file");
  el("upload-face").textContent = "Upload your own audio…";
  [...el("samples").children].forEach((c) =>
    c.setAttribute("aria-checked", String(c.querySelector(".s-title") &&
      c.querySelector(`audio[src="/samples/${id}"]`) !== null))
  );
  updateGo();
}

function onFile(e) {
  const f = e.target.files[0];
  if (!f) return;
  state.file = f;
  state.sample = null;
  [...el("samples").children].forEach((c) => c.setAttribute("aria-checked", "false"));
  const face = el("upload-face");
  face.textContent = f.name;
  face.classList.add("has-file");
  updateGo();
}

function renderInstruments(instruments) {
  const wrap = el("instruments");
  wrap.innerHTML = "";
  instruments.forEach((i) => {
    const chip = document.createElement("button");
    chip.className = "chip";
    chip.type = "button";
    chip.setAttribute("role", "radio");
    chip.dataset.id = i.id;
    chip.dataset.good = String(i.good);
    chip.setAttribute("aria-checked", String(i.id === state.instrument));
    chip.innerHTML = i.label + (i.good ? "" : ' <span class="warn">▲</span>');
    chip.addEventListener("click", () => selectInstrument(i.id, i.good));
    wrap.appendChild(chip);
  });
}

function selectInstrument(id, good) {
  state.instrument = id;
  [...el("instruments").children].forEach((c) =>
    c.setAttribute("aria-checked", String(c.dataset.id === id))
  );
  const note = el("inst-note");
  if (good === false) {
    note.hidden = false;
    note.textContent = "No dedicated separation model for this — it comes from the leftover mix, so accuracy drops.";
  } else {
    note.hidden = true;
  }
}

function updateGo() {
  el("go").disabled = !(state.sample || state.file);
}

// ---------- Run the pipeline ----------
let progressTimer = null;

function startProgress() {
  el("progress").hidden = false;
  el("result").hidden = true;
  el("error").hidden = true;
  el("go").disabled = true;

  const stages = ["separate", "transcribe", "engrave"];
  const stageEls = {};
  document.querySelectorAll(".stage").forEach((s) => (stageEls[s.dataset.stage] = s));
  stages.forEach((s) => stageEls[s].classList.remove("active", "done"));

  // We can't get true per-stage progress from one request, so animate a
  // believable sweep: separation is the slow part, then transcribe, then engrave.
  let pct = 0;
  let idx = 0;
  stageEls[stages[0]].classList.add("active");
  const fill = el("bar-fill");
  progressTimer = setInterval(() => {
    pct = Math.min(pct + Math.random() * 2.2, 95);
    fill.style.width = pct + "%";
    const target = idx === 0 ? 65 : idx === 1 ? 88 : 95;
    if (pct >= target && idx < stages.length - 1) {
      stageEls[stages[idx]].classList.remove("active");
      stageEls[stages[idx]].classList.add("done");
      idx++;
      stageEls[stages[idx]].classList.add("active");
    }
  }, 350);
}

function finishProgress(ok) {
  clearInterval(progressTimer);
  el("bar-fill").style.width = "100%";
  document.querySelectorAll(".stage").forEach((s) => {
    s.classList.remove("active");
    if (ok) s.classList.add("done");
  });
  setTimeout(() => (el("progress").hidden = true), ok ? 400 : 0);
}

async function run() {
  startProgress();
  const fd = new FormData();
  fd.append("instrument", state.instrument);
  if (state.file) fd.append("file", state.file);
  else fd.append("sample", state.sample);

  try {
    const res = await fetch("/api/transcribe", { method: "POST", body: fd });
    if (!res.ok) {
      const detail = await res.json().catch(() => ({}));
      throw new Error(detail.detail || `Server error (${res.status})`);
    }
    const data = await res.json();
    finishProgress(true);
    await showResult(data);
  } catch (err) {
    finishProgress(false);
    showError(err.message);
  } finally {
    updateGo();
  }
}

function showError(msg) {
  const e = el("error");
  e.hidden = false;
  e.textContent = "Couldn't transcribe that: " + msg;
}

async function showResult(data) {
  el("result").hidden = false;

  el("result-meta").innerHTML =
    `<strong>${data.instrument}</strong> · ${data.n_notes} notes · ${data.duration}s` +
    `<span class="sub">${data.method} · rhythm lightly quantized</span>`;

  el("lowacc").hidden = !data.low_accuracy;

  // Almost no notes => the chosen instrument probably isn't in this track.
  const empty = el("empty-notice");
  if (data.n_notes < 5) {
    empty.hidden = false;
    empty.innerHTML =
      `We found almost no <strong>${data.instrument}</strong> in this track — it likely ` +
      `doesn't contain that part. Try a different instrument, or a song that has it.`;
  } else {
    empty.hidden = true;
  }

  el("downloads").innerHTML = `
    <a href="${data.musicxml_url}" download>MusicXML</a>
    <a href="${data.midi_url}" download>MIDI</a>
    <a href="${data.stem_url}" download>Isolated audio</a>`;

  const audio = el("stem-audio");
  audio.src = data.stem_url;

  // Render the score with OpenSheetMusicDisplay
  const scoreDiv = el("score");
  scoreDiv.innerHTML = "";
  try {
    const osmd = new opensheetmusicdisplay.OpenSheetMusicDisplay(scoreDiv, {
      autoResize: true,
      drawingParameters: "compact",
      drawTitle: false,
    });
    await osmd.load(data.musicxml);
    osmd.render();
  } catch (e) {
    scoreDiv.innerHTML =
      `<p style="color:#8b857a">Couldn't render the score in-browser, but your downloads above are ready. (${e.message})</p>`;
  }

  el("result").scrollIntoView({ behavior: "smooth", block: "start" });
}

init();
