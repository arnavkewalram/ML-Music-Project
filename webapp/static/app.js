"use strict";

const state = { sample: null, file: null, instrument: "piano" };

const el = (id) => document.getElementById(id);

// ---------- Load config (samples + instruments) ----------
async function init() {
  let cfg;
  try {
    const res = await fetch("/api/config");
    if (!res.ok) throw new Error(`server returned ${res.status}`);
    cfg = await res.json();
  } catch (err) {
    // Without config there are no songs and no instruments to pick, so the page
    // is unusable. Say so instead of leaving two silently empty panels.
    showError(
      `couldn't reach the server (${err.message}). Check that it's running, then reload.`
    );
    return;
  }
  renderSamples(cfg.samples);
  renderInstruments(cfg.instruments);
  el("go").addEventListener("click", run);
  el("file").addEventListener("change", onFile);
}

// ---------- Radio group keyboard support ----------
// role="radiogroup" carries a keyboard contract: one tab stop for the group,
// arrows to move between options. Without it these are unreachable by keyboard
// in the way a screen reader announces them.
function setupRadioGroup(container, onSelect) {
  container.addEventListener("keydown", (e) => {
    const keys = ["ArrowRight", "ArrowDown", "ArrowLeft", "ArrowUp", "Home", "End"];
    if (!keys.includes(e.key)) return;
    const radios = [...container.querySelectorAll('[role="radio"]')];
    if (!radios.length) return;
    const current = radios.indexOf(document.activeElement);
    const forward = e.key === "ArrowRight" || e.key === "ArrowDown";
    let next;
    if (e.key === "Home") next = 0;
    else if (e.key === "End") next = radios.length - 1;
    else if (current === -1) next = 0;
    else next = (current + (forward ? 1 : -1) + radios.length) % radios.length;

    e.preventDefault();
    radios[next].focus();
    onSelect(radios[next].dataset.id);
  });
}

function updateRovingTabindex(container, selectedId) {
  const radios = [...container.querySelectorAll('[role="radio"]')];
  radios.forEach((r) => {
    const checked = r.dataset.id === selectedId;
    r.setAttribute("aria-checked", String(checked));
    // One tab stop per group; if nothing is chosen yet the first option takes it.
    r.tabIndex = checked || (!selectedId && r === radios[0]) ? 0 : -1;
  });
}

// ---------- Samples ----------
function renderSamples(samples) {
  const wrap = el("samples");
  wrap.textContent = "";
  samples.forEach((s) => {
    // The <audio> player is a sibling of the radio, not a child of it: interactive
    // controls cannot be nested inside a button, and doing so breaks both
    // keyboard traversal and the click target.
    const card = document.createElement("div");
    card.className = "sample";
    card.setAttribute("role", "presentation");

    const choice = document.createElement("button");
    choice.className = "sample-select";
    choice.type = "button";
    choice.setAttribute("role", "radio");
    choice.setAttribute("aria-checked", "false");
    choice.dataset.id = s.id;

    const title = document.createElement("span");
    title.className = "s-title";
    title.textContent = s.title;
    const hint = document.createElement("span");
    hint.className = "s-hint";
    hint.textContent = s.hint;
    choice.append(title, hint);
    choice.addEventListener("click", () => selectSample(s.id));

    const audio = document.createElement("audio");
    audio.controls = true;
    audio.preload = "none";
    audio.src = `/samples/${s.id}`;
    audio.setAttribute("aria-label", `Preview ${s.title}`);

    card.append(choice, audio);
    wrap.appendChild(card);
  });
  setupRadioGroup(wrap, selectSample);
  updateRovingTabindex(wrap, state.sample);
}

function selectSample(id) {
  state.sample = id;
  state.file = null;
  el("file").value = "";
  el("upload-face").classList.remove("has-file");
  el("upload-face").textContent = "Upload your own audio…";
  markSampleSelection(id);
  updateGo();
}

function markSampleSelection(id) {
  const wrap = el("samples");
  updateRovingTabindex(wrap, id);
  [...wrap.querySelectorAll(".sample")].forEach((card) => {
    const choice = card.querySelector('[role="radio"]');
    card.classList.toggle("is-selected", Boolean(choice) && choice.dataset.id === id);
  });
}

function onFile(e) {
  const f = e.target.files[0];
  if (!f) return;
  state.file = f;
  state.sample = null;
  markSampleSelection(null);
  const face = el("upload-face");
  face.textContent = f.name;
  face.classList.add("has-file");
  updateGo();
}

// ---------- Instruments ----------
function renderInstruments(instruments) {
  const wrap = el("instruments");
  wrap.textContent = "";
  instruments.forEach((i) => {
    const chip = document.createElement("button");
    chip.className = "chip";
    chip.type = "button";
    chip.setAttribute("role", "radio");
    chip.dataset.id = i.id;
    chip.dataset.good = String(i.good);
    chip.textContent = i.label;
    if (!i.good) {
      const warn = document.createElement("span");
      warn.className = "warn";
      warn.textContent = "▲";
      warn.setAttribute("aria-hidden", "true");
      chip.appendChild(warn);
    }
    chip.addEventListener("click", () => selectInstrument(i.id));
    wrap.appendChild(chip);
  });
  setupRadioGroup(wrap, selectInstrument);
  selectInstrument(state.instrument);
}

function selectInstrument(id) {
  const chip = el("instruments").querySelector(`[data-id="${CSS.escape(id)}"]`);
  if (!chip) return;
  state.instrument = id;
  updateRovingTabindex(el("instruments"), id);

  const note = el("inst-note");
  const good = chip.dataset.good === "true";
  note.hidden = good;
  if (!good) {
    note.textContent =
      "No dedicated separation model for this — it comes from the leftover mix, so accuracy drops.";
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

  const meta = el("result-meta");
  meta.textContent = "";
  const name = document.createElement("strong");
  name.textContent = data.instrument;
  const counts = document.createTextNode(
    ` · ${data.n_notes} notes · ${data.duration}s${data.tempo ? ` · ${data.tempo} BPM` : ""}`
  );
  const sub = document.createElement("span");
  sub.className = "sub";
  sub.textContent = `${data.method} · rhythm lightly quantized`;
  meta.append(name, counts, sub);

  el("lowacc").hidden = !data.low_accuracy;

  // Long uploads are only analysed up to the server's cap; say so rather than
  // letting the score just stop partway through the song.
  const trunc = el("truncated-notice");
  trunc.hidden = !data.truncated;
  if (data.truncated) {
    trunc.textContent =
      `This track is ${Math.round(data.duration)}s long; only the first ` +
      `${Math.round(data.analyzed_duration)}s were transcribed.`;
  }

  // The chosen instrument probably isn't in this track: either almost no notes,
  // or the separated stem was near-silent (so any notes are noise/bleed).
  const empty = el("empty-notice");
  const missing = data.instrument_present === false || data.n_notes < 5;
  empty.hidden = !missing;
  if (missing) {
    const noisy = data.instrument_present === false && data.n_notes >= 5;
    empty.textContent = "";
    empty.append(
      document.createTextNode("We found almost no "),
      Object.assign(document.createElement("strong"), { textContent: data.instrument }),
      document.createTextNode(
        " in this track — it likely doesn't contain that part. " +
          (noisy ? "The notes below are probably transcribed from bleed/noise. " : "") +
          "Try a different instrument, or a song that has it."
      )
    );
  }

  const downloads = el("downloads");
  downloads.textContent = "";
  [
    ["MusicXML", data.musicxml_url],
    ["MIDI", data.midi_url],
    ["Isolated audio", data.stem_url],
  ].forEach(([label, href]) => {
    const a = document.createElement("a");
    a.href = href;
    a.download = "";
    a.textContent = label;
    downloads.appendChild(a);
  });

  el("stem-audio").src = data.stem_url;

  // Render the score with OpenSheetMusicDisplay
  const scoreDiv = el("score");
  scoreDiv.textContent = "";
  try {
    if (typeof opensheetmusicdisplay === "undefined") {
      throw new Error("the score renderer failed to load");
    }
    const osmd = new opensheetmusicdisplay.OpenSheetMusicDisplay(scoreDiv, {
      autoResize: true,
      drawingParameters: "compact",
      drawTitle: false,
      // Only ever one instrument here, and it's already named above the score.
      // Left on, the label column eats about 40% of the first system's width,
      // and a staff with no name of its own renders its internal part id
      // ("Instr. P537481a9f11e...") instead. The names are still written into
      // the MusicXML download, where MuseScore and Finale use them properly.
      drawPartNames: false,
    });
    await osmd.load(data.musicxml);
    osmd.render();
  } catch (e) {
    const p = document.createElement("p");
    p.style.color = "#8b857a";
    p.textContent =
      `Couldn't render the score in-browser, but your downloads above are ready. (${e.message})`;
    scoreDiv.appendChild(p);
  }

  el("result").scrollIntoView({ behavior: "smooth", block: "start" });
}

init();
