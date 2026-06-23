"""
Web server for the song -> sheet music transcriber.

Serves a single-page frontend and exposes a /api/transcribe endpoint that runs
the Demucs -> basic-pitch -> music21 pipeline on an uploaded file or a bundled
sample, returning MusicXML the browser renders as sheet music.

Run:
    python -m uvicorn webapp.server:app --reload --port 8000
    # then open http://localhost:8000
"""

import os
import uuid
import shutil

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from webapp import pipeline

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(HERE, "static")
SAMPLES_DIR = os.path.join(HERE, "samples")
RUNS_DIR = os.path.join(HERE, "runs")
os.makedirs(RUNS_DIR, exist_ok=True)

# Instruments the UI offers, mapped to Demucs stems. `good=False` => Demucs has
# no dedicated stem, so it falls into "other" and accuracy is lower.
INSTRUMENTS = [
    {"id": "piano", "label": "Piano / Keys", "stem": "piano", "good": True},
    {"id": "guitar", "label": "Guitar", "stem": "guitar", "good": True},
    {"id": "bass", "label": "Bass", "stem": "bass", "good": True},
    {"id": "vocals", "label": "Vocals / Melody", "stem": "vocals", "good": True},
    {"id": "drums", "label": "Drums", "stem": "drums", "good": True},
    {"id": "other", "label": "Strings / Sax / Other", "stem": "other", "good": False},
]
INSTRUMENT_BY_ID = {i["id"]: i for i in INSTRUMENTS}

SAMPLES = [
    {"id": "brahms", "title": "Brahms — Hungarian Dance No. 5", "hint": "String orchestra (polyphonic)"},
    {"id": "trumpet", "title": "Solo Trumpet", "hint": "Single brass line (monophonic)"},
    {"id": "nutcracker", "title": "Tchaikovsky — Sugar Plum Fairy", "hint": "Full orchestra"},
    {"id": "vibeace", "title": "Vibe Ace", "hint": "Jazz combo (multiple instruments)"},
]

app = FastAPI(title="Song to Sheet Music")


@app.on_event("startup")
def _startup():
    # Load models in the background-ish (first call still warms them if skipped)
    try:
        pipeline.warmup()
    except Exception as e:  # pragma: no cover - non-fatal
        print(f"[warmup] model preload skipped: {e}")


@app.get("/api/config")
def config():
    return {"instruments": INSTRUMENTS, "samples": SAMPLES}


@app.get("/samples/{name}")
def get_sample(name: str):
    path = os.path.join(SAMPLES_DIR, f"{name}.ogg")
    if not os.path.isfile(path):
        raise HTTPException(404, "sample not found")
    return FileResponse(path, media_type="audio/ogg")


@app.get("/runs/{run_id}/{filename}")
def get_run_file(run_id: str, filename: str):
    # Guard against path traversal
    if "/" in run_id or "/" in filename or ".." in run_id or ".." in filename:
        raise HTTPException(400, "bad path")
    path = os.path.join(RUNS_DIR, run_id, filename)
    if not os.path.isfile(path):
        raise HTTPException(404, "file not found")
    return FileResponse(path)


@app.post("/api/transcribe")
def transcribe(
    instrument: str = Form(...),
    sample: str = Form(None),
    file: UploadFile = File(None),
):
    inst = INSTRUMENT_BY_ID.get(instrument)
    if inst is None:
        raise HTTPException(400, f"unknown instrument '{instrument}'")

    run_id = uuid.uuid4().hex[:12]
    work_dir = os.path.join(RUNS_DIR, run_id)
    os.makedirs(work_dir, exist_ok=True)

    # Resolve the input audio: uploaded file takes priority, else a bundled sample
    if file is not None and file.filename:
        in_path = os.path.join(work_dir, "input_" + os.path.basename(file.filename))
        with open(in_path, "wb") as out:
            shutil.copyfileobj(file.file, out)
    elif sample:
        src = os.path.join(SAMPLES_DIR, f"{sample}.ogg")
        if not os.path.isfile(src):
            raise HTTPException(404, f"sample '{sample}' not found")
        in_path = os.path.join(work_dir, f"input_{sample}.ogg")
        shutil.copy(src, in_path)
    else:
        raise HTTPException(400, "provide either an uploaded file or a sample id")

    try:
        result = pipeline.run(in_path, inst["stem"], work_dir)
    except Exception as e:
        raise HTTPException(500, f"transcription failed: {e}")

    return JSONResponse({
        "run_id": run_id,
        "instrument": inst["label"],
        "method": result["method"],
        "low_accuracy": not inst["good"],
        "n_notes": result["n_notes"],
        "duration": round(result["duration"], 1),
        "musicxml": result["musicxml"],
        "midi_url": f"/runs/{run_id}/transcription.mid",
        "stem_url": f"/runs/{run_id}/stem_{inst['stem']}.wav",
        "musicxml_url": f"/runs/{run_id}/sheet_music.musicxml",
    })


# Serve the frontend at the root
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
