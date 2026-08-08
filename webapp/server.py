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
import re
import uuid
import shutil
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
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

# Below this stem/mix RMS ratio, the instrument is treated as "not really present"
# (Demucs returns a near-silent stem). Calibrated against absent-instrument runs.
STEM_PRESENT_RATIO = 0.06

# Uploads are spooled to disk before the pipeline touches them, so an unbounded
# body is a disk-exhaustion vector. 100MB is a generous lossless album track.
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
UPLOAD_CHUNK_BYTES = 1024 * 1024

# Every run leaves a stem WAV + MIDI + MusicXML behind so the browser can
# download them. Keep the recent ones and drop the rest; without this the
# directory grows without bound.
MAX_RUNS_KEPT = 20

# Anything outside this is stripped from an uploaded filename before it is used
# as a path component.
UNSAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9._-]+")

# `try` lists the instruments actually present in each track, so users don't ask
# for (say) piano on a track that has none and get a blank score.
SAMPLES = [
    {"id": "chopin", "title": "Chopin — Grande Valse Brillante", "hint": "Solo piano — try Piano", "try": ["piano"]},
    {"id": "vibeace", "title": "Vibe Ace", "hint": "Jazz combo — try Piano, Bass, or Drums", "try": ["piano", "bass", "drums"]},
    {"id": "brahms", "title": "Brahms — Hungarian Dance No. 5", "hint": "String orchestra — try Strings / Other", "try": ["other"]},
    # Demucs' 6-stem model has no brass stem, so a trumpet lands in "other".
    # Asking for Vocals here returns a near-silent stem (measured ratio 0.0007)
    # and a handful of notes hallucinated from bleed.
    {"id": "trumpet", "title": "Solo Trumpet", "hint": "Single brass line — try Strings / Sax / Other", "try": ["other"]},
    {"id": "nutcracker", "title": "Tchaikovsky — Sugar Plum Fairy", "hint": "Full orchestra — try Other or Piano", "try": ["other", "piano"]},
]
# Allowlist. `sample` arrives as a form field and is used to build a filesystem
# path, so it must never be trusted as a path fragment.
SAMPLE_IDS = frozenset(s["id"] for s in SAMPLES)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        pipeline.warmup()
    except Exception as e:  # pragma: no cover - non-fatal
        print(f"[warmup] model preload skipped: {e}")
    yield


app = FastAPI(title="Song to Sheet Music", lifespan=lifespan)


@app.get("/api/config")
def config():
    return {"instruments": INSTRUMENTS, "samples": SAMPLES}


def _sample_path(name: str) -> str:
    """Resolve a sample id to its audio file, or 404.

    Only ids from the published allowlist are accepted, so no caller-supplied
    text ever reaches the filesystem.
    """
    if name not in SAMPLE_IDS:
        raise HTTPException(404, f"sample '{name}' not found")
    path = os.path.join(SAMPLES_DIR, f"{name}.ogg")
    if not os.path.isfile(path):
        raise HTTPException(404, f"sample '{name}' not found")
    return path


def _contained(root: str, *parts: str) -> str:
    """Join under `root` and refuse anything that escapes it.

    Belt and braces alongside the checks below: resolving both sides and
    comparing prefixes catches symlinks and encodings a substring test misses.
    """
    root_real = os.path.realpath(root)
    target = os.path.realpath(os.path.join(root_real, *parts))
    if target != root_real and not target.startswith(root_real + os.sep):
        raise HTTPException(400, "bad path")
    return target


@app.get("/samples/{name}")
def get_sample(name: str):
    return FileResponse(_sample_path(name), media_type="audio/ogg")


@app.get("/runs/{run_id}/{filename}")
def get_run_file(run_id: str, filename: str):
    if any(sep in part for part in (run_id, filename) for sep in ("/", "\\")) \
            or ".." in run_id or ".." in filename:
        raise HTTPException(400, "bad path")
    path = _contained(RUNS_DIR, run_id, filename)
    if not os.path.isfile(path):
        raise HTTPException(404, "file not found")
    return FileResponse(path)


def _safe_upload_name(filename: str) -> str:
    """Reduce an uploaded filename to a harmless single path component."""
    stripped = UNSAFE_FILENAME_CHARS.sub("_", os.path.basename(filename or ""))
    return stripped.strip("._")[-100:] or "upload"


def _save_upload(upload: UploadFile, dest: str) -> int:
    """Stream an upload to disk, refusing anything over MAX_UPLOAD_BYTES."""
    written = 0
    with open(dest, "wb") as out:
        while True:
            chunk = upload.file.read(UPLOAD_CHUNK_BYTES)
            if not chunk:
                break
            written += len(chunk)
            if written > MAX_UPLOAD_BYTES:
                raise ValueError(
                    f"file is larger than the {MAX_UPLOAD_BYTES // (1024 * 1024)}MB limit"
                )
            out.write(chunk)
    return written


@app.post("/api/transcribe")
def transcribe(
    request: Request,
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

    def fail(status, detail):
        # Don't leave an orphan run directory behind on any failure.
        shutil.rmtree(work_dir, ignore_errors=True)
        raise HTTPException(status, detail)

    # Resolve the input audio: uploaded file takes priority, else a bundled sample
    if file is not None and file.filename:
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > MAX_UPLOAD_BYTES:
            fail(413, f"that file is larger than the "
                      f"{MAX_UPLOAD_BYTES // (1024 * 1024)}MB limit")
        in_path = os.path.join(work_dir, "input_" + _safe_upload_name(file.filename))
        try:
            _save_upload(file, in_path)
        except ValueError as e:
            fail(413, str(e))
        except OSError as e:
            fail(400, f"could not save the uploaded file: {e}")
    elif sample:
        src = _sample_path(sample) if sample in SAMPLE_IDS else None
        if src is None:
            fail(404, f"sample '{sample}' not found")
        in_path = os.path.join(work_dir, f"input_{sample}.ogg")
        try:
            shutil.copy(src, in_path)
        except OSError as e:
            fail(500, f"could not read sample '{sample}': {e}")
    else:
        fail(400, "provide either an uploaded file or a sample id")

    # Validate at the boundary: confirm it's readable audio before the pipeline,
    # so bad uploads return a clean 400 instead of a bare 500 deep in librosa.
    bad_audio = False
    try:
        import librosa
        probe, _ = librosa.load(in_path, sr=22050, duration=0.5)
        if probe is None or len(probe) == 0:
            bad_audio = True
    except Exception:
        bad_audio = True
    if bad_audio:
        fail(400, "could not read that file as audio — please upload a valid audio file (wav, mp3, ogg, flac).")

    try:
        result = pipeline.run(in_path, inst["stem"], work_dir)
    except Exception as e:
        msg = str(e) or e.__class__.__name__   # some exceptions stringify to ""
        fail(500, f"transcription failed: {msg}")

    # Reclaim disk from older runs now that this one has its artefacts.
    try:
        pipeline.prune_runs(RUNS_DIR, keep=MAX_RUNS_KEPT)
    except OSError as e:  # pragma: no cover - housekeeping must never fail a request
        print(f"[runs] could not prune old runs: {e}")

    return JSONResponse({
        "run_id": run_id,
        "instrument": inst["label"],
        "method": result["method"],
        "tempo": result.get("tempo"),
        "low_accuracy": not inst["good"],
        "stem_energy_ratio": result.get("stem_energy_ratio"),
        "instrument_present": result.get("stem_energy_ratio", 1.0) >= STEM_PRESENT_RATIO,
        "n_notes": result["n_notes"],
        "duration": round(result["duration"], 1),
        "analyzed_duration": round(result.get("analyzed_duration", result["duration"]), 1),
        "truncated": bool(result.get("truncated")),
        "musicxml": result["musicxml"],
        "midi_url": f"/runs/{run_id}/transcription.mid",
        "stem_url": f"/runs/{run_id}/stem_{inst['stem']}.wav",
        "musicxml_url": f"/runs/{run_id}/sheet_music.musicxml",
    })


# Serve the frontend at the root
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
