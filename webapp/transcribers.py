"""
Per-instrument transcription. Each Demucs stem is routed to the model that is
actually best at that kind of audio, instead of one generic model for everything:

    piano   -> ByteDance high-resolution piano model (near state-of-the-art)
    bass    -> monophonic pitch tracking (pYIN), low register
    vocals  -> monophonic pitch tracking (pYIN), melody register
    guitar  -> basic-pitch (general polyphonic)
    other   -> basic-pitch (general polyphonic)
    drums   -> onset detection + frequency-band classification (GM drum map)

Every transcriber returns (pretty_midi.PrettyMIDI, method_label). Heavy models
are cached at module level so only the first request pays to load them.
"""

import os
import tempfile
import threading
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import librosa
import pretty_midi

# ---------------------------------------------------------------- model caches
# Guarded by locks: FastAPI serves sync endpoints from a threadpool, so two
# first-requests can race here and each load a few hundred megabytes of weights.
_piano = None
_piano_lock = threading.Lock()
_bp_model = None
_bp_lock = threading.Lock()


# ByteDance checkpoint (the package itself downloads this with `wget`, which
# isn't always present; we fetch it with urllib so setup works anywhere).
_PIANO_CKPT = os.path.join(
    os.path.expanduser("~"), "piano_transcription_inference_data",
    "note_F1=0.9677_pedal_F1=0.9186.pth")
_PIANO_CKPT_URL = ("https://zenodo.org/record/4034264/files/"
                   "CRNN_note_F1%3D0.9677_pedal_F1%3D0.9186.pth?download=1")


def _ensure_piano_checkpoint():
    if os.path.exists(_PIANO_CKPT):
        return
    import urllib.request
    os.makedirs(os.path.dirname(_PIANO_CKPT), exist_ok=True)
    print("[transcribers] downloading ByteDance piano checkpoint (~165MB)...")
    urllib.request.urlretrieve(_PIANO_CKPT_URL, _PIANO_CKPT)


def _get_piano():
    global _piano
    if _piano is None:
        with _piano_lock:
            if _piano is None:
                from piano_transcription_inference import PianoTranscription
                _ensure_piano_checkpoint()
                # The model has ops MPS doesn't support; CPU is reliable and fast enough.
                _piano = PianoTranscription(device="cpu", checkpoint_path=_PIANO_CKPT)
    return _piano


def _get_basic_pitch():
    global _bp_model
    if _bp_model is None:
        with _bp_lock:
            if _bp_model is None:
                from basic_pitch.inference import Model
                from basic_pitch import ICASSP_2022_MODEL_PATH
                _bp_model = Model(ICASSP_2022_MODEL_PATH)
    return _bp_model


def warmup():
    """Preload the transcription models (best-effort)."""
    try:
        _get_basic_pitch()
    except Exception as e:
        print(f"[transcribers] basic-pitch preload skipped: {e}")
    try:
        _get_piano()
    except Exception as e:
        print(f"[transcribers] piano model preload skipped: {e}")


# ------------------------------------------------------------- piano (ByteDance)
def transcribe_piano(stem_path: str):
    from piano_transcription_inference import sample_rate as PIANO_SR

    audio, _ = librosa.load(stem_path, sr=PIANO_SR, mono=True)
    tr = _get_piano()
    # NamedTemporaryFile creates the file atomically (no mktemp TOCTOU race).
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tf:
        tmp = tf.name
    try:
        tr.transcribe(audio, tmp)
        pm = pretty_midi.PrettyMIDI(tmp)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return pm, "ByteDance high-resolution piano model"


# --------------------------------------------------------- monophonic (pYIN)
# Frames of median filtering applied to the pitch track before it is rounded to
# semitones. At hop 512 / 22.05kHz each frame is 23ms, so 9 frames is ~209ms —
# just over one cycle of typical 5.5Hz vibrato.
#
# Without it, a note with wide vibrato (or one sung near a semitone boundary)
# flickers between two rounded pitches every few frames, every fragment lands
# under `min_dur`, and the whole note is discarded: measured 0.09s notated out
# of a 3.0s tone. At 9 frames the same tone comes back as one 2.90s note, while
# a 16th-note scale at 120 BPM is transcribed identically to no smoothing at
# all. Larger kernels start swallowing fast passages.
PITCH_SMOOTHING_FRAMES = 9


def _smooth_pitch_track(track: np.ndarray, frames: int) -> np.ndarray:
    """Median-filter a per-frame pitch track, treating NaN as 'unvoiced'."""
    if frames <= 1:
        return track
    half = frames // 2
    out = np.full_like(track, np.nan)
    for i in np.flatnonzero(~np.isnan(track)):
        window = track[max(0, i - half):i + half + 1]
        window = window[~np.isnan(window)]
        if len(window):
            out[i] = np.median(window)
    return out


def _f0_to_midi(stem_path: str, fmin: float, fmax: float, program: int, min_dur: float = 0.08):
    """Track a single melodic line with pYIN and segment it into notes."""
    sr = 22050
    hop = 512
    y, _ = librosa.load(stem_path, sr=sr, mono=True)
    f0, voiced, _ = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=sr,
                                 frame_length=2048, hop_length=hop)

    usable = voiced & ~np.isnan(f0)
    pitch_track = np.full(len(f0), np.nan)
    pitch_track[usable] = librosa.hz_to_midi(f0[usable])
    pitch_track = _smooth_pitch_track(pitch_track, PITCH_SMOOTHING_FRAMES)

    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=program)

    cur_pitch, start_t = None, 0.0
    n_frames = len(f0)
    for i in range(n_frames):
        t = i * hop / sr
        if np.isnan(pitch_track[i]):
            midi = None
        else:
            midi = int(round(float(pitch_track[i])))
        if midi != cur_pitch:
            if cur_pitch is not None and (t - start_t) >= min_dur:
                inst.notes.append(pretty_midi.Note(
                    velocity=90, pitch=int(np.clip(cur_pitch, 0, 127)),
                    start=start_t, end=t))
            cur_pitch, start_t = midi, t
    # close the final note
    end_t = n_frames * hop / sr
    if cur_pitch is not None and (end_t - start_t) >= min_dur:
        inst.notes.append(pretty_midi.Note(
            velocity=90, pitch=int(np.clip(cur_pitch, 0, 127)),
            start=start_t, end=end_t))

    pm.instruments.append(inst)
    return pm


def transcribe_bass(stem_path: str):
    pm = _f0_to_midi(stem_path, fmin=41.0, fmax=400.0, program=33)   # E1..~G4, electric bass
    return pm, "monophonic pitch tracking (pYIN), bass register"


def transcribe_vocals(stem_path: str):
    pm = _f0_to_midi(stem_path, fmin=82.0, fmax=1000.0, program=53)  # E2..~B5, voice
    return pm, "monophonic pitch tracking (pYIN), melody register"


# ----------------------------------------------------------- polyphonic (basic-pitch)
def transcribe_polyphonic(stem_path: str):
    from basic_pitch.inference import predict
    _, midi_data, _ = predict(stem_path, _get_basic_pitch())
    return midi_data, "basic-pitch (general polyphonic)"


# ------------------------------------------------------------------ drums (onsets)
DRUM_KICK, DRUM_SNARE, DRUM_HIHAT = 36, 38, 42  # General MIDI percussion


def transcribe_drums(stem_path: str):
    sr = 22050
    y, _ = librosa.load(stem_path, sr=sr, mono=True)
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time", backtrack=True)

    pm = pretty_midi.PrettyMIDI()
    drum = pretty_midi.Instrument(program=0, is_drum=True, name="Drums")
    for t in onsets:
        i0 = int(t * sr)
        seg = y[i0:i0 + int(0.05 * sr)]
        if len(seg) < 16:
            continue
        S = np.abs(np.fft.rfft(seg))
        freqs = np.fft.rfftfreq(len(seg), 1 / sr)
        low = S[freqs < 150].sum()
        mid = S[(freqs >= 150) & (freqs < 2000)].sum()
        high = S[freqs >= 2000].sum()
        if low >= mid and low >= high:
            pitch = DRUM_KICK
        elif high > mid:
            pitch = DRUM_HIHAT
        else:
            pitch = DRUM_SNARE
        drum.notes.append(pretty_midi.Note(velocity=100, pitch=pitch, start=float(t), end=float(t) + 0.08))

    pm.instruments.append(drum)
    return pm, "onset detection + frequency-band classification"


# ------------------------------------------------------------------------ router
ROUTER = {
    "piano": transcribe_piano,
    "bass": transcribe_bass,
    "vocals": transcribe_vocals,
    "guitar": transcribe_polyphonic,
    "other": transcribe_polyphonic,
    "drums": transcribe_drums,
}

# Stems that are one line by nature. A bass or a voice cannot sound two pitches
# at once, so anything overlapping in the output is an artefact of independent
# rounding rather than something the player performed.
MONOPHONIC_STEMS = frozenset({"bass", "vocals"})


def transcribe(stem: str, stem_path: str):
    """Route a stem to its specialist transcriber. Returns (pretty_midi, method)."""
    fn = ROUTER.get(stem, transcribe_polyphonic)
    try:
        return fn(stem_path)
    except Exception as e:
        # Never hard-fail a request because a specialist errored; fall back.
        if fn is not transcribe_polyphonic:
            print(f"[transcribers] '{stem}' specialist failed ({e}); falling back to basic-pitch")
            return transcribe_polyphonic(stem_path)
        raise
