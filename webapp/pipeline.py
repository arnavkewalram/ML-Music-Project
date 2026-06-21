"""
Transcription pipeline used by the web server.

    song -> Demucs (separate one instrument) -> basic-pitch (transcribe)
         -> music21 (light quantization + MusicXML)

Heavy models (Demucs, basic-pitch) are loaded once and cached at module level
so each request only pays for inference, not model loading.
"""

import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import librosa
import soundfile as sf

# Demucs 6-stem source order: drums, bass, other, vocals, guitar, piano
DEMUCS_MODEL = "htdemucs_6s"
VALID_STEMS = ["drums", "bass", "other", "vocals", "guitar", "piano"]

_demucs = None
_demucs_device = None
_bp_model = None


def _get_demucs():
    global _demucs, _demucs_device
    if _demucs is None:
        import torch
        from demucs.pretrained import get_model
        _demucs = get_model(DEMUCS_MODEL)
        _demucs.eval()
        # Prefer MPS (Apple GPU) when available, fall back to CPU
        _demucs_device = "mps" if torch.backends.mps.is_available() else "cpu"
    return _demucs, _demucs_device


def _get_basic_pitch():
    global _bp_model
    if _bp_model is None:
        from basic_pitch.inference import Model
        from basic_pitch import ICASSP_2022_MODEL_PATH
        _bp_model = Model(ICASSP_2022_MODEL_PATH)
    return _bp_model


def warmup():
    """Load both models ahead of the first request."""
    _get_demucs()
    _get_basic_pitch()


def separate(audio_path: str, stem: str, out_path: str) -> str:
    """Isolate one instrument stem from a full mix and write it to out_path."""
    import torch
    from demucs.apply import apply_model

    if stem not in VALID_STEMS:
        raise ValueError(f"stem must be one of {VALID_STEMS}, got '{stem}'")

    model, device = _get_demucs()
    sr = model.samplerate

    audio, _ = librosa.load(audio_path, sr=sr, mono=False)
    if audio.ndim == 1:
        audio = np.stack([audio, audio])
    elif audio.shape[0] == 1:
        audio = np.repeat(audio, 2, axis=0)

    wav = torch.from_numpy(audio).float()
    ref = wav.mean(0)
    wav = (wav - ref.mean()) / (ref.std() + 1e-8)

    try:
        with torch.no_grad():
            sources = apply_model(model, wav[None], device=device, progress=False)[0]
    except Exception:
        # Some ops can be unsupported on MPS; retry on CPU
        with torch.no_grad():
            sources = apply_model(model, wav[None], device="cpu", progress=False)[0]

    sources = sources * ref.std() + ref.mean()
    idx = model.sources.index(stem)
    sf.write(out_path, sources[idx].cpu().numpy().T, sr)
    return out_path


def transcribe(stem_path: str):
    """Transcribe an audio stem to a pretty_midi object."""
    from basic_pitch.inference import predict

    model = _get_basic_pitch()
    _, midi_data, _ = predict(stem_path, model)
    return midi_data


def notate(midi_data, out_path: str, quantize: bool = True) -> str:
    """Convert a pretty_midi transcription to MusicXML; return the XML as a string."""
    import tempfile
    from music21 import converter as m21converter

    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
        mid_tmp = tmp.name
        midi_data.write(mid_tmp)
    try:
        score = m21converter.parse(mid_tmp)
        if quantize:
            try:
                # Snap to 16th-note and triplet grid so the rhythm is readable
                score.quantize((4, 3), inPlace=True, recurse=True)
            except Exception:
                pass
        score.write("musicxml", fp=out_path)
        with open(out_path, "r", encoding="utf-8") as f:
            return f.read()
    finally:
        os.unlink(mid_tmp)


def run(audio_path: str, stem: str, work_dir: str) -> dict:
    """Full chain. Returns paths + the MusicXML string + simple stats."""
    os.makedirs(work_dir, exist_ok=True)
    stem_path = os.path.join(work_dir, f"stem_{stem}.wav")
    midi_path = os.path.join(work_dir, "transcription.mid")
    xml_path = os.path.join(work_dir, "sheet_music.musicxml")

    separate(audio_path, stem, stem_path)
    midi_data = transcribe(stem_path)
    midi_data.write(midi_path)
    musicxml = notate(midi_data, xml_path)

    n_notes = sum(len(inst.notes) for inst in midi_data.instruments)
    duration = float(midi_data.get_end_time())
    return {
        "stem_path": stem_path,
        "midi_path": midi_path,
        "musicxml_path": xml_path,
        "musicxml": musicxml,
        "n_notes": n_notes,
        "duration": duration,
    }
