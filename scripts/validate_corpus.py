"""
Batch validation harness for the Staff transcription app.

Reads webapp/corpus/manifest.json, trims each track for speed, POSTs it through
the RUNNING server (http://127.0.0.1:8000/api/transcribe) exactly like a real
user, and scores the result. Produces a JSON report under webapp/validation/.

Metrics per (track, instrument):
  - ok / error
  - n_notes, method, low_accuracy
  - notes_per_measure + distinct rhythmic values  -> readability
  - overlap_pairs (0 => monophonic) for sanity vs. the routed model
  - pitch range

Usage:
  python scripts/validate_corpus.py [--trim 40] [--server http://127.0.0.1:8000]
"""

import os
import sys
import io
import json
import time
import glob
import argparse
import tempfile
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import librosa
import soundfile as sf
import requests
import pretty_midi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.join(ROOT, "webapp", "corpus")
VALID_DIR = os.path.join(ROOT, "webapp", "validation")
DENSE_THRESHOLD = 20    # notes/measure above this = hard to read
MESSY_RHYTHMS = 10      # distinct rhythmic values above this = ragged/unreadable


def trim(path, seconds):
    y, sr = librosa.load(path, sr=22050, mono=False)
    n = int(seconds * sr)
    y = y[..., :n]
    tmp = tempfile.mktemp(suffix=".wav")
    sf.write(tmp, y.T if y.ndim > 1 else y, sr)
    return tmp


def density(musicxml: str):
    """notes/measure and distinct rhythmic values from a MusicXML string."""
    from music21 import converter
    tmp = tempfile.mktemp(suffix=".musicxml")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(musicxml)
    try:
        sc = converter.parse(tmp)
        notes = list(sc.recurse().notes)
        measures = list(sc.recurse().getElementsByClass("Measure"))
        npm = len(notes) / max(1, len(measures))
        durs = len(set(round(float(n.duration.quarterLength), 3) for n in notes))
        return round(npm, 1), durs
    except Exception:
        return None, None
    finally:
        os.unlink(tmp)


def overlaps(server, midi_url):
    raw = requests.get(server + midi_url, timeout=60).content
    pm = pretty_midi.PrettyMIDI(io.BytesIO(raw))
    notes = sorted((n for inst in pm.instruments for n in inst.notes), key=lambda n: n.start)
    ov = sum(1 for a, b in zip(notes, notes[1:]) if b.start < a.end - 1e-3)
    pitches = [n.pitch for n in notes]
    rng = (min(pitches), max(pitches)) if pitches else (None, None)
    return ov, rng


def run_one(server, audio_path, stem, trim_sec):
    tmp = trim(audio_path, trim_sec)
    try:
        with open(tmp, "rb") as fh:
            r = requests.post(
                server + "/api/transcribe",
                files={"file": (os.path.basename(tmp), fh, "audio/wav")},
                data={"instrument": stem},
                timeout=900,
            )
        if r.status_code != 200:
            return {"ok": False, "error": f"HTTP {r.status_code}: {r.text[:160]}"}
        d = r.json()
        xml = d.get("musicxml", "")
        npm, distinct = density(xml)
        # accidentals shown per 100 notes (lower = cleaner spelling / good key sig)
        n_acc = xml.count("<accidental")
        n_note_tags = max(1, xml.count("<note"))
        acc_per_100 = round(100.0 * n_acc / n_note_tags, 1)
        has_key = "<key>" in xml or "<key " in xml
        ov, rng = overlaps(server, d["midi_url"])
        return {
            "ok": True,
            "method": d.get("method"),
            "n_notes": d.get("n_notes"),
            "low_accuracy": d.get("low_accuracy"),
            "notes_per_measure": npm,
            "distinct_rhythms": distinct,
            "overlap_pairs": ov,
            "pitch_range": rng,
            "tempo": d.get("tempo"),
            "accidentals_per_100": acc_per_100,
            "has_key_sig": has_key,
            "dense": (npm is not None and npm > DENSE_THRESHOLD),
            "messy": (distinct is not None and distinct > MESSY_RHYTHMS),
        }
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trim", type=int, default=40)
    ap.add_argument("--server", default="http://127.0.0.1:8000")
    args = ap.parse_args()

    manifest_path = os.path.join(CORPUS, "manifest.json")
    if not os.path.isfile(manifest_path):
        print(f"No corpus yet at {manifest_path}. Nothing to validate.")
        return 1
    manifest = json.load(open(manifest_path))
    os.makedirs(VALID_DIR, exist_ok=True)

    report = {"trim_sec": args.trim, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "tracks": []}
    print(f"Validating {len(manifest)} tracks (trim={args.trim}s)...\n")

    for entry in manifest:
        path = os.path.join(CORPUS, entry["file"])
        if not os.path.isfile(path):
            print(f"  [skip] missing file: {entry['file']}")
            continue
        stems = entry.get("stems") or ["other"]
        track = {"file": entry["file"], "title": entry.get("title"),
                 "license": entry.get("license"), "results": {}}
        print(f"• {entry.get('title', entry['file'])}")
        for stem in stems:
            res = run_one(args.server, path, stem, args.trim)
            track["results"][stem] = res
            if res["ok"]:
                flags = ("".join([" ⚠DENSE" if res["dense"] else "",
                                  " ⚠MESSY" if res["messy"] else ""]))
                print(f"    {stem:7s} {res['n_notes']:>4} notes | "
                      f"{res['notes_per_measure']} n/measure | "
                      f"{res['distinct_rhythms']} rhythms | "
                      f"{res['overlap_pairs']} overlaps{flags}")
            else:
                print(f"    {stem:7s} ERROR: {res['error']}")
        report["tracks"].append(track)

    out = os.path.join(VALID_DIR, f"report_{time.strftime('%Y%m%d_%H%M%S')}.json")
    json.dump(report, open(out, "w"), indent=2)

    # Aggregate
    ok = dense = messy = errs = total = 0
    for t in report["tracks"]:
        for stem, res in t["results"].items():
            total += 1
            if res["ok"]:
                ok += 1
                dense += int(res["dense"])
                messy += int(res["messy"])
            else:
                errs += 1
    print(f"\n=== {ok}/{total} ok | {errs} errors | {dense} dense | {messy} messy-rhythm ===")
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
