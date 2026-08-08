"""Tests for input validation, resource limits and concurrency safety."""

import os
import sys
import threading
import types
import warnings

warnings.filterwarnings("ignore")

import pytest

from webapp import pipeline, server


class TestSampleAllowlist:
    """`sample` is a form field that used to be pasted straight into a path."""

    def test_a_traversing_sample_id_is_rejected(self, client, tmp_path):
        # Plant a real .ogg outside the samples directory: the old existence
        # check would have found and transcribed it.
        outside = tmp_path / "outside.ogg"
        outside.write_bytes(b"OggS-not-really-but-the-check-was-os.path.isfile")
        relative = os.path.relpath(str(outside.with_suffix("")), server.SAMPLES_DIR)

        r = client.post("/api/transcribe", data={"instrument": "piano", "sample": relative})

        assert r.status_code == 404
        assert client.pipeline_calls == []

    @pytest.mark.parametrize(
        "attack",
        ["../corpus/anything", "../../etc/passwd", "..%2F..%2Fsecret", "/etc/passwd"],
    )
    def test_traversal_attempts_never_reach_the_pipeline(self, client, attack):
        r = client.post("/api/transcribe", data={"instrument": "piano", "sample": attack})

        assert r.status_code == 404
        assert client.pipeline_calls == []

    def test_a_rejected_sample_leaves_no_run_directory(self, client):
        client.post("/api/transcribe", data={"instrument": "piano", "sample": "../x"})
        assert os.listdir(server.RUNS_DIR) == []

    def test_an_ogg_in_the_samples_directory_is_still_rejected_if_unlisted(
        self, client, tmp_path, monkeypatch
    ):
        """Membership of the published list is the rule, not merely existing."""
        fake_samples = tmp_path / "samples"
        fake_samples.mkdir()
        (fake_samples / "unlisted.ogg").write_bytes(b"OggS")
        monkeypatch.setattr(server, "SAMPLES_DIR", str(fake_samples))

        assert client.get("/samples/unlisted").status_code == 404
        r = client.post("/api/transcribe", data={"instrument": "piano", "sample": "unlisted"})
        assert r.status_code == 404

    def test_listed_samples_still_work(self, client):
        r = client.post("/api/transcribe", data={"instrument": "piano", "sample": "trumpet"})
        assert r.status_code == 200


class TestUploadLimits:
    def test_an_oversized_body_is_refused_before_it_is_processed(
        self, client, tone_wav, monkeypatch
    ):
        monkeypatch.setattr(server, "MAX_UPLOAD_BYTES", 64)

        with open(tone_wav, "rb") as fh:
            r = client.post(
                "/api/transcribe",
                data={"instrument": "piano"},
                files={"file": ("tone.wav", fh, "audio/wav")},
            )

        assert r.status_code == 413
        assert "limit" in r.json()["detail"]
        assert client.pipeline_calls == []
        assert os.listdir(server.RUNS_DIR) == []

    def test_the_streaming_guard_stops_a_body_with_no_declared_length(
        self, tmp_path, tone_wav, monkeypatch
    ):
        """Chunked uploads arrive without Content-Length, so the copy must count."""
        monkeypatch.setattr(server, "MAX_UPLOAD_BYTES", 128)
        upload = types.SimpleNamespace(file=open(tone_wav, "rb"))
        dest = str(tmp_path / "out.wav")

        with pytest.raises(ValueError, match="larger than"):
            server._save_upload(upload, dest)

        upload.file.close()

    def test_an_upload_within_the_limit_is_written_whole(self, tmp_path, tone_wav):
        upload = types.SimpleNamespace(file=open(tone_wav, "rb"))
        dest = str(tmp_path / "out.wav")

        written = server._save_upload(upload, dest)

        upload.file.close()
        assert written == os.path.getsize(tone_wav)
        assert os.path.getsize(dest) == written


class TestUploadFilenames:
    @pytest.mark.parametrize(
        "hostile",
        ["../../../etc/passwd.wav", "..\\..\\windows\\system32.wav", "/absolute/path.wav",
         "....//....//escape.wav", "", "..."],
    )
    def test_hostile_filenames_reduce_to_one_harmless_component(self, hostile):
        safe = server._safe_upload_name(hostile)

        assert safe
        assert "/" not in safe and "\\" not in safe
        assert ".." not in safe
        assert not os.path.isabs(safe)

    def test_ordinary_filenames_are_left_recognisable(self):
        assert server._safe_upload_name("My Song (live).mp3") == "My_Song_live_.mp3"

    def test_an_upload_is_written_inside_its_own_run_directory(self, client, tone_wav):
        with open(tone_wav, "rb") as fh:
            client.post(
                "/api/transcribe",
                data={"instrument": "piano"},
                files={"file": ("../../escape.wav", fh, "audio/wav")},
            )

        call = client.pipeline_calls[-1]
        assert os.path.realpath(call["audio_path"]).startswith(
            os.path.realpath(call["work_dir"]) + os.sep
        )


class TestRunRetention:
    def _make_runs(self, root, n):
        import time

        paths = []
        for i in range(n):
            p = os.path.join(root, f"run{i:02d}")
            os.makedirs(p)
            os.utime(p, (1_000_000 + i, 1_000_000 + i))  # oldest first
            paths.append(p)
            time.sleep(0)
        return paths

    def test_only_the_newest_runs_survive(self, tmp_path):
        root = str(tmp_path)
        paths = self._make_runs(root, 6)

        removed = pipeline.prune_runs(root, keep=2)

        assert removed == 4
        assert sorted(os.listdir(root)) == ["run04", "run05"]
        assert not os.path.exists(paths[0])

    def test_nothing_is_removed_below_the_threshold(self, tmp_path):
        self._make_runs(str(tmp_path), 3)
        assert pipeline.prune_runs(str(tmp_path), keep=10) == 0
        assert len(os.listdir(str(tmp_path))) == 3

    def test_a_missing_runs_directory_is_not_an_error(self, tmp_path):
        assert pipeline.prune_runs(str(tmp_path / "nope"), keep=5) == 0

    def test_loose_files_are_left_alone(self, tmp_path):
        (tmp_path / "notes.txt").write_text("keep me")
        self._make_runs(str(tmp_path), 3)

        pipeline.prune_runs(str(tmp_path), keep=1)

        assert (tmp_path / "notes.txt").exists()

    def test_a_successful_transcription_prunes_old_runs(
        self, client, tone_wav, monkeypatch
    ):
        monkeypatch.setattr(server, "MAX_RUNS_KEPT", 2)
        for i in range(4):
            stale = os.path.join(server.RUNS_DIR, f"stale{i}")
            os.makedirs(stale)
            os.utime(stale, (1_000_000 + i, 1_000_000 + i))

        with open(tone_wav, "rb") as fh:
            r = client.post(
                "/api/transcribe",
                data={"instrument": "piano"},
                files={"file": ("tone.wav", fh, "audio/wav")},
            )

        assert r.status_code == 200
        surviving = os.listdir(server.RUNS_DIR)
        assert len(surviving) == 2
        assert r.json()["run_id"] in surviving, "the run just created must not be pruned"

    def test_the_new_run_is_still_downloadable_afterwards(
        self, client, tone_wav, monkeypatch
    ):
        monkeypatch.setattr(server, "MAX_RUNS_KEPT", 1)
        with open(tone_wav, "rb") as fh:
            body = client.post(
                "/api/transcribe",
                data={"instrument": "piano"},
                files={"file": ("tone.wav", fh, "audio/wav")},
            ).json()

        assert client.get(body["midi_url"]).status_code == 200


class TestAnalysisWindow:
    def test_a_short_input_is_not_truncated(self, tone_wav):
        full, analysed, truncated = pipeline.analysis_window(tone_wav)

        assert truncated is False
        assert analysed == pytest.approx(full)

    def test_a_long_input_is_capped_and_flagged(self, tone_wav, monkeypatch):
        monkeypatch.setattr(pipeline, "MAX_ANALYSIS_SECONDS", 0.4)

        full, analysed, truncated = pipeline.analysis_window(tone_wav)

        assert truncated is True
        assert analysed == 0.4
        assert full > analysed

    def test_the_energy_ratio_compares_equal_length_windows(self, tmp_path):
        """Scoring a truncated stem against the whole mix reports it as absent.

        A song with a quiet intro and a loud body: the stem covers only the
        analysed window, so measuring the mix over its full length makes a
        perfectly separated instrument look like silence.
        """
        import numpy as np
        import soundfile as sf

        sr = 22050
        quiet = 0.03 * np.sin(2 * np.pi * 440 * np.linspace(0, 0.5, int(sr * 0.5)))
        loud = 1.00 * np.sin(2 * np.pi * 440 * np.linspace(0, 1.5, int(sr * 1.5)))
        mix = str(tmp_path / "mix.wav")
        stem = str(tmp_path / "stem.wav")
        sf.write(mix, np.concatenate([quiet, loud]).astype("float32"), sr)
        sf.write(stem, quiet.astype("float32"), sr)  # a perfect 0.5s separation

        naive = pipeline._stem_energy_ratio(mix, stem, max_seconds=2.0)
        windowed = pipeline._stem_energy_ratio(mix, stem, max_seconds=0.5)

        assert naive < server.STEM_PRESENT_RATIO, "sanity: the old comparison misreports"
        assert windowed == pytest.approx(1.0, abs=0.05)
        assert windowed >= server.STEM_PRESENT_RATIO

    def test_the_response_reports_truncation(self, client, tone_wav, monkeypatch):
        from webapp import pipeline as pl

        original = pl.run
        monkeypatch.setattr(
            pl, "run",
            lambda *a, **k: {**original(*a, **k),
                             "duration": 900.0, "analyzed_duration": 360.0,
                             "truncated": True},
        )

        with open(tone_wav, "rb") as fh:
            body = client.post(
                "/api/transcribe",
                data={"instrument": "piano"},
                files={"file": ("tone.wav", fh, "audio/wav")},
            ).json()

        assert body["truncated"] is True
        assert body["duration"] == 900.0
        assert body["analyzed_duration"] == 360.0


N_RACERS = 8
LOAD_DELAY = 0.05  # wide enough that an unguarded check-then-set loses the race


def _race(target):
    """Call `target` from N threads at once; return (results, errors)."""
    start = threading.Event()
    results, errors = [], []

    def worker():
        start.wait(timeout=10)
        try:
            results.append(target())
        except Exception as e:  # surface thread failures instead of hanging
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(N_RACERS)]
    for t in threads:
        t.start()
    start.set()
    for t in threads:
        t.join(timeout=20)
    return results, errors


class TestModelLoadingIsThreadSafe:
    """FastAPI serves sync endpoints from a threadpool; the caches raced.

    Each loader pulls in hundreds of megabytes of weights, so losing this race
    means several copies resident at once — and two threads publishing a
    half-built model to the same global.
    """

    def test_demucs_is_loaded_exactly_once_under_concurrency(self, monkeypatch):
        import time

        loads = []

        def slow_get_model(name):
            time.sleep(LOAD_DELAY)
            loads.append(name)
            return types.SimpleNamespace(eval=lambda: None, samplerate=44100,
                                         sources=list(pipeline.VALID_STEMS))

        monkeypatch.setitem(
            sys.modules, "torch",
            types.SimpleNamespace(backends=types.SimpleNamespace(
                mps=types.SimpleNamespace(is_available=lambda: False))),
        )
        monkeypatch.setitem(
            sys.modules, "demucs.pretrained",
            types.SimpleNamespace(get_model=slow_get_model),
        )
        monkeypatch.setattr(pipeline, "_demucs", None)
        monkeypatch.setattr(pipeline, "_demucs_device", None)

        results, errors = _race(pipeline._get_demucs)

        assert errors == []
        assert len(results) == N_RACERS
        assert len(loads) == 1, f"model loaded {len(loads)} times"
        assert len({id(model) for model, _ in results}) == 1
        assert all(device == "cpu" for _, device in results)

    def test_basic_pitch_is_loaded_exactly_once_under_concurrency(self, monkeypatch):
        import time

        from webapp import transcribers

        loads = []

        def slow_model(path):
            time.sleep(LOAD_DELAY)
            loads.append(path)
            return object()

        monkeypatch.setitem(
            sys.modules, "basic_pitch",
            types.SimpleNamespace(ICASSP_2022_MODEL_PATH="fake-path"),
        )
        monkeypatch.setitem(
            sys.modules, "basic_pitch.inference",
            types.SimpleNamespace(Model=slow_model),
        )
        monkeypatch.setattr(transcribers, "_bp_model", None)

        results, errors = _race(transcribers._get_basic_pitch)

        assert errors == []
        assert len(loads) == 1, f"model loaded {len(loads)} times"
        assert len({id(m) for m in results}) == 1
