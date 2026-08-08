"""HTTP contract tests for webapp/server.py.

`pipeline.run` is stubbed by the `client` fixture, so these cover request
validation, error mapping, response shape and file serving — not transcription
quality. See test_e2e.py for the real chain.
"""

import os
import warnings

warnings.filterwarnings("ignore")

import pytest

from webapp import server


class TestConfig:
    def test_lists_instruments_and_samples(self, client):
        cfg = client.get("/api/config").json()
        assert {i["id"] for i in cfg["instruments"]} == set(server.INSTRUMENT_BY_ID)
        assert cfg["samples"]

    def test_every_instrument_maps_to_a_real_demucs_stem(self, client):
        from webapp import pipeline

        cfg = client.get("/api/config").json()
        assert all(i["stem"] in pipeline.VALID_STEMS for i in cfg["instruments"])

    def test_every_sample_suggestion_is_a_real_instrument(self, client):
        cfg = client.get("/api/config").json()
        known = {i["id"] for i in cfg["instruments"]}
        for sample in cfg["samples"]:
            assert set(sample["try"]) <= known, sample["id"]

    def test_every_advertised_sample_file_exists(self, client):
        cfg = client.get("/api/config").json()
        for sample in cfg["samples"]:
            assert client.get(f"/samples/{sample['id']}").status_code == 200


class TestRequestValidation:
    def test_unknown_instrument_is_rejected(self, client):
        r = client.post("/api/transcribe", data={"instrument": "banjo", "sample": "trumpet"})
        assert r.status_code == 400
        assert "banjo" in r.json()["detail"]

    def test_a_missing_instrument_field_is_rejected(self, client):
        assert client.post("/api/transcribe", data={"sample": "trumpet"}).status_code == 422

    def test_no_audio_source_is_rejected(self, client):
        r = client.post("/api/transcribe", data={"instrument": "piano"})
        assert r.status_code == 400
        assert "sample" in r.json()["detail"]

    def test_unknown_sample_is_rejected(self, client):
        r = client.post("/api/transcribe", data={"instrument": "piano", "sample": "nope"})
        assert r.status_code == 404

    def test_undecodable_upload_gets_a_helpful_400(self, client, not_audio):
        with open(not_audio, "rb") as fh:
            r = client.post(
                "/api/transcribe",
                data={"instrument": "piano"},
                files={"file": ("broken.wav", fh, "audio/wav")},
            )
        assert r.status_code == 400
        assert "valid audio file" in r.json()["detail"]

    def test_an_empty_upload_gets_a_helpful_400(self, client, tmp_path):
        empty = tmp_path / "empty.wav"
        empty.write_bytes(b"")
        with open(empty, "rb") as fh:
            r = client.post(
                "/api/transcribe",
                data={"instrument": "piano"},
                files={"file": ("empty.wav", fh, "audio/wav")},
            )
        assert r.status_code == 400


class TestTranscribeResponse:
    def _post(self, client, tone_wav, instrument="piano"):
        with open(tone_wav, "rb") as fh:
            return client.post(
                "/api/transcribe",
                data={"instrument": instrument},
                files={"file": ("tone.wav", fh, "audio/wav")},
            )

    def test_a_valid_upload_succeeds(self, client, tone_wav):
        assert self._post(client, tone_wav).status_code == 200

    def test_the_response_carries_everything_the_ui_renders(self, client, tone_wav):
        body = self._post(client, tone_wav).json()
        expected = {
            "run_id", "instrument", "method", "tempo", "low_accuracy",
            "stem_energy_ratio", "instrument_present", "n_notes", "duration",
            "musicxml", "midi_url", "stem_url", "musicxml_url",
        }
        assert expected <= set(body)

    def test_download_urls_point_at_files_that_exist(self, client, tone_wav):
        body = self._post(client, tone_wav).json()
        for key in ("midi_url", "stem_url", "musicxml_url"):
            assert client.get(body[key]).status_code == 200, key

    def test_the_selected_instrument_reaches_the_pipeline(self, client, tone_wav):
        self._post(client, tone_wav, instrument="bass")
        assert client.pipeline_calls[-1]["stem"] == "bass"

    def test_instruments_without_a_dedicated_stem_are_flagged(self, client, tone_wav):
        assert self._post(client, tone_wav, instrument="other").json()["low_accuracy"] is True
        assert self._post(client, tone_wav, instrument="piano").json()["low_accuracy"] is False

    def test_a_near_silent_stem_is_reported_as_instrument_absent(
        self, client, tone_wav, monkeypatch
    ):
        from webapp import pipeline

        original = pipeline.run
        monkeypatch.setattr(
            pipeline, "run",
            lambda *a, **k: {**original(*a, **k), "stem_energy_ratio": 0.001},
        )

        assert self._post(client, tone_wav).json()["instrument_present"] is False

    def test_a_loud_stem_is_reported_as_instrument_present(self, client, tone_wav):
        assert self._post(client, tone_wav).json()["instrument_present"] is True

    def test_a_pipeline_crash_becomes_a_500_with_a_readable_message(
        self, client, tone_wav, monkeypatch
    ):
        from webapp import pipeline

        def boom(*a, **k):
            raise RuntimeError("demucs ran out of memory")

        monkeypatch.setattr(pipeline, "run", boom)

        r = self._post(client, tone_wav)
        assert r.status_code == 500
        assert "demucs ran out of memory" in r.json()["detail"]

    def test_an_exception_with_no_message_still_produces_a_detail(
        self, client, tone_wav, monkeypatch
    ):
        from webapp import pipeline

        def boom(*a, **k):
            raise ValueError()

        monkeypatch.setattr(pipeline, "run", boom)

        assert "ValueError" in self._post(client, tone_wav).json()["detail"]

    def test_a_failed_run_leaves_no_directory_behind(self, client, tone_wav, monkeypatch):
        from webapp import pipeline

        monkeypatch.setattr(
            pipeline, "run", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("nope"))
        )

        self._post(client, tone_wav)

        assert os.listdir(server.RUNS_DIR) == []


class TestFileServing:
    def test_a_known_sample_is_served(self, client):
        r = client.get("/samples/trumpet")
        assert r.status_code == 200
        assert r.headers["content-type"] == "audio/ogg"

    def test_an_unknown_sample_is_404(self, client):
        assert client.get("/samples/nope").status_code == 404

    @pytest.mark.parametrize("attack", ["../server", "..%2Fserver", "....//server"])
    def test_sample_paths_cannot_escape_the_samples_directory(self, client, attack):
        assert client.get(f"/samples/{attack}").status_code in (400, 404)

    @pytest.mark.parametrize(
        "run_id,filename",
        [("..", "server.py"), ("abc", ".."), ("abc", "../../server.py")],
    )
    def test_run_paths_cannot_escape_the_runs_directory(self, client, run_id, filename):
        r = client.get(f"/runs/{run_id}/{filename}")
        assert r.status_code in (400, 404)

    def test_an_unknown_run_is_404(self, client):
        assert client.get("/runs/deadbeef/transcription.mid").status_code == 404

    def test_the_frontend_is_served_at_the_root(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "<title>" in r.text
