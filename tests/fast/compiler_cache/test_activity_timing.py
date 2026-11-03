"""Timestamp cache activity without changing cache decisions."""

import json
from types import SimpleNamespace

from miles.utils.compiler_cache import startup_cache


def test_activity_has_time_key_and_artifact_without_payload(tmp_path, monkeypatch):
    class Manager:
        cache_dir = "/tmp/triton/abcdef"

        def get_group(self, filename):
            return None

        def put(self, data, filename, binary=True):
            return "/tmp/compiled"

    monkeypatch.setattr(
        startup_cache.importlib, "import_module", lambda name: SimpleNamespace(FileCacheManager=Manager)
    )
    startup_cache.observe_triton(tmp_path)
    manager = Manager()
    assert manager.get_group("kernel.json") is None
    assert manager.put("private_payload", filename="kernel.cubin") == "/tmp/compiled"
    lines = next(tmp_path.glob("activity-*.jsonl")).read_text()
    records = [json.loads(line) for line in lines.splitlines()]
    assert [r["event"] for r in records] == ["group_miss", "put"]
    assert [r["filename"] for r in records] == ["kernel.json", "kernel.cubin"]
    assert all(r["timestamp"] > 0 and r["operation_seconds"] >= 0 and r["cache_key"] == "abcdef" for r in records)
    assert "private_payload" not in lines


def test_activity_archive_preserves_timestamps(tmp_path):
    local = tmp_path / "local"
    local.mkdir()
    payload = '{"event":"group_miss","timestamp":12.5}\n'
    (local / "activity-123.jsonl").write_text(payload)
    target = tmp_path / "reports" / "activity" / "trainer"
    assert startup_cache.archive_activity({"activity_directory": str(target)}, local) == str(target)
    assert (target / "activity-123.jsonl").read_text() == payload
    assert (local / "activity-123.jsonl").read_text() == payload
