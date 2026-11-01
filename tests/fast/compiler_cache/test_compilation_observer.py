"""Compiler timing observation preserves an existing listener and cache decisions."""

import json
from types import SimpleNamespace

from miles.utils.compiler_cache import compilation_observer


def test_listener_preserves_callback_and_reports_compiler_work(tmp_path, monkeypatch):
    calls = []
    knobs = SimpleNamespace(compilation=SimpleNamespace(listener=lambda **kw: calls.append(kw)))
    monkeypatch.setattr(compilation_observer.importlib, "import_module", lambda name: knobs)
    compilation_observer.install(tmp_path)
    observer = knobs.compilation.listener
    compilation_observer.install(tmp_path)
    assert knobs.compilation.listener is observer
    for hit in (False, True):
        observer(
            src=SimpleNamespace(name="kernel"),
            metadata={},
            metadata_group={},
            cache_hit=hit,
            times=SimpleNamespace(ir_initialization=11, lowering_stages=[("ptx", 13)], store_results=17),
        )
    records = [json.loads(line) for line in next(tmp_path.glob("activity-*.jsonl")).read_text().splitlines()]
    assert [r["event"] for r in records] == ["compile_miss", "compile_hit"]
    assert all(r["compiler_microseconds"] == 41 for r in records)
    assert len(calls) == 2
