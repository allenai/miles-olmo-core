"""Large initialization copies get a separate bound from serving-cell probes."""

from tests.fast.ray.rollout import test_inference_controller_tick as tick_tests

from miles.ray.rollout import inference_controller as controller_module


async def test_slow_initialization_finishes_while_a_wedged_serving_probe_times_out(monkeypatch):
    initializing = tick_tests._RecordingCell(delay=0.02, cell_id="initializing")
    initializing.is_initializing = True
    serving = tick_tests._RecordingCell(delay=60.0, cell_id="serving")
    controller = tick_tests._make_controller({"default": tick_tests._StubServer({"a": initializing, "b": serving})})
    monkeypatch.setattr(controller_module, "CELL_TICK_TIMEOUT_SECONDS", 0.005)
    monkeypatch.setattr(controller_module, "CELL_INITIALIZATION_TICK_TIMEOUT_SECONDS", 1.0)

    await controller._tick_cells()

    assert initializing.finished_count == 1
    assert serving.finished_count == 0


async def test_initialization_still_has_a_deadline(monkeypatch):
    initializing = tick_tests._RecordingCell(delay=60.0, cell_id="initializing")
    initializing.is_initializing = True
    healthy = tick_tests._RecordingCell(cell_id="healthy")
    controller = tick_tests._make_controller({"default": tick_tests._StubServer({"a": initializing, "b": healthy})})
    monkeypatch.setattr(controller_module, "CELL_INITIALIZATION_TICK_TIMEOUT_SECONDS", 0.01)

    await controller._tick_cells()

    assert initializing.finished_count == 0
    assert healthy.finished_count == 1
