"""Diagnostic snapshots preserve pre-conversion data and flush or report failure."""

import asyncio
from types import SimpleNamespace

import pytest
import torch

from miles.backends.core_utils.rollout.diagnostic_writer import DiagnosticWriter
from miles.utils.types import Sample


def args(tmp_path):
    return SimpleNamespace(
        rollout_sample_rate=1.0,
        rollout_seed=1,
        save_debug_rollout_data=str(tmp_path / "rollout_{rollout_id}.pt"),
        save_debug_trajectory_data=None,
    )


def test_writer_snapshots_mutable_fields_and_flushes(tmp_path):
    async def scenario():
        writer = DiagnosticWriter(args(tmp_path))
        sample = Sample(index=0, group_index=0, tokens=[1, 2], response_length=1, reward=1.0)
        sample.loss_mask = [1]
        sample.metadata = {"before": [1]}
        metadata = {"version": [2]}
        metrics = await writer.submit([sample], 3, metadata, None)
        sample.loss_mask[0] = 0
        sample.tokens[0] = 99
        sample.metadata["before"][0] = 9
        metadata["version"][0] = 9
        await writer.close()
        saved = torch.load(tmp_path / "rollout_3.pt", weights_only=False)
        assert saved["samples"][0]["loss_mask"] == [1]
        assert saved["samples"][0]["tokens"] == [1, 2]
        assert saved["samples"][0]["metadata"] == {"before": [1]}
        assert saved["metadata"] == {"version": [2]}
        assert metrics["rollout/diagnostics/samples"] == 1

    asyncio.run(scenario())


def test_writer_failure_is_not_silently_dropped(tmp_path):
    async def scenario():
        config = args(tmp_path)
        config.save_debug_rollout_data = str(tmp_path / "no_placeholder.pt")
        writer = DiagnosticWriter(config)
        await writer.submit([Sample(index=0, group_index=0, tokens=[1, 2], response_length=1)], 0, {}, None)
        with pytest.raises(AssertionError, match="placeholder"):
            await writer.close()

    asyncio.run(scenario())
