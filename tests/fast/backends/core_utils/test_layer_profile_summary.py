from miles.backends.core_utils import layer_profile


def test_summary_keeps_slowest_rank_and_layer_identity():
    def record(seconds):
        return dict(
            device_seconds=seconds,
            max_call_device_seconds=seconds / 2,
            host_seconds=0.1,
            max_call_host_seconds=0.05,
            calls=2,
        )

    result = layer_profile.summary(
        [
            {"blocks.1/forward": record(2), "blocks.2/backward": record(3)},
            {"blocks.1/forward": record(7), "blocks.2/backward": record(4)},
        ]
    )
    assert result["slowest"]["rank"] == 1
    assert result["slowest"]["layer_phase"] == "blocks.1/forward"
    assert result["metrics"]["profile/max_layer_call_device_seconds"] == 3.5
