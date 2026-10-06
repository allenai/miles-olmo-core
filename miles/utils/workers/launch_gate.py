import httpx

from miles.utils import infra_timeouts
from miles.utils.http_utils import GeneralHttpClientProvider
from miles.utils.retry_utils import retry_until_deadline

GATE_PORT_NAME = "gate"

_INITIAL_DELAY_SECONDS = 1.0
_MAX_DELAY_SECONDS = 5.0
_ATTEMPT_TIMEOUT_SECONDS = 30.0


async def activate_launch_gate(gate_url: str, timeout: float = 1800.0) -> None:
    async def _activate(remaining_seconds: float) -> None:
        response = await infra_timeouts.request(
            GeneralHttpClientProvider.client(),
            "post",
            f"{gate_url}/gate/activate",
            json={},
            timeout=min(_ATTEMPT_TIMEOUT_SECONDS, remaining_seconds / infra_timeouts.seconds(1)),
        )
        response.raise_for_status()

    await retry_until_deadline(
        _activate,
        total_seconds=infra_timeouts.seconds(timeout),
        retry_on=(httpx.HTTPError, OSError),
        initial_delay=_INITIAL_DELAY_SECONDS,
        max_delay=_MAX_DELAY_SECONDS,
        log_fields=dict(gate_url=gate_url),
    )
