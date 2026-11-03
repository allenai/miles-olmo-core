"""Aggregate optional engine telemetry without using generation or health slots."""

import asyncio

import httpx
from prometheus_client import core, exposition, parser, registry

SCRAPE_TIMEOUT_SECONDS = 5.0


async def fetch_engine_metrics(client: httpx.AsyncClient, url: str) -> tuple[str, str | None]:
    try:
        # Bound the whole request, including waiting for the dedicated pool.
        async with asyncio.timeout(SCRAPE_TIMEOUT_SECONDS):
            response = await client.get(f"{url}/metrics")
            response.raise_for_status()
            return url, response.text
    except (httpx.HTTPError, TimeoutError):
        return url, None


def render_engine_metrics(responses: list[tuple[str, str | None]]) -> bytes:
    """Preserve worker labels and expose missing scrapes instead of zero values."""
    families = {}
    availability = core.GaugeMetricFamily(
        "miles_router_engine_metrics_up", "Whether the engine metrics scrape succeeded", labels=["worker_addr"]
    )
    for url, text in responses:
        try:
            parsed = list(parser.text_string_to_metric_families(text)) if text is not None else None
        except ValueError:
            parsed = None
        availability.add_metric([url], int(parsed is not None))
        for metric in parsed or ():
            key = (metric.name, metric.type, metric.unit)
            if key not in families:
                families[key] = core.Metric(metric.name, metric.documentation, metric.type, metric.unit)
            families[key].samples.extend(
                sample._replace(labels={**sample.labels, "worker_addr": url}) for sample in metric.samples
            )

    class Collector:
        def collect(self):
            return (*families.values(), availability)

    output = registry.CollectorRegistry()
    output.register(Collector())
    return exposition.generate_latest(output)


class EngineMetrics:
    def __init__(self):
        self.client = httpx.AsyncClient(limits=httpx.Limits(max_connections=8), timeout=SCRAPE_TIMEOUT_SECONDS)

    async def scrape(self, urls: tuple[str, ...]) -> bytes:
        responses = await asyncio.gather(*(fetch_engine_metrics(self.client, url) for url in urls))
        # Parsing/formatting large metric sets must not block generation routing.
        return await asyncio.to_thread(render_engine_metrics, responses)

    async def close(self):
        await self.client.aclose()
