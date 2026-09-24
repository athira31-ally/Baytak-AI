"""Logging + Azure Monitor (Application Insights) via OpenTelemetry.
Enabled automatically when APPLICATIONINSIGHTS_CONNECTION_STRING is set."""
import logging

from app.config import Settings


def setup(settings: Settings) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # The Azure SDKs log every HTTP request/response at INFO - far too noisy for container logs
    for noisy in ("azure", "azure.core.pipeline.policies.http_logging_policy", "azure.monitor", "httpx", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    if settings.applicationinsights_connection_string:
        from azure.monitor.opentelemetry import configure_azure_monitor

        configure_azure_monitor(connection_string=settings.applicationinsights_connection_string,
                                logger_name="homematch")
        logging.getLogger(__name__).info("Azure Monitor telemetry enabled")


# Structured events land in App Insights `traces` with customDimensions
events = logging.getLogger("homematch")


# ------------------------------------------------------------------ tracing
# OpenTelemetry spans for every agent step and tool call. With App Insights configured they show up
# as an end-to-end transaction (request -> supervisor -> specialists -> tools -> writer) in
# Application Insights > Transaction search; without it they are no-ops.
from contextlib import contextmanager  # noqa: E402

from opentelemetry import trace  # noqa: E402

tracer = trace.get_tracer("baytak.agents")


@contextmanager
def span(name: str, **attributes):
    with tracer.start_as_current_span(name) as s:
        for k, v in attributes.items():
            if v is not None:
                s.set_attribute(f"baytak.{k}", v if isinstance(v, (str, int, float, bool)) else str(v))
        yield s
