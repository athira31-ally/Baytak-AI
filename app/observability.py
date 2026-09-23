"""Logging + Azure Monitor (Application Insights) via OpenTelemetry.
Enabled automatically when APPLICATIONINSIGHTS_CONNECTION_STRING is set."""
import logging

from app.config import Settings


def setup(settings: Settings) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if settings.applicationinsights_connection_string:
        from azure.monitor.opentelemetry import configure_azure_monitor

        configure_azure_monitor(connection_string=settings.applicationinsights_connection_string,
                                logger_name="homematch")
        logging.getLogger(__name__).info("Azure Monitor telemetry enabled")


# Structured events land in App Insights `traces` with customDimensions
events = logging.getLogger("homematch")
