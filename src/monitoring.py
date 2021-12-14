import os
from typing import Callable

from prometheus_client import Histogram
from prometheus_fastapi_instrumentator import Instrumentator, metrics
from prometheus_fastapi_instrumentator.metrics import Info

NAMESPACE = os.environ.get("METRICS_NAMESPACE", "visdrone")
SUBSYSTEM = os.environ.get("METRICS_SUBSYSTEM", "model")

COUNT_BUCKETS = (30.0, 60.0, 90.0, 120.0, 150.0, 180.0, 210.0, 240.0, 250.0, float("inf"))

METRICS = [
    metrics.request_size,
    metrics.response_size,
    metrics.latency,
    metrics.requests,
]

# ----- custom metrics -----
def count_output(
        metric_name: str = "people_count",
        metric_doc: str = "People count",
        metric_namespace: str = NAMESPACE,
        metric_subsystem: str = SUBSYSTEM,
        buckets=COUNT_BUCKETS,
) -> Callable[[Info], None]:
    METRIC = Histogram(
        metric_name,
        metric_doc,
        buckets=buckets,
        namespace=metric_namespace,
        subsystem=metric_subsystem,
    )

    def instrumentation(info: Info) -> None:
        if info.modified_handler == '/predictions/images':
            counting = info.response.headers['count']
            if counting:
                print('Conteggio header' + str(counting))
                METRIC.observe(float(counting))

    return instrumentation


def initialize_instrumentator():
    instrumentator = Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=True,
        should_instrument_requests_inprogress=True,
        excluded_handlers=["/metrics"],
        inprogress_name="fastapi_inprogress",
        inprogress_labels=True,
    )

    # ----- add metrics -----
    for metric in METRICS:
        instrumentator.add(
            metric(
                should_include_handler=True,
                should_include_method=True,
                should_include_status=True,
                metric_namespace=NAMESPACE,
                metric_subsystem=SUBSYSTEM,
            )
        )

    buckets = COUNT_BUCKETS
    instrumentator.add(
        count_output(metric_namespace=NAMESPACE, metric_subsystem=SUBSYSTEM, buckets=buckets)
    )
    return instrumentator