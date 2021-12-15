import os
import numpy as np
from typing import Callable

from prometheus_client import Histogram, Counter
from prometheus_fastapi_instrumentator import Instrumentator, metrics
from prometheus_fastapi_instrumentator.metrics import Info

NAMESPACE = os.environ.get("METRICS_NAMESPACE", "visdrone")
SUBSYSTEM = os.environ.get("METRICS_SUBSYSTEM", "model")

COUNT_BUCKETS = list(np.linspace(start=30, stop=300, num=10)) + [float("inf")]
VIDEO_BUCKETS = (1, 2, 3, 5, 8, 13, 21, 34, 55, float("inf"))

METRICS = [
    metrics.request_size,
    metrics.response_size,
    metrics.latency,
    lambda should_include_handler,
           should_include_method,
           should_include_status,
           metric_namespace,
           metric_subsystem:
    metrics.latency(
        should_include_handler=True,
        should_include_method=True,
        should_include_status=True,
        metric_namespace=NAMESPACE,
        metric_subsystem=SUBSYSTEM,
        metric_name='video_time',
        buckets=VIDEO_BUCKETS
    ),
    metrics.requests,
]


# ----- custom metrics -----
def count_output() -> Callable[[Info], None]:
    """
    Used to create count_output custom metric.
    count_output returns an histogram of the prediction counting result
    """
    METRIC = Histogram(
        name="people_count",
        documentation="People count",
        buckets=COUNT_BUCKETS,
        namespace=NAMESPACE,
        subsystem=SUBSYSTEM,
    )

    def instrumentation(info: Info) -> None:
        if info.modified_handler == '/predictions/images':
            counting = info.response.headers.get('count')
            if counting:
                METRIC.observe(float(counting))

    return instrumentation


def query_parameter_count() -> Callable[[Info], None]:
    counts = Counter('query_parameter_count', 'counts the various query parameters',
                     ['query_count'],
                     namespace=NAMESPACE,
                     subsystem=SUBSYSTEM,
                     )

    def instrumentation(info: Info) -> None:
        if info.request.query_params:
            if info.request.query_params.get('count') == 'true' and info.request.query_params.get('heatmap') == 'false':
                counts.labels('count').inc()
            if info.request.query_params.get('heatmap') == 'true' and info.request.query_params.get('count') == 'false':
                counts.labels('heatmap').inc()
            if info.request.query_params.get('heatmap') == 'true' and info.request.query_params.get('count') == 'true':
                counts.labels('both').inc()

    return instrumentation


CUSTOM_METRICS = [
    count_output,
    query_parameter_count
]


def initialize_instrumentator():
    """
    Create and initialize instrumentator by adding all the metrics given in METRICS and CUSTOM METRICS
    """
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

    for metric in CUSTOM_METRICS:
        instrumentator.add(metric())
    return instrumentator
