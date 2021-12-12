import os
from typing import Callable

import numpy as np
from prometheus_client import Histogram
from prometheus_fastapi_instrumentator import Instrumentator, metrics
from prometheus_fastapi_instrumentator.metrics import Info

NAMESPACE = os.environ.get("METRICS_NAMESPACE", "visdrone")
SUBSYSTEM = os.environ.get("METRICS_SUBSYSTEM", "model")


# ----- custom metrics -----
def count_output(
        metric_name: str = "people_count",
        metric_doc: str = "People count",
        metric_namespace: str = NAMESPACE,
        metric_subsystem: str = SUBSYSTEM,
        buckets=(0, 30, 60, 90, 120, 150, 180, 210, 240, 250, float("inf")),
) -> Callable[[Info], None]:
    METRIC = Histogram(
        metric_name,
        metric_doc,
        buckets=buckets,
        namespace=metric_namespace,
        subsystem=metric_subsystem,
    )

    def instrumentation(info: Info) -> None:
        if info.modified_handler == "/predictions/images?count=true&heatmap=true":
            counting = info.response.headers['count']
            if counting:
                METRIC.observe(float(counting))
        elif info.modified_handler == "/predictions/images?count=true&heatmap=false":
            counting = info.response.body.json()['count']
            if counting:
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
    instrumentator.add(
        metrics.request_size(
            should_include_handler=True,
            should_include_method=True,
            should_include_status=True,
            metric_namespace=NAMESPACE,
            metric_subsystem=SUBSYSTEM,
        )
    )
    instrumentator.add(
        metrics.response_size(
            should_include_handler=True,
            should_include_method=True,
            should_include_status=True,
            metric_namespace=NAMESPACE,
            metric_subsystem=SUBSYSTEM,
        )
    )
    instrumentator.add(
        metrics.latency(
            should_include_handler=True,
            should_include_method=True,
            should_include_status=True,
            metric_namespace=NAMESPACE,
            metric_subsystem=SUBSYSTEM,
        )
    )
    instrumentator.add(
        metrics.requests(
            should_include_handler=True,
            should_include_method=True,
            should_include_status=True,
            metric_namespace=NAMESPACE,
            metric_subsystem=SUBSYSTEM,
        )
    )

    buckets = (*np.arange(0, 10.5, 0.5).tolist(), float("inf"))
    instrumentator.add(
        count_output(metric_namespace=NAMESPACE, metric_subsystem=SUBSYSTEM, buckets=buckets)
    )
    return instrumentator