from fastapi import FastAPI, Request, HTTPException, Response, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import joblib
import pandas as pd
import logging
import time
import json

from google.cloud.logging_v2.handlers import CloudLoggingHandler
from google.cloud import logging as gcp_logging

# --- OpenTelemetry Setup ---
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.exporter.cloud_monitoring import CloudMonitoringMetricsExporter
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

# Initialize Tracer
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)
span_processor = BatchSpanProcessor(CloudTraceSpanExporter())
trace.get_tracer_provider().add_span_processor(span_processor)

# Initialize Cloud Logging
client = gcp_logging.Client()
cloud_handler = CloudLoggingHandler(client)
cloud_handler.setLevel(logging.INFO)

logger = logging.getLogger("oppe2-fastapi-logger")
logger.setLevel(logging.INFO)
logger.addHandler(cloud_handler)

# Initialize Cloud Monitoring Metrics
exporter = CloudMonitoringMetricsExporter()
reader = PeriodicExportingMetricReader(exporter)
provider = MeterProvider(metric_readers=[reader])
metrics.set_meter_provider(provider)
meter = metrics.get_meter(__name__)

latency_metric = meter.create_histogram(
    name="oppe2_request_latency_ms",
    description="Request latency for oppe2 predictions",
    unit="ms"
)

# --- FastAPI App ---
app = FastAPI(title="OPPE2 API")
model = joblib.load("models/model.pkl")

class InputData(BaseModel):
    age: float
    gender: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float

app_state = {"is_ready": False, "is_alive": True}

@app.on_event("startup")
async def startup_event():
    time.sleep(2)
    app_state["is_ready"] = True

@app.get("/live_check", tags=["Probe"])
async def liveness_probe():
    if app_state["is_alive"]:
        return {"status": "alive"}
    return Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

@app.get("/ready_check", tags=["Probe"])
async def readiness_probe():
    if app_state["is_ready"]:
        return {"status": "ready"}
    return Response(status_code=status.HTTP_503_SERVICE_UNAVAILABLE)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = round((time.time() - start_time) * 1000, 2)
    response.headers["X-Process-Time-ms"] = str(duration)
    return response

@app.exception_handler(Exception)
async def exception_handler(request: Request, exc: Exception):
    span = trace.get_current_span()
    trace_id = format(span.get_span_context().trace_id, "032x")
    logger.exception(json.dumps({
        "event": "unhandled_exception",
        "trace_id": trace_id,
        "path": str(request.url),
        "error": str(exc)
    }))
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error", "trace_id": trace_id},
    )

@app.post("/predict/")
async def predict_species(data: InputData, request: Request):
    with tracer.start_as_current_span("model_inference") as span:
        start_time = time.time()
        trace_id = format(span.get_span_context().trace_id, "032x")
        try:
            input_df = pd.DataFrame([data.dict()])
            prediction_proba = model.predict_proba(input_df)
            probability = float(prediction_proba[0][1])
            is_heart_disease_present = "Yes" if probability > 0.5 else "No"
            latency = round((time.time() - start_time) * 1000, 2)
            latency_metric.record(latency)

            logger.info(json.dumps({
                "event": "prediction_success",
                "trace_id": trace_id,
                "input": data.dict(),
                "result": is_heart_disease_present,
                "latency_ms": latency,
                "status": "success"
            }))
            return {"predicted_class": is_heart_disease_present, "probability_heart_disease_present": probability}
        except Exception as e:
            logger.exception(json.dumps({
                "event": "prediction_error",
                "trace_id": trace_id,
                "error": str(e)
            }))
            raise HTTPException(status_code=500, detail="Prediction failed")
