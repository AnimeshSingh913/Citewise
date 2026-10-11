from prometheus_client import Counter, Gauge, Histogram

INGESTION_JOBS = Counter("citewise_ingestion_jobs_total", "Ingestion jobs finished", ["outcome"])
INGESTION_SECONDS = Histogram(
    "citewise_ingestion_seconds", "Time to ingest one document", buckets=(1, 2, 5, 10, 30, 60, 120, 300, 600)
)
QUEUE_DEPTH = Gauge("citewise_queue_depth", "Jobs waiting to be processed")
RETRIEVAL_SECONDS = Histogram("citewise_retrieval_seconds", "Hybrid retrieval latency")
GENERATION_SECONDS = Histogram(
    "citewise_generation_seconds", "Answer generation latency", buckets=(0.5, 1, 2, 5, 10, 20, 40, 80, 160)
)
LLM_ERRORS = Counter("citewise_llm_errors_total", "Claude API errors", ["kind"])
LLM_FALLBACKS = Counter("citewise_llm_fallbacks_total", "Answers served by a fallback model")
