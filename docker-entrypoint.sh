#!/bin/sh
set -e

case "$1" in
  serve)
    alembic upgrade head
    exec uvicorn citewise.main:app --host 0.0.0.0 --port 8000 --proxy-headers --forwarded-allow-ips='*'
    ;;
  worker)
    exec python -m citewise worker
    ;;
  migrate)
    exec alembic upgrade head
    ;;
  *)
    exec "$@"
    ;;
esac
