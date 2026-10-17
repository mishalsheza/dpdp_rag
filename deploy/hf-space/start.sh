#!/bin/sh
# Starts the API in the background, waits for /healthz, then runs the UI on the Space port.
set -e

dpdp-api &

python - <<'EOF'
import time
import urllib.request

for _ in range(120):
    try:
        urllib.request.urlopen("http://127.0.0.1:8000/healthz", timeout=2)
        print("API is up")
        break
    except Exception:
        time.sleep(1)
else:
    raise SystemExit("API did not start within 120 s")
EOF

exec streamlit run src/dpdp_rag/ui/app.py \
    --server.address=0.0.0.0 --server.port=7860 --server.headless=true \
    --server.enableCORS=false --server.enableXsrfProtection=false \
    --browser.gatherUsageStats=false
