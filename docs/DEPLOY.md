# Deploying to Hugging Face Spaces

The live demo runs as a Docker Space on the free CPU tier. One container runs both the API
and the Streamlit UI:

- `deploy/hf-space/Dockerfile` clones this repository, installs it and bakes in the
  embedding model.
- `deploy/hf-space/start.sh` starts `dpdp-api` on `127.0.0.1:8000`, waits for `/healthz`,
  then runs the UI on port 7860, the only port a Space exposes.
- `configs/hf.yaml` (selected with `DPDP_CONFIG_OVERRIDES=hf.yaml`) uses in-memory Qdrant
  with `auto_index: true`, so no Qdrant service is needed.

## One-time setup

1. On huggingface.co, create a new Space: **SDK: Docker**, template **Blank**, hardware
   **CPU basic (free)**.
2. In the Space, open **Settings → Variables and secrets** and add a **secret** named
   `GROQ_API_KEY`. Optional: `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`.
3. Upload the three files from `deploy/hf-space/` (`Dockerfile`, `start.sh`, `README.md`)
   to the root of the Space, through **Files → Add file → Upload files** or with git:

   ```bash
   git clone https://huggingface.co/spaces/<user>/<space>
   cp deploy/hf-space/* <space>/
   cd <space> && git add . && git commit -m "Deploy DPDP RAG" && git push
   ```

4. The Space builds (about 5 minutes) and then serves the UI at
   `https://<user>-<space>.hf.space`. Put that URL in the README.

## Things to know

- **First question is slow.** The in-memory index is built on the first question (364
  chunks, about 45 s in a local test). Later questions are fast. Ask one question
  before a demo.
- **Sleeping.** A free Space sleeps after 48 hours without visitors. Open the URL a few
  minutes before a demo.
- **Updating.** The image clones `main` at build time, and Docker caches that step. After merging changes, use
  **Settings → Factory rebuild** in the Space.
- **Data does not persist.** `data/metrics.db` lives inside the container, so the metrics
  dashboard resets when the Space restarts.

## Testing the image locally

Build from a branch and run it with the stub LLM, which needs no API key:

```bash
docker build -t dpdp-hf --build-arg REF=main deploy/hf-space
docker run --rm -p 7860:7860 -e DPDP_CONFIG_OVERRIDES=hf.yaml,loadtest.yaml dpdp-hf
```

Then open http://localhost:7860. For real answers, pass `-e GROQ_API_KEY=...` and drop
`loadtest.yaml`.
