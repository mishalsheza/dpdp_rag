"""Run the API: `python -m dpdp_rag.api` (host/port from configs/default.yaml)."""

from __future__ import annotations

import uvicorn

from dpdp_rag.config import load_system_config


def main() -> None:
    cfg = load_system_config()["api"]
    uvicorn.run(
        "dpdp_rag.api.app:create_app", factory=True, host=cfg["host"], port=int(cfg["port"])
    )


if __name__ == "__main__":
    main()
