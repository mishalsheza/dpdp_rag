"""HTTP interface: POST /ask, GET /healthz, GET /version."""

from dpdp_rag.api.app import create_app

__all__ = ["create_app"]
