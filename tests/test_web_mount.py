"""Static web serving.

Regression guard: an invalid return annotation on the SPA fallback route used to
make `create_app()` raise at import time whenever `web/dist` existed — i.e. the
server could not start after `npm run build`. These tests cover both paths.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from slmforge.main import create_app


def test_api_only_when_no_build_is_present(tmp_path: Path) -> None:
    app = create_app(web_dist=tmp_path / "missing")
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        body = response.json()
        assert body["service"] == "slmforge"
        assert "not built yet" in body["message"]

        # The API is unaffected by the absent frontend.
        assert client.get("/api/health").json()["status"] == "ok"


def test_serves_built_spa_with_client_side_routing(tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><div id=root></div>", encoding="utf-8")
    (dist / "assets" / "app.js").write_text("console.log('slmforge')", encoding="utf-8")
    (dist / "favicon.svg").write_text("<svg/>", encoding="utf-8")

    app = create_app(web_dist=dist)
    with TestClient(app) as client:
        index = client.get("/")
        assert index.status_code == 200
        assert "root" in index.text

        # Hashed asset bundles are served from the mounted static directory.
        asset = client.get("/assets/app.js")
        assert asset.status_code == 200
        assert "slmforge" in asset.text

        # A real file at the root is served directly...
        assert client.get("/favicon.svg").status_code == 200

        # ...and an unknown deep route falls back to the SPA shell, not a 404.
        deep = client.get("/projects/abc123")
        assert deep.status_code == 200
        assert "root" in deep.text

        # The API still wins over the catch-all route.
        assert client.get("/api/health").json()["status"] == "ok"


def test_spa_fallback_rejects_path_traversal(tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<!doctype html><div id=root></div>", encoding="utf-8")
    secret = tmp_path / "secret.txt"
    secret.write_text("do not serve me", encoding="utf-8")

    app = create_app(web_dist=dist)
    with TestClient(app) as client:
        response = client.get("/../secret.txt")
        assert response.status_code == 200
        assert "root" in response.text
        assert "do not serve me" not in response.text
