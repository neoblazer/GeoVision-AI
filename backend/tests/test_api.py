from __future__ import annotations

import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("fastapi"), "FastAPI is not installed")
class ApiTests(unittest.TestCase):
    def test_health_and_public_config(self) -> None:
        from fastapi.testclient import TestClient

        from geovision.core.config import Settings
        from geovision.main import create_app

        settings = Settings(detector_backend="mock", tracker_backend="botsort")
        client = TestClient(create_app(settings))

        health = client.get("/api/v1/health")
        config = client.get("/api/v1/config")

        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["status"], "ok")
        self.assertEqual(config.status_code, 200)
        self.assertEqual(config.json()["detector_backend"], "mock")
        self.assertEqual(config.json()["tracker_backend"], "botsort")


if __name__ == "__main__":
    unittest.main()

