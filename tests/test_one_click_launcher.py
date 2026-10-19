"""Launcher regressions: no Docker mutations, model calls, or browser opening."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import httpx

from app.services.readiness import readiness
from app.core.config import Settings

spec = importlib.util.spec_from_file_location("campus_launcher", Path(__file__).resolve().parents[1] / "scripts/launch.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def config(**kwargs):
    return Settings(_env_file=None, ai_provider="mock", knowledge_vector_enabled=False, **kwargs)


class OneClickTests(unittest.TestCase):
    def test_model_probe_retries_temporary_network_failures(self):
        operation = Mock(side_effect=[httpx.ConnectError("offline"), "ready"])
        with patch.object(launcher.time, "sleep"):
            self.assertEqual(launcher.model_probe("模型", operation), "ready")
        self.assertEqual(operation.call_count, 2)

    def test_model_probe_does_not_retry_unauthorized_requests(self):
        request = httpx.Request("GET", "https://example.com/models")
        operation = Mock(side_effect=httpx.HTTPStatusError("unauthorized", request=request,
            response=httpx.Response(401, request=request)))
        with self.assertRaises(httpx.HTTPStatusError):
            launcher.model_probe("模型", operation)
        self.assertEqual(operation.call_count, 1)

    def test_only_this_compose_ports_are_managed(self):
        current = config(database_url="mysql+pymysql://user:pass@127.0.0.1:13306/campus",
            redis_url="redis://127.0.0.1:16379/0")
        self.assertEqual(launcher.docker_services(current), ["mysql", "redis"])
        current.database_url = "mysql+pymysql://user:pass@127.0.0.1:3306/campus"
        current.redis_url = "redis://remote.school:16379/0"
        self.assertEqual(launcher.docker_services(current), [])

    def test_unrelated_web_app_is_not_mistaken_for_this_project(self):
        def handler(request):
            return httpx.Response(200, json={"status": "UP", "info": {"title": "Other project"}})
        with patch.object(launcher.httpx, "Client", return_value=httpx.Client(transport=httpx.MockTransport(handler))):
            self.assertFalse(launcher.app_ready("http://test"))

    def test_waits_for_readiness_not_just_listening_port(self):
        def handler(request):
            if request.url.path == "/openapi.json":
                return httpx.Response(200, json={"info": {"title": "校园有应 · 校园资讯与生活支持"}})
            return httpx.Response(503, json={"status": "DOWN"})
        with patch.object(launcher.httpx, "Client", return_value=httpx.Client(transport=httpx.MockTransport(handler))):
            self.assertFalse(launcher.app_ready("http://test"))

    def test_success_requires_frontend_and_ready_backend(self):
        def handler(request):
            if request.url.path == "/openapi.json":
                return httpx.Response(200, json={"info": {"title": "校园有应 · 校园资讯与生活支持"}})
            return httpx.Response(200, json={"status": "UP"})
        with patch.object(launcher.httpx, "Client", return_value=httpx.Client(transport=httpx.MockTransport(handler))):
            self.assertTrue(launcher.app_ready("http://test"))

    def test_server_exit_fails_without_waiting_for_timeout(self):
        with self.assertRaisesRegex(RuntimeError, "提前退出"):
            launcher.wait_for("测试后端", lambda: False, 300, SimpleNamespace(poll=lambda: 1))

    def test_dead_worker_prevents_false_ready(self):
        db = Mock()
        db.execute.return_value.scalar.return_value = 1
        cfg = config(tool_queue_enabled=True, memory_maintenance_queue_enabled=True)
        state = SimpleNamespace(tool_queue_worker=SimpleNamespace(dispatcher=SimpleNamespace(is_alive=lambda: False)),
            memory_maintenance_worker=SimpleNamespace(thread=SimpleNamespace(is_alive=lambda: True)))
        with patch("app.services.readiness.Redis") as redis:
            redis.from_url.return_value.ping.return_value = True
            result = readiness(db, cfg, state)
        self.assertEqual(result["status"], "DOWN")
        self.assertEqual(result["checks"]["toolQueue"], "DOWN")

    def test_disabled_workers_are_reported_explicitly(self):
        db = Mock()
        db.execute.return_value.scalar.return_value = 1
        cfg = config(tool_queue_enabled=False, memory_maintenance_queue_enabled=False)
        with patch("app.services.readiness.Redis") as redis:
            redis.from_url.return_value.ping.return_value = True
            result = readiness(db, cfg, SimpleNamespace())
        self.assertEqual(result["status"], "UP")
        self.assertEqual(result["checks"]["toolQueue"], "DISABLED")


if __name__ == "__main__":
    unittest.main()
