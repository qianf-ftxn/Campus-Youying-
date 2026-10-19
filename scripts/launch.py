"""macOS one-click supervisor; uses the existing .env, never substitutes mock services."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import httpx
from redis import Redis
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from app.core.config import get_settings
from app.services.agent_models import AGENT_MODEL_ALIASES, AgentModelRegistry
from app.services.vector_store import ChromaKnowledgeStore


def say(message):
    print(message, flush=True)


def command(args, *, timeout=30, **kwargs):
    result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=timeout, **kwargs)
    if result.returncode:
        raise RuntimeError(f"{args[0]} 执行失败，请检查 {ROOT / 'target/launcher'} 中的日志。")
    return result.stdout


def wait_for(label, probe, timeout, child=None):
    started, last_notice = time.monotonic(), -10
    while time.monotonic() - started < timeout:
        if child is not None and child.poll() is not None:
            raise RuntimeError(f"{label} 提前退出，请查看启动日志。")
        try:
            if probe():
                return
        except Exception:
            pass
        elapsed = time.monotonic() - started
        if elapsed - last_notice >= 10:
            say(f"正在等待{label}就绪……（{int(elapsed)} 秒）")
            last_notice = elapsed
        time.sleep(1)
    raise RuntimeError(f"等待{label}超时（{timeout} 秒）。")


def port_in_use(port):
    with socket.socket() as sock:
        return sock.connect_ex(("127.0.0.1", port)) == 0


def app_ready(base_url):
    try:
        with httpx.Client(timeout=3, trust_env=False) as client:
            info = client.get(base_url + "/openapi.json")
            if info.status_code != 200 or info.json().get("info", {}).get("title") != "校园有应 · 校园资讯与生活支持":
                return False
            ready = client.get(base_url + "/actuator/ready")
            page = client.get(base_url + "/")
            return ready.status_code == 200 and ready.json().get("status") == "UP" and page.status_code == 200
    except (httpx.HTTPError, ValueError):
        return False


def docker_services(settings):
    db = make_url(settings.database_url)
    cache = urlsplit(settings.redis_url)
    local = {"127.0.0.1", "localhost", "::1"}
    services = []
    # 仅管理本项目 Compose 已发布的端口，不接管本机其他 MySQL/Redis。
    if db.get_backend_name() == "mysql" and db.host in local and (db.port or 3306) == 13306:
        services.append("mysql")
    if cache.hostname in local and (cache.port or 6379) == 16379:
        services.append("redis")
    return services


def start_docker(services, logfile):
    if not services:
        return
    if not shutil.which("docker"):
        raise RuntimeError("未找到 Docker Desktop，请先安装后重新双击。")
    def running():
        return subprocess.run(["docker", "info"], stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, timeout=5).returncode == 0
    try:
        available = running()
    except subprocess.TimeoutExpired:
        available = False
    if not available:
        say("正在启动 Docker Desktop……")
        command(["open", "-a", "Docker"])
        wait_for("Docker Desktop", running, 180)
    say("正在启动 " + "、".join(services) + "……")
    with logfile.open("a") as output:
        result = subprocess.run(["docker", "compose", "up", "-d", *services], cwd=ROOT,
            stdout=output, stderr=subprocess.STDOUT, timeout=180)
    if result.returncode:
        raise RuntimeError(f"数据库容器启动失败，详情：{logfile}")


def check_data_services(settings):
    engine = create_engine(settings.database_url, connect_args={"connect_timeout": 3}
        if make_url(settings.database_url).get_backend_name() == "mysql" else {})
    try:
        def database_ready():
            with engine.connect() as connection:
                return connection.execute(text("SELECT 1")).scalar() == 1
        wait_for("数据库", database_ready, 120)
    finally:
        engine.dispose()
    cache = Redis.from_url(settings.redis_url, socket_connect_timeout=3, socket_timeout=3)
    try:
        wait_for("Redis", cache.ping, 90)
    finally:
        cache.close()


def model_probe(label, operation):
    """启动时重试连接/超时、限流和供应商临时故障，不重试无效鉴权。"""
    for attempt in range(3):
        try:
            return operation()
        except (httpx.RequestError, httpx.HTTPStatusError) as exc:
            if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code not in {429, 500, 502, 503, 504}:
                raise
            if attempt == 2:
                raise RuntimeError(f"{label}连接暂时不可用，已重试 3 次，请稍后重新双击启动。") from exc
            say(f"{label}暂未就绪，正在重试（{attempt + 2}/3）……")
            time.sleep(2)


def check_models(settings, children, logfile):
    registry = AgentModelRegistry(settings)
    profiles = [registry.profile_for(name) for name in AGENT_MODEL_ALIASES]
    if any(profile.provider == "openai" for profile in profiles):
        if not settings.openai_api_key:
            raise RuntimeError("云端模型缺少 OPENAI_API_KEY，请在项目 .env 配置后重试。")
        say("正在检查云端模型连接……")
        def cloud_models():
            response = httpx.get(settings.openai_base_url.rstrip("/") + "/models",
                headers={"Authorization": "Bearer " + settings.openai_api_key}, timeout=20)
            if response.status_code in {429, 500, 502, 503, 504}:
                response.raise_for_status()
            return response
        response = model_probe("云端模型", cloud_models)
        if response.status_code in (401, 403):
            raise RuntimeError("云端模型鉴权失败，请检查 .env 中的 API Key 与接口权限。")
        if response.status_code not in (404, 405):
            response.raise_for_status()
    ollama_profiles = [profile for profile in profiles if profile.provider == "ollama"]
    if ollama_profiles:
        url = settings.ollama_base_url.rstrip("/") + "/api/tags"
        def ollama_ready():
            return httpx.get(url, timeout=3).status_code == 200
        try:
            running = ollama_ready()
        except httpx.HTTPError:
            running = False
        if not running:
            if urlsplit(url).hostname not in {"localhost", "127.0.0.1", "::1"}:
                raise RuntimeError("远程 Ollama 未就绪，请检查 OLLAMA_BASE_URL。")
            address = urlsplit(settings.ollama_base_url)
            env = {**os.environ, "OLLAMA_BASE_URL": settings.ollama_base_url,
                "OLLAMA_HOST": f"{address.hostname}:{address.port or 11434}"}
            with logfile.open("a") as output:
                children.append(subprocess.Popen(["bash", "scripts/start-ollama.sh"], cwd=ROOT,
                    env=env, stdout=output, stderr=subprocess.STDOUT))
            wait_for("Ollama", ollama_ready, 60, children[-1])
        available = {row["name"] for row in httpx.get(url, timeout=5).json().get("models", [])}
        missing = {p.model for p in ollama_profiles if p.model not in available and p.model + ":latest" not in available}
        if missing:
            raise RuntimeError("Ollama 尚未安装所配置的模型：" + "、".join(sorted(missing)))
    if settings.knowledge_vector_enabled:
        say("正在检查本地 Chroma 和 Embedding 接口……")
        store = ChromaKnowledgeStore(settings)
        if not store.can_embed:
            raise RuntimeError("向量检索尚未就绪：" + store.error)
        model_probe("Embedding 接口", lambda: store.embed_texts(["校园资讯服务启动检查"]))


def open_frontend(url, no_browser):
    if not no_browser:
        command(["open", url])


def main(argv=None):
    parser = argparse.ArgumentParser(description="校园有应一键启动")
    parser.add_argument("--no-browser", action="store_true", help="验证模式：不打开浏览器")
    parser.add_argument("--port", type=int, default=int(os.environ.get("SERVER_PORT", "7070")))
    parser.add_argument("--startup-timeout", type=int, default=900)
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("端口应在 1—65535 之间")
    os.chdir(ROOT)
    runtime = ROOT / "target/launcher"
    runtime.mkdir(parents=True, exist_ok=True)
    logfile = runtime / "startup.log"
    url = f"http://127.0.0.1:{args.port}"
    children = []
    def stop_signal(signum, frame):
        raise KeyboardInterrupt
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, stop_signal)
    try:
        with (runtime / f"launch-{args.port}.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                say("已有启动窗口，等待服务就绪后打开页面。")
                wait_for("已有服务", lambda: app_ready(url), args.startup_timeout)
                open_frontend(url, args.no_browser)
                return 0
            if app_ready(url):
                say(f"校园有应已在运行：{url}")
                open_frontend(url, args.no_browser)
                return 0
            if port_in_use(args.port):
                raise RuntimeError(f"端口 {args.port} 已被其他进程占用，且不是就绪的校园有应服务。请释放端口后重试。")
            settings = get_settings()
            say("校园有应一键启动：保留现有 .env 与业务数据。")
            start_docker(docker_services(settings), logfile)
            check_data_services(settings)
            say("MySQL / 数据库与 Redis 已就绪。")
            check_models(settings, children, logfile)
            say("正在启动后端、前端静态服务、工具队列与记忆维护任务……")
            with logfile.open("a") as output:
                output.write(f"\n=== launch {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
                output.flush()
                server = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--host",
                    "127.0.0.1", "--port", str(args.port)], cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
            children.append(server)
            wait_for("校园有应（首次迁移和知识索引可能较慢）", lambda: app_ready(url), args.startup_timeout, server)
            say(f"\n全部就绪：{url}\n学生账号：student / student123；管理账号：admin / admin123（已有账号沿用原密码）")
            say(f"启动日志：{logfile}\n请保留本窗口；Ctrl+C 或关闭窗口会停止本次后端。MySQL / Redis 容器保留运行。")
            open_frontend(url, args.no_browser)
            code = server.wait()
            if code:
                raise RuntimeError(f"后端退出（{code}），请查看 {logfile}")
            return 0
    except KeyboardInterrupt:
        say("\n正在停止本次启动的服务……")
        return 0
    except Exception as exc:
        # 不打印 HTTP 请求头或 Settings，避免将 API Key 写入终端。
        detail = str(exc) if isinstance(exc, RuntimeError) else '请检查依赖连接与启动日志。'
        if isinstance(exc, httpx.HTTPStatusError):
            detail = f"模型接口返回 HTTP {exc.response.status_code}，请检查模型配置或接口权限。"
        with logfile.open("a") as output:
            output.write(f"\n启动探测失败 {type(exc).__name__}：{detail}\n")
        say(f"\n启动失败：{type(exc).__name__}。{detail}")
        say(f"详细日志：{logfile}")
        return 1
    finally:
        for child in reversed(children):
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()


if __name__ == "__main__":
    raise SystemExit(main())
