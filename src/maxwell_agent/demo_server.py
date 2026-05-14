from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from threading import Lock, Thread
from typing import Any
from urllib.parse import parse_qs

from .agent import MaxwellAgent
from .demo import DemoBundle, execute_demo


def _now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@dataclass(slots=True)
class DemoJobState:
    state: str = "idle"
    progress: int = 0
    stage: str = "等待任务"
    requirement: str = ""
    started_at: str | None = None
    finished_at: str | None = None
    error_message: str = ""
    timeline: list[str] = field(default_factory=list)
    bundle: DemoBundle | None = None

    def push(self, message: str) -> None:
        entry = f"{_now_text()} {message}"
        if not self.timeline or self.timeline[-1] != entry:
            self.timeline.append(entry)


def serve_demo(agent: MaxwellAgent, host: str, port: int) -> None:
    class DemoHTTPServer(ThreadingHTTPServer):
        def __init__(self, server_address, handler_class):
            super().__init__(server_address, handler_class)
            self.agent = agent
            self.run_lock = Lock()
            self.state_lock = Lock()
            self.job = DemoJobState()
            self.last_bundle: DemoBundle | None = None

        def start_job(self, requirement: str) -> None:
            with self.state_lock:
                self.job = DemoJobState(
                    state="running",
                    progress=3,
                    stage="任务已提交，等待启动",
                    requirement=requirement,
                    started_at=_now_text(),
                )
                self.job.push("任务已提交")

        def update_progress(self, percent: int, message: str) -> None:
            with self.state_lock:
                if self.job.state != "running":
                    return
                self.job.progress = max(0, min(100, percent))
                self.job.stage = message
                self.job.push(message)

        def finish_job(self, bundle: DemoBundle) -> None:
            with self.state_lock:
                self.job.progress = 100
                self.job.state = bundle.status
                self.job.stage = bundle.message
                self.job.finished_at = _now_text()
                self.job.bundle = bundle
                self.job.error_message = "" if bundle.status == "completed" else bundle.message
                self.job.push(bundle.message)
                self.last_bundle = bundle
            self._release_run_lock()

        def fail_job(self, requirement: str, error_message: str) -> None:
            with self.state_lock:
                self.job.progress = 100
                self.job.state = "failed"
                self.job.stage = "运行失败"
                self.job.requirement = requirement
                self.job.finished_at = _now_text()
                self.job.error_message = error_message
                self.job.push(f"运行失败: {error_message}")
            self._release_run_lock()

        def _release_run_lock(self) -> None:
            try:
                self.run_lock.release()
            except RuntimeError:
                pass

        def status_payload(self) -> dict[str, Any]:
            with self.state_lock:
                bundle = self.job.bundle
                if not bundle and self.job.state == "idle":
                    bundle = self.last_bundle
                return {
                    "state": self.job.state,
                    "progress": self.job.progress,
                    "stage": self.job.stage,
                    "requirement": self.job.requirement,
                    "started_at": self.job.started_at,
                    "finished_at": self.job.finished_at,
                    "error_message": self.job.error_message,
                    "timeline": list(self.job.timeline[-10:]),
                    "bundle_html": bundle.to_html_document(page_title="Maxwell 智能体运行结果") if bundle else "",
                    "bundle_status": bundle.status if bundle else "",
                    "bundle_message": bundle.message if bundle else "",
                    "run_directory": str(bundle.run_directory) if bundle else "",
                    "project_file": str(bundle.project_file) if bundle and bundle.project_file else "",
                    "summary_html_path": str(bundle.summary_html_path) if bundle and bundle.summary_html_path else "",
                    "case_report_html_path": str(bundle.case_report_html_path)
                    if bundle and bundle.case_report_html_path
                    else "",
                    "case_report_markdown_path": str(bundle.case_report_markdown_path)
                    if bundle and bundle.case_report_markdown_path
                    else "",
                }

        def run_job_async(self, requirement: str) -> None:
            def worker() -> None:
                try:
                    bundle = execute_demo(self.agent, requirement, progress_callback=self.update_progress)
                except Exception as exc:  # pragma: no cover
                    self.fail_job(requirement, str(exc))
                    return
                self.finish_job(bundle)

            Thread(target=worker, name="maxwell-demo-job", daemon=True).start()

    class DemoRequestHandler(BaseHTTPRequestHandler):
        server: DemoHTTPServer

        def do_GET(self) -> None:  # noqa: N802
            normalized = self.path.rstrip("/") or "/"
            if normalized == "/health":
                self._write_text("ok", status=HTTPStatus.OK)
                return
            if normalized == "/status":
                self._write_json(self.server.status_payload(), status=HTTPStatus.OK)
                return
            self._write_html(_render_page())

        def do_POST(self) -> None:  # noqa: N802
            if self.path.rstrip("/") != "/run":
                self._write_text("Not Found", status=HTTPStatus.NOT_FOUND)
                return

            content_length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(content_length).decode("utf-8", errors="ignore")
            requirement = parse_qs(body).get("requirement", [""])[0].strip()
            if not requirement:
                self._write_json({"accepted": False, "error": "请输入需求后再运行。"}, status=HTTPStatus.BAD_REQUEST)
                return

            if not self.server.run_lock.acquire(blocking=False):
                payload = self.server.status_payload()
                payload["accepted"] = False
                payload["error"] = "当前已有一个 Maxwell 任务正在运行，请等待本次计算完成后再提交。"
                self._write_json(payload, status=HTTPStatus.CONFLICT)
                return

            self.server.start_job(requirement)
            self.server.run_job_async(requirement)
            payload = self.server.status_payload()
            payload["accepted"] = True
            self._write_json(payload, status=HTTPStatus.ACCEPTED)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            return

        def _write_html(self, payload: str, status: HTTPStatus = HTTPStatus.OK) -> None:
            encoded = payload.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def _write_text(self, payload: str, status: HTTPStatus = HTTPStatus.OK) -> None:
            encoded = payload.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def _write_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
            encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    server = DemoHTTPServer((host, port), DemoRequestHandler)
    print(f"Maxwell demo page is running at http://{host}:{port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _render_page() -> str:
    return """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Maxwell 智能体演示</title>
  <style>
    :root {
      --paper: #fffdf8;
      --text: #152018;
      --muted: #5c665f;
      --line: #d8cfbc;
      --accent: #1f5e4a;
      --warn: #b55f30;
      --track: rgba(31, 94, 74, 0.12);
      --shadow: 0 18px 50px rgba(31, 39, 32, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Microsoft YaHei UI", "PingFang SC", "Segoe UI", sans-serif;
      color: var(--text);
      background: linear-gradient(180deg, #f7f0e2 0%, #efe5d4 100%);
      min-height: 100vh;
    }
    .shell { width: min(1180px, calc(100vw - 32px)); margin: 24px auto 40px; }
    .hero {
      padding: 30px 32px;
      border-radius: 14px;
      background: linear-gradient(135deg, rgba(23, 58, 47, 0.98), rgba(31, 94, 74, 0.95));
      color: #f8f3ea;
      box-shadow: var(--shadow);
    }
    h1, h2, p { margin: 0; }
    h1 { font-size: clamp(28px, 4vw, 42px); font-weight: 800; }
    .hero p { margin-top: 12px; max-width: 920px; line-height: 1.72; color: rgba(248, 243, 234, 0.92); }
    .grid { margin-top: 18px; display: grid; grid-template-columns: minmax(320px, 430px) minmax(0, 1fr); gap: 18px; align-items: start; }
    .card { background: var(--paper); border: 1px solid var(--line); border-radius: 10px; padding: 22px; box-shadow: var(--shadow); }
    .card h2 { color: var(--accent); font-size: 18px; margin-bottom: 12px; }
    .field-label { display: block; font-size: 13px; color: var(--muted); margin-bottom: 8px; line-height: 1.6; }
    textarea {
      width: 100%;
      min-height: 220px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      resize: vertical;
      font: inherit;
      line-height: 1.7;
      background: #fffdf9;
    }
    button {
      margin-top: 14px;
      width: 100%;
      border: none;
      border-radius: 8px;
      padding: 14px 16px;
      font: inherit;
      font-weight: 700;
      color: #fff9f0;
      background: linear-gradient(135deg, var(--accent), #194738);
      cursor: pointer;
    }
    button:hover { filter: brightness(1.03); }
    button[disabled] { cursor: not-allowed; opacity: 0.72; filter: grayscale(0.12); }
    .hint { margin-top: 12px; color: var(--muted); line-height: 1.65; font-size: 13px; }
    .status-panel { margin-top: 18px; padding: 16px; border-radius: 10px; background: rgba(31, 94, 74, 0.08); border: 1px solid rgba(31, 94, 74, 0.14); }
    .status-line { display: flex; justify-content: space-between; gap: 12px; align-items: center; font-size: 14px; font-weight: 700; color: var(--accent); }
    .progress-track { margin-top: 12px; width: 100%; height: 12px; border-radius: 999px; background: var(--track); overflow: hidden; }
    .progress-bar { width: 0%; height: 100%; border-radius: 999px; background: linear-gradient(90deg, var(--accent), #36a17b); transition: width 0.35s ease; }
    .meta { margin-top: 10px; color: var(--muted); font-size: 13px; line-height: 1.6; word-break: break-word; }
    .timeline { margin-top: 12px; padding-left: 18px; color: var(--muted); line-height: 1.7; font-size: 13px; }
    .timeline li:last-child { color: var(--text); font-weight: 600; }
    .error-box { margin-top: 14px; border-radius: 8px; padding: 12px 14px; background: rgba(181, 95, 48, 0.12); color: #8b4323; line-height: 1.6; font-weight: 600; display: none; }
    .result-head { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; margin-bottom: 12px; }
    .pill { display: inline-flex; border-radius: 999px; padding: 7px 12px; font-weight: 700; font-size: 13px; background: rgba(31, 94, 74, 0.12); color: var(--accent); }
    .pill.warn { background: rgba(181, 95, 48, 0.12); color: var(--warn); }
    .result-frame { border: 1px solid var(--line); border-radius: 10px; overflow: hidden; background: #fff; min-height: 720px; }
    iframe { border: none; width: 100%; min-height: 720px; background: #fff; }
    .placeholder { border: 1px dashed var(--line); border-radius: 10px; min-height: 720px; padding: 24px; display: flex; align-items: center; justify-content: center; color: var(--muted); line-height: 1.8; background: rgba(255, 253, 248, 0.72); text-align: center; }
    @media (max-width: 920px) { .grid { grid-template-columns: 1fr; } .result-frame, iframe, .placeholder { min-height: 580px; } }
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <h1>Maxwell 智能体演示</h1>
      <p>输入一句工程需求，系统会调用云端大模型整理规格，本地自动执行 Maxwell 2D，并把每轮迭代、约束满足情况、几何识别和单案例交付报告展示出来。</p>
    </section>

    <div class="grid">
      <section class="card">
        <h2>输入需求</h2>
        <form id="run-form">
          <label class="field-label" for="requirement">示例：做一个24V直流电磁铁，气隙2mm，电流不超过2A，尽量提高吸力，外形不要太大。</label>
          <textarea id="requirement" name="requirement" placeholder="请输入要演示的 Maxwell 需求..."></textarea>
          <button id="run-button" type="submit">开始运行</button>
        </form>
        <div id="error-box" class="error-box"></div>
        <div class="hint">
          提交后页面会立即返回，进度条会轮询显示当前阶段。运行期间不要重复提交，结果文件会保存到项目 workspace 目录。
        </div>

        <div class="status-panel">
          <div class="status-line">
            <span id="status-stage">等待任务</span>
            <span id="status-percent">0%</span>
          </div>
          <div class="progress-track"><div id="progress-bar" class="progress-bar"></div></div>
          <div id="status-meta" class="meta">当前没有运行中的任务。</div>
          <ul id="timeline" class="timeline"></ul>
        </div>
      </section>

      <section class="card">
        <h2>运行结果</h2>
        <div class="result-head">
          <span id="result-pill" class="pill">等待结果</span>
          <span id="result-meta" class="meta"></span>
        </div>
        <div class="result-frame">
          <iframe id="result-frame" title="demo-result"></iframe>
          <div id="result-placeholder" class="placeholder">还没有结果。提交任务后，右侧会自动更新为完整结果页面。</div>
        </div>
      </section>
    </div>
  </div>

  <script>
    const form = document.getElementById("run-form");
    const requirementField = document.getElementById("requirement");
    const runButton = document.getElementById("run-button");
    const errorBox = document.getElementById("error-box");
    const stageEl = document.getElementById("status-stage");
    const percentEl = document.getElementById("status-percent");
    const metaEl = document.getElementById("status-meta");
    const progressBar = document.getElementById("progress-bar");
    const timelineEl = document.getElementById("timeline");
    const pillEl = document.getElementById("result-pill");
    const resultMetaEl = document.getElementById("result-meta");
    const resultFrame = document.getElementById("result-frame");
    const resultPlaceholder = document.getElementById("result-placeholder");
    let pollTimer = null;

    function setBusy(busy) {
      requirementField.readOnly = busy;
      runButton.disabled = busy;
      runButton.textContent = busy ? "运行中，请等待..." : "开始运行";
    }
    function showError(message) {
      errorBox.style.display = message ? "block" : "none";
      errorBox.textContent = message || "";
    }
    function renderTimeline(items) {
      timelineEl.innerHTML = "";
      for (const item of items || []) {
        const li = document.createElement("li");
        li.textContent = item;
        timelineEl.appendChild(li);
      }
    }
    function renderResult(payload) {
      if (payload.bundle_html) {
        resultFrame.srcdoc = payload.bundle_html;
        resultFrame.style.display = "block";
        resultPlaceholder.style.display = "none";
      } else {
        resultFrame.removeAttribute("srcdoc");
        resultFrame.style.display = "none";
        resultPlaceholder.style.display = "flex";
      }
      const state = payload.state || "idle";
      pillEl.className = state !== "completed" && state !== "idle" ? "pill warn" : "pill";
      pillEl.textContent = state === "completed" ? "已完成" :
        state === "running" ? "运行中" :
        state === "failed" ? "失败" :
        state === "blocked" ? "阻塞" : "等待结果";
      const parts = [];
      if (payload.run_directory) parts.push("运行目录: " + payload.run_directory);
      if (payload.case_report_html_path) parts.push("单案例报告: " + payload.case_report_html_path);
      if (payload.summary_html_path) parts.push("摘要: " + payload.summary_html_path);
      resultMetaEl.textContent = parts.join(" | ");
    }
    function renderStatus(payload) {
      const progress = Number(payload.progress || 0);
      stageEl.textContent = payload.stage || "等待任务";
      percentEl.textContent = progress + "%";
      progressBar.style.width = progress + "%";
      const metaParts = [];
      if (payload.requirement) metaParts.push("当前需求: " + payload.requirement);
      if (payload.started_at) metaParts.push("开始时间: " + payload.started_at);
      if (payload.finished_at) metaParts.push("结束时间: " + payload.finished_at);
      metaEl.textContent = metaParts.join(" | ") || "当前没有运行中的任务。";
      renderTimeline(payload.timeline || []);
      renderResult(payload);
      const running = payload.state === "running";
      setBusy(running);
      showError(!running ? payload.error_message || "" : "");
      if (!running && pollTimer) {
        clearInterval(pollTimer);
        pollTimer = null;
      }
    }
    async function fetchStatus() {
      const response = await fetch("/status", { cache: "no-store" });
      const payload = await response.json();
      renderStatus(payload);
      return payload;
    }
    function startPolling() {
      if (pollTimer) return;
      pollTimer = setInterval(() => {
        fetchStatus().catch((error) => showError("状态轮询失败: " + error.message));
      }, 1000);
    }
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const requirement = requirementField.value.trim();
      if (!requirement) {
        showError("请输入需求后再运行。");
        return;
      }
      showError("");
      setBusy(true);
      const body = new URLSearchParams();
      body.set("requirement", requirement);
      try {
        const response = await fetch("/run", {
          method: "POST",
          headers: { "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8" },
          body: body.toString(),
        });
        const payload = await response.json();
        renderStatus(payload);
        if (!response.ok && payload.error) showError(payload.error);
        if (response.status === 202) startPolling();
        else setBusy(false);
      } catch (error) {
        setBusy(false);
        showError("启动任务失败: " + error.message);
      }
    });
    fetchStatus().catch((error) => showError("初始化状态失败: " + error.message));
  </script>
</body>
</html>"""
