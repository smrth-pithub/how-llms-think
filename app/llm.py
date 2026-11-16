"""Runs llama.cpp's llama-server in the background and asks it for next-token probabilities.

Everything is offline: the model files are in models/, the server binary in tools/llama/.
"""
import asyncio
import os
import subprocess
import time

import httpx

from . import config as C

LOG_DIR = C.ROOT / "logs"


class LlamaManager:
    def __init__(self):
        self.proc: subprocess.Popen | None = None
        self.model: str | None = None
        self.state = "stopped"          # stopped | loading | ready | error
        self.error: str | None = None
        self._lock = asyncio.Lock()
        self.client = httpx.AsyncClient(base_url=f"http://127.0.0.1:{C.LLAMA_PORT}", timeout=30)

    def status(self):
        return dict(model=self.model, state=self.state, error=self.error,
                    models=[dict(name=n, available=(C.MODELS_DIR / f).exists()) for n, f in C.LLM_MODELS.items()])

    async def ensure(self, name: str | None = None):
        name = name or self.model or C.LLM_DEFAULT
        async with self._lock:
            if self.model == name and self.state == "ready" and self.proc and self.proc.poll() is None:
                return
            await self._start(name)

    async def _start(self, name: str):
        self._kill()
        path = C.MODELS_DIR / C.LLM_MODELS.get(name, "")
        exe = C.LLAMA_DIR / ("llama-server.exe" if os.name == "nt" else "llama-server")
        self.model, self.state, self.error = name, "loading", None
        if name not in C.LLM_MODELS or not path.exists():
            self.state, self.error = "error", f"model file missing: {path.name}"
            return
        if not exe.exists():
            self.state, self.error = "error", f"llama-server not found in {C.LLAMA_DIR}"
            return
        await self._clear_leftover()
        LOG_DIR.mkdir(exist_ok=True)
        log = open(LOG_DIR / "llama-server.log", "w", encoding="utf-8", errors="replace")
        # -ngl 99: every layer on the GPU.  -lm none: don't memory-map the file
        # (halves the RAM llama-server holds: ~0.75 GB instead of ~1.8 GB)
        cmd = [str(exe), "-m", str(path), "-ngl", "99", "-c", "4096", "-np", "1", "-lm", "none",
               "--host", "127.0.0.1", "--port", str(C.LLAMA_PORT)]
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        self.proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, creationflags=flags,
                                     cwd=str(C.LLAMA_DIR))
        t0 = time.time()
        while time.time() - t0 < 90:
            if self.proc.poll() is not None:
                self.state, self.error = "error", "llama-server exited, see logs/llama-server.log"
                return
            try:
                r = await self.client.get("/health", timeout=2)
                if r.status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.3)
        else:
            self.state, self.error = "error", "llama-server took too long to start"
            return
        try:  # the first request is slow (GPU warm-up), so do it now, not on stage
            await self.next_tokens("Hello, my name is", k=20)
            await self.next_tokens("The cat sat on the", k=20)
        except Exception as e:
            self.state, self.error = "error", f"warm-up failed: {e}"
            return
        self.state = "ready"
        print(f"[llm] {name} ready in {time.time() - t0:.1f}s", flush=True)

    async def next_tokens(self, text: str, k: int = 100):
        """The model's top-k guesses for the next token, as raw log-probabilities.
        Temperature is applied in the browser, so the slider never re-runs the model."""
        t0 = time.perf_counter()
        r = await self.client.post("/completion", json={
            "prompt": text, "n_predict": 1, "n_probs": k, "temperature": 0,
            "cache_prompt": True, "return_tokens": False,
        })
        r.raise_for_status()
        d = r.json()
        probs = d.get("completion_probabilities") or []
        top = probs[0]["top_logprobs"] if probs else []
        ms = (time.perf_counter() - t0) * 1000
        return dict(tokens=[dict(id=t["id"], token=t["token"], bytes=t.get("bytes"), logprob=t["logprob"])
                            for t in top], ms=round(ms, 1), model=self.model)

    async def tokenize(self, text: str):
        r = await self.client.post("/tokenize", json={"content": text, "with_pieces": True})
        r.raise_for_status()
        return r.json().get("tokens", [])

    async def _clear_leftover(self):
        """If the app was closed with the window's X, its llama-server can outlive it and
        keep the port busy. Stop that one before starting a fresh server."""
        try:
            r = await self.client.get("/health", timeout=1)
        except httpx.HTTPError:
            return
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"], capture_output=True,
                           creationflags=subprocess.CREATE_NO_WINDOW)
        await asyncio.sleep(1.0)

    def _kill(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None

    def stop(self):
        self._kill()
        self.state = "stopped"
