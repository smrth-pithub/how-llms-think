"""One local web app for the whole session.

    python -m app.server                 then open http://localhost:8000
    python -m app.server --camera 1      if the external webcam isn't camera 0

Pages: /gesture (live demo), /record (collect + train), /tokenizer, /llm (next-token viewer)
"""
import argparse
import asyncio
import json
import subprocess
import sys
import time
import webbrowser
from contextlib import asynccontextmanager

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config as C
from .gesture_engine import GestureEngine
from .llm import LlamaManager

ARGS = argparse.Namespace(camera=0, width=1280, height=720, pose="full", no_browser=True, port=C.APP_PORT,
                          keep_llm=False, replay=None)
engine: GestureEngine | None = None
llm = LlamaManager()


@asynccontextmanager
async def lifespan(app):
    global engine
    engine = GestureEngine(ARGS.camera, ARGS.width, ARGS.height, ARGS.pose, replay=ARGS.replay)
    yield
    llm.stop()


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=C.WEB_DIR), name="static")

PAGES = {"/": "index.html", "/gesture": "gesture.html", "/record": "record.html",
         "/tokenizer": "tokenizer.html", "/llm": "llm.html"}


def _page(name):
    async def handler():
        return FileResponse(C.WEB_DIR / name, headers={"Cache-Control": "no-store"})
    return handler


for route, fname in PAGES.items():
    app.add_api_route(route, _page(fname), methods=["GET"], include_in_schema=False)


# ---------------- gesture ----------------
@app.websocket("/ws/gesture")
async def ws_gesture(ws: WebSocket):
    await ws.accept()
    engine.acquire()
    if not ARGS.keep_llm and llm.state != "stopped":
        llm.stop()          # one demo at a time: free the LLM's memory while the camera runs
    last_id, last_sent = -1, 0.0
    try:
        while True:
            state, jpeg = engine.snapshot()
            now = time.time()
            if state["frame_id"] != last_id or now - last_sent > 0.5:
                last_id, last_sent = state["frame_id"], now
                await ws.send_text(json.dumps(state))
                if jpeg:
                    await ws.send_bytes(jpeg)
            await asyncio.sleep(0.008)
    except (WebSocketDisconnect, RuntimeError, ConnectionError):
        pass
    finally:
        engine.release()


@app.get("/api/gesture/info")
async def gesture_info():
    return dict(gestures=[dict(key=k, label=l, hint=C.GESTURE_HINTS.get(k, "")) for k, l in C.GESTURES],
                model=engine.model_info(), window_seconds=C.WINDOW_SECONDS, n_points=len(C.POINTS))


class RecordReq(BaseModel):
    person: str
    gestures: list[str] = []
    seconds: float = 10
    countdown: float = 3


@app.post("/api/record/start")
async def record_start(req: RecordReq):
    engine.start_recording(req.person, req.gestures, req.seconds, req.countdown)
    return {"ok": True}


@app.post("/api/record/stop")
async def record_stop():
    engine.stop_recording()
    return {"ok": True}


@app.post("/api/record/discard-last")
async def record_discard_last():
    return {"discarded": engine.discard_last()}


@app.get("/api/record/summary")
async def record_summary():
    people = {}
    keys = sorted(C.GESTURE_KEYS, key=len, reverse=True)
    if C.DATA_DIR.exists():
        for f in sorted(C.DATA_DIR.glob("*/*.npz")):
            g = next((k for k in keys if f.stem == k or f.stem.startswith(k + "_")), None)
            if g:
                counts = people.setdefault(f.parent.name, {})
                counts[g] = counts.get(g, 0) + 1
    return dict(people=people, gestures=[dict(key=k, label=l) for k, l in C.GESTURES])


@app.post("/api/train")
async def train():
    """Runs training in a separate process (so PyTorch never sits in the demo's memory)
    and streams its printout to the page."""
    async def gen():
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-u", "-m", "app.train", cwd=str(C.ROOT),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            yield line.decode("utf-8", errors="replace")
        code = await proc.wait()
        engine.reload_model()
        yield f"\n[exit code {code}]\n"
    return StreamingResponse(gen(), media_type="text/plain; charset=utf-8")


# ---------------- LLM ----------------
class LoadReq(BaseModel):
    model: str | None = None


class TextReq(BaseModel):
    text: str
    k: int = 100


@app.get("/api/llm/status")
async def llm_status():
    return llm.status()


@app.post("/api/llm/load")
async def llm_load(req: LoadReq):
    await llm.ensure(req.model)
    return llm.status()


@app.post("/api/llm/next")
async def llm_next(req: TextReq):
    if not req.text:
        return dict(tokens=[], ms=0, model=llm.model)
    if llm.state != "ready":
        await llm.ensure()
        if llm.state != "ready":
            return JSONResponse(llm.status(), status_code=503)
    try:
        return await llm.next_tokens(req.text[-6000:], k=max(10, min(req.k, 200)))
    except Exception as e:
        return JSONResponse(dict(error=str(e)), status_code=500)


@app.post("/api/llm/tokenize")
async def llm_tokenize(req: TextReq):
    if llm.state != "ready":
        await llm.ensure()
        if llm.state != "ready":
            return JSONResponse(llm.status(), status_code=503)
    try:
        return dict(tokens=await llm.tokenize(req.text))
    except Exception as e:
        return JSONResponse(dict(error=str(e)), status_code=500)


@app.post("/api/llm/stop")
async def llm_stop():
    llm.stop()
    return llm.status()


def main():
    ap = argparse.ArgumentParser(description="How LLMs Think - live demo app")
    ap.add_argument("--camera", type=int, default=0, help="webcam index (try 1 for an external webcam)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--pose", choices=["lite", "full"], default="full",
                    help="MediaPipe pose model: lite is faster, full is steadier")
    ap.add_argument("--port", type=int, default=C.APP_PORT)
    ap.add_argument("--keep-llm", action="store_true",
                    help="keep the LLM loaded while the camera runs (default: unload it, as in the plan)")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--replay", default=None, help=argparse.SUPPRESS)  # testing: replay recordings, no webcam
    a = ap.parse_args()
    ARGS.__dict__.update(vars(a))
    url = f"http://localhost:{a.port}/"
    print(f"\n  How LLMs Think demo  ->  {url}\n  (Ctrl+C to stop)\n", flush=True)
    if not a.no_browser:
        import threading
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
