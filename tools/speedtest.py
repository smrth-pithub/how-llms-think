"""Speed test from the plan: how long does each model take to update the top-10?

Rule: if it's over about 0.5 s, use the smaller model. Smooth beats smart on a projector.
Close the demo app first (both use the same port).

    speedtest.bat        (or: .venv\\Scripts\\python -m tools.speedtest)
"""
import asyncio
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import config as C  # noqa: E402
from app.llm import LlamaManager  # noqa: E402

TEXT = "Once upon a time, in a small village near the mountains, there lived a young girl who loved to"


async def main():
    for name in C.LLM_MODELS:
        m = LlamaManager()
        t0 = time.time()
        await m.ensure(name)
        if m.state != "ready":
            print(f"{name}: could not load ({m.error})")
            m.stop()
            continue
        load = time.time() - t0
        times = []
        for i in range(12, len(TEXT), 4):            # like someone typing
            d = await m.next_tokens(TEXT[:i], k=100)
            times.append(d["ms"])
        med, worst = statistics.median(times), max(times)
        verdict = "OK" if worst < 500 else "TOO SLOW: use the smaller model"
        print(f"{name:12s} load {load:4.1f} s · per update: median {med:4.0f} ms, worst {worst:4.0f} ms  -> {verdict}")
        m.stop()
        await m.client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
