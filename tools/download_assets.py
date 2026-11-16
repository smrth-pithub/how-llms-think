"""Downloads the big files that aren't in the GitHub repo (about 3.3 GB):
llama.cpp (Windows CUDA build), the two Qwen3 models, and MediaPipe's pose models.

Skips anything already there with the right size, so it's safe to run again.
setup.bat runs it for you.

    .venv\\Scripts\\python -m tools.download_assets
"""
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
LLAMA = ROOT / "tools" / "llama"

LLAMA_BUILD = "b12409"
LLAMA_URL = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_BUILD}/"

# (where it goes, where it comes from, exact size in bytes)
FILES = [
    (MODELS / "Qwen3-1.7B-Q8_0.gguf",
     "https://huggingface.co/Qwen/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q8_0.gguf", 1834426016),
    (MODELS / "Qwen3-0.6B-Q8_0.gguf",
     "https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf", 639446688),
    (MODELS / "pose_landmarker_full.task",
     "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/"
     "pose_landmarker_full.task", 9398198),
    (MODELS / "pose_landmarker_lite.task",
     "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/"
     "pose_landmarker_lite.task", 5777746),
]
LLAMA_ZIPS = [
    (f"llama-{LLAMA_BUILD}-bin-win-cuda-13.5-x64.zip", 203425350),
    ("cudart-llama-bin-win-cuda-13.5-x64.zip", 426314199),   # CUDA runtime; on laptops without NVIDIA it runs on the CPU
]


def download(url: str, dest: Path, size: int):
    if dest.exists() and dest.stat().st_size == size:
        print(f"  have      {dest.name}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    print(f"  download  {dest.name} ({size / 1e6:,.0f} MB)")
    req = urllib.request.Request(url, headers={"User-Agent": "how-llms-think-setup"})
    t0, done, last = time.time(), 0, 0.0
    with urllib.request.urlopen(req, timeout=60) as r, open(part, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if time.time() - last > 1:
                last = time.time()
                speed = done / max(last - t0, 1e-6) / 1e6
                print(f"\r            {100 * done / size:5.1f}%  {speed:5.1f} MB/s", end="", flush=True)
    print(f"\r            100.0%  in {time.time() - t0:.0f} s        ")
    if part.stat().st_size != size:
        part.unlink()
        raise RuntimeError(f"{dest.name}: download incomplete, run setup.bat again")
    part.replace(dest)


def main():
    print("Big files (models and llama.cpp):")
    for dest, url, size in FILES:
        download(url, dest, size)

    if (LLAMA / "llama-server.exe").exists() and (LLAMA / "ggml-cuda.dll").exists():
        print("  have      llama.cpp")
    else:
        for name, size in LLAMA_ZIPS:
            z = LLAMA / name
            download(LLAMA_URL + name, z, size)
            with zipfile.ZipFile(z) as zf:
                zf.extractall(LLAMA)
            z.unlink()
        if not (LLAMA / "llama-server.exe").exists():
            raise RuntimeError("llama-server.exe missing after unzipping")
        print("  unpacked  llama.cpp")
    print("All big files present.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\nDownload failed: {e}")
        sys.exit(1)
