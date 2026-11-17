# How LLMs Think: live demo app

One local web app for the whole session. Everything runs offline on this laptop:
no Wi-Fi, no API keys, no cost.

| Page | Segment | What it shows |
|---|---|---|
| `/gesture` | 1–2 | Webcam → body points → small neural net → 7 ranked guesses, temperature slider |
| `/tokenizer` | 3 | "unbelievable" → `un` `belie` `vable` |
| `/llm` | 3–4 | Live top-10 next tokens from Qwen3-1.7B, pick / auto-write loop, same slider |
| `/record` | prep | Record people doing each gesture, then train (1–3 minutes) |

---

## Before the day: get the gesture model trained (about 30 min)

1. **Plug the laptop in** and set Windows power mode to **Best performance**
   (Settings → System → Power). On battery the camera demo runs at half speed.
2. Plug in the webcam, double-click **`check_camera.bat`**. It prints which camera
   number shows a real picture. (This laptop's own camera doesn't work: it shows up as
   camera 0 with a black picture, so the external webcam will probably be camera 1.)
3. Double-click **`start.bat`** (or run `start.bat --camera 1` if the check said 1).
   The browser opens at http://localhost:8000.
4. Go to **Record & train**. For each person: type their name, press **Start recording**,
   and they follow the prompts on screen: 7 gestures, 3 s to get ready, then 10 s of recording
   each (about 1.5 min per person). Stand **2–3 m back**, like on stage.
   - Record **5–6 different people**. Different people matter more than more takes.
   - Ask them to vary it: speed, size, left/right hand.
   - Did someone do the wrong gesture? Press **Discard last clip** straight away.
5. Press **Train** (1–3 minutes). It prints how well the model does on each person **it never saw**
   (it trains without them, then tests on them). Over 80% overall is good for the demo.
   The table at the bottom shows which gestures get mixed up: record more of those.
6. Open the **Gesture demo** and try it, including the head-scratch.

Only body-point numbers are saved (in `data/<name>/`), never video.

Training stretches your recordings: each clip is also used mirrored (left/right swapped) and
once more as a slightly different "person": a bit faster or slower, camera tilted up to 8°,
arms up to 12% longer or shorter. That helps the model cope with people it hasn't seen.
(`python -m app.train --augment 0` turns this off, if you want to compare.)

**Extra public data: MPOSE2021.** Training also uses [MPOSE2021](https://zenodo.org/records/5507363)
(CC BY 4.0, Mazzia et al. 2021): one-second body-point clips of 100 more people, covering 5 of
our 7 gestures (not thumbs up or hands up, which come only from club recordings). `setup.bat`
downloads it (~90 MB, once), or run `.venv\Scripts\python -m tools.import_mpose`.
With 4 club members it scored 82.6% on unseen club members, vs 84.1% for club recordings
alone: its lab cameras film people differently from our webcam. To train without it:
`.venv\Scripts\python -m app.train --no-mpose`. The club-only model from 17 Nov is kept as
`models/gesture_model_club_only_backup.npz` (+ `.json`): copy both over `gesture_model.*` to go back.

Its "scratch-head" clips are never trained on; every training run reports what the model
says for them (a head-scratch test). They're filmed full-body in labs, so still try a real
head-scratch on your own webcam. Recording your own people takes about 15 minutes, and
"we collected our own data" is part of the Segment 5 story.

## Before the session (in the room)

- [ ] Laptop **plugged in**, power mode **Best performance**, other apps closed (8 GB RAM)
- [ ] Laptop, webcam, projector and HDMI tested an hour before
- [ ] `check_camera.bat` → `start.bat --camera N`
- [ ] Camera placed so a volunteer 2–3 m away is fully in frame (top of head to hips)
- [ ] Room lights the same as when you tested
- [ ] **Wi-Fi off**, then click through every page once (proves nothing needs the internet)
- [ ] Open `/llm` once so the model is warm, then go back to `/gesture`
- [ ] Backup video ready (record one with **Win + Alt + R** while the demo runs fullscreen)
- [ ] Two volunteers lined up

## During the session

Press **F** for fullscreen and **H** to hide the top bar. **D** switches dark/light
(use whichever reads better on the projector).

**Segment 1 · Gesture demo** (`/gesture`)
- Volunteers try to fool it. Ask someone to scratch their head: it has no "nothing"
  option, so it must pick one of the seven, often confidently.
- Hand the temperature slider to someone (or use **← →**). Left: the top guess heads
  for 99%. Right: the bars flatten into a coin toss. **1** resets to 1.0.
- **Space** samples a guess from the percentages, the same way an LLM picks a word.

**Segment 2 · Under the hood** (`/gesture`, press **N**)
- Shows the raw scores next to each bar, the live numbers the body turns into,
  and the actual `softmax()` source code that is making the percentages.

**Segment 3 · From gestures to words**
- `/tokenizer`: type or click the presets (`unbelievable`, `नमस्ते` splits into raw bytes).
- `/llm`: type a sentence, the top-10 next tokens update as you type (about 45 ms).
  **Enter** picks the next token and adds it to the text: the loop. The model's own
  words are highlighted. **Ctrl+Enter** auto-writes, **Esc** stops, **Undo** goes back.
- Temperature on text: choose "Once upon a time" from the example prompts and auto-write
  at **0.2** (safe, repetitive), then **1.0**, then **3+** (nonsense).

**Segment 4 · Hallucination**
- `/llm` example prompts: "The first person to walk on Mars was" and "The winner of the
  2031 Cricket World Cup was". It can't know, but it still has to pick a next word, and
  the bars show it picking confidently. Same as the head-scratch.

## If something goes wrong

| Problem | Do this |
|---|---|
| Webcam or lighting breaks the demo | Play the backup video and narrate over it |
| Black camera picture / wrong camera | Close Zoom/Teams/Camera, run `check_camera.bat`, restart with `--camera N` |
| Choppy camera (the corner shows under ~12 fps) | Check the charger; restart with `start.bat --pose lite` |
| Bars flicker too fast to read | Open `/gesture?smooth=0.3` (default 0.15; bigger = steadier) |
| Works only on the people it trained on | Record more people; or make it the teaching moment about training data |
| Next-token viewer lags (the ms badge turns orange, over 500 ms) | Pick **Qwen3-0.6B** in the model menu |
| "Loading the model…" for more than 30 s | Check `logs/llama-server.log`; restart `start.bat` |
| Session runs long | Cut Segment 4 short; Q&A keeps its 15 min |

Measured on this laptop (on battery): Qwen3-1.7B 41 ms per update, loads in 6 s;
Qwen3-0.6B 32 ms. Run `speedtest.bat` (with the app closed) to re-check.

**One demo at a time:** opening the camera page unloads the language model (frees about
0.8 GB of RAM), and opening a text page loads it again (5–10 s). To keep both loaded,
start with `start.bat --keep-llm`.

---

## Running it on another laptop

Works on any **Windows** laptop with Python 3.11–3.14 (install **Python 3.14** from python.org
if it has none). Two ways to get the folder there:

**A. Copy the folder (fastest, keeps your recordings).** Copy `how-llms-think` across by
pendrive, cable or network. You can leave out `.venv`: it only works on the laptop that made it.
Then double-click **`setup.bat`**: it rebuilds `.venv` (~600 MB of packages, 15–25 minutes,
needs internet once). Your recordings (`data/`) and trained gesture model come along.

**B. From GitHub (no pendrive).** On the repo page: **Code → Download ZIP** and unzip it, or
`git clone` it. Then double-click **`setup.bat`**. It also downloads the big files GitHub can't
hold: llama.cpp and the Qwen and MediaPipe models, about 3.3 GB. Allow 30–60 minutes,
depending on the internet.

GitHub has the code only. People's recordings and the trained gesture model stay on the laptop
where you recorded (they have names in them), so with route B either record again or copy
`data/` and `models/gesture_model.*` across separately.

Either way, afterwards: `check_camera.bat`, then `start.bat`. Everything runs offline.

**No NVIDIA GPU?** The language model falls back to the CPU on its own. Measured on this
laptop with the GPU switched off: Qwen3-1.7B about 230 ms per update (worst 360 ms),
Qwen3-0.6B about 55 ms. On a slower laptop, if the ms badge turns orange, pick **Qwen3-0.6B**.
The gesture demo never uses the GPU.

**Two laptops** (the plan's option): run the gesture demo on the borrowed laptop and the
language model on this one. Set up the folder on both; on each, only open the pages it runs.

---

## How it's built (for Segment 5)

```
app/
  tracker.py         MediaPipe: camera frame -> 33 body points (x, y)
  features.py        sliding window (last 1 s): body shape + movement speed -> 110 numbers
  train.py           PyTorch: 110 -> 64 -> 32 -> 7 raw scores; tests on unseen people
  classifier.py      the trained net running live (plain numpy: 3 matrix multiplies)
  gesture_engine.py  camera loop + recorder, runs only while a camera page is open
  llm.py             runs llama.cpp's server, asks for the top-100 next tokens
  server.py          the web app (FastAPI)
web/
  common.js          softmax(), sampling, temperature slider, ranked bars: shared by both demos
  gesture.*, llm.*, tokenizer.html, record.*
models/              Qwen3-1.7B and 0.6B (Q8 GGUF), MediaPipe pose model, trained gesture model
tools/llama/         llama.cpp b12409, CUDA 13 build, all layers on the GPU
```

The same `softmax(scores, temperature)` function in `web/common.js` produces the
percentages in **both** demos: gesture scores in one, token log-probabilities in the other.

Command line: `.venv\Scripts\python -m app.server --help`, `-m app.train`.
Exact package versions are in `requirements.txt` (what `setup.bat` installs).
