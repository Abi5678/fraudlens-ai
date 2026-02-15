# GTC 2026 video recording checklist

Use this before and during recording. See [DEMO_SCRIPT.md](DEMO_SCRIPT.md) for full script and [demo_assets/](demo_assets/) for title/end slides.

## Before recording

- [ ] **Env:** `source venv/bin/activate` (or your venv), `NVIDIA_API_KEY` set in `.env`
- [ ] **App:** `streamlit run ui/app.py` — dashboard loads at http://localhost:8501
- [ ] **Sample claim:** In UI: "Use Sample Claim" → "Load Sample Claim" (no upload needed)
- [ ] **Pre-run (optional):** Click "Analyze Claim" once so you know timing (~30–60 s) and can use results on camera
- [ ] **Tests:** `pytest tests/ -v` — all 61 pass (for Scene 5)
- [ ] **Slides:** Open `demo_assets/title_card.html`, `architecture_slide.html`, `end_card.html` in a browser; set resolution to 1920×1080
- [ ] **Recording:** OBS / Loom / QuickTime ready; mic checked (or use AI voice — see below); one browser tab, no bookmarks bar

**AI voice (optional):** Either copy each scene from `demo_assets/narration_script.txt` into a TTS web UI, or run `python scripts/generate_demo_tts.py` to generate `demo_assets/audio/scene1.mp3` … `scene6.mp3`. For free TTS: run `docker run -d -p 5050:5050 travisvn/openai-edge-tts:latest` and set `TTS_BASE_URL=http://localhost:5050` in `.env`. Overlay the MP3s on your screen recording and record video with mic muted.

## During recording (6 scenes, ~2 min)

| Scene | Time | What to show |
|-------|------|--------------|
| 1 | 0:00–0:15 | Title card (or README) — hook |
| 2 | 0:15–0:35 | Architecture slide — Nemotron, NIM, NeMo, cuGraph, Milvus |
| 3 | 0:35–0:55 | Streamlit: Use Sample Claim → Load → Analyze; show spinner |
| 4 | 0:55–1:30 | Results: score, risk factors, Inconsistencies, Patterns, Network, Narrative |
| 5 | 1:30–1:50 | Terminal: `pytest tests/ -v`; differentiators |
| 6 | 1:50–2:00 | Impact + end card (repo + demo + #NVIDIAGTC) |

## After recording

- [ ] Trim / add captions; export MP4 1080p (H.264). Optional: auto-captions for accessibility.
- [ ] Post on LinkedIn and/or X with #NVIDIAGTC, tag judge, add GitHub + demo links (see template below).
- [ ] Update [SUBMISSION.md](SUBMISSION.md) and [README.md](README.md) with the video URL.

### Social post template (copy and adapt)

```
FraudLens AI: multi-agent insurance fraud detection in under 5 minutes — from 60+ days per claim to minutes. Built with NVIDIA NIM, Nemotron-Parse, NeMo Retriever, cuGraph, and Milvus. Detects fraud rings and explains every score. Open source.

https://github.com/Abi5678/fraudlens-ai
https://fraudlensai.streamlit.app

@CarterAbdallah #NVIDIAGTC
```

(Use official handle for the judge you’re submitting to: Carter Abdallah, Nader Khalil, or Bryan Catanzaro.)

**Deadline:** February 15, 2026
