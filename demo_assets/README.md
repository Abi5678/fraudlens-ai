# Video demo assets (GTC 2026)

Use these in a browser at **1920×1080** when recording the 2-minute demo.

| File | Scene | Usage |
|------|--------|--------|
| `title_card.html` | 1 (0:00–0:15) | Open in browser, full screen; hook + title |
| `architecture_slide.html` | 2 (0:15–0:35) | Architecture overview with NVIDIA stack |
| `end_card.html` | 6 (1:50–2:00) | Repo + demo URL + #NVIDIAGTC |
| `narration_script.txt` | All | Narration copy for TTS (ElevenLabs, OpenAI TTS, etc.); one block per scene |

Open with: `open title_card.html` (macOS) or double-click. Refresh to ensure clean state before recording.

**AI voice:** Copy each scene block from `narration_script.txt` into your TTS tool; export as scene1.mp3 … scene6.mp3 and overlay on your screen recording in the editor.

### Generate audio with TTS (free by default)

The script uses **Microsoft Edge's online text-to-speech** for an OpenAI-like experience at no cost. No API key or Docker required.

**Free (default):** From the project root, run:

```bash
pip install edge-tts   # or use project venv with requirements.txt
python scripts/generate_demo_tts.py
```

Output is written to `demo_assets/audio/scene1.mp3` … `scene6.mp3`.

**Other options:**
- **Free (OpenAI-compatible server):** [openai-edge-tts](https://github.com/travisvn/openai-edge-tts). Run `docker run -d -p 5050:5050 travisvn/openai-edge-tts:latest`, set `TTS_BASE_URL=http://localhost:5050` in `.env`, then run the script.
- **Paid:** Set `OPENAI_API_KEY` in `.env` (and do not set `TTS_BASE_URL`) to use OpenAI TTS.

Optional args: `--voice` (alloy, echo, fable, onyx, nova, shimmer), `--speed` (0.25–4.0).

### Sample claim PDF (no real identity)

To regenerate the sample claim PDF (text from `sample_claim.json`, images from a source PDF such as `MB (2).PDF`):  
`python scripts/generate_sample_claim_pdf.py`  
Output: `demo_assets/sample_claim.pdf`. Optional: `--source-pdf`, `--claim-json`, `--output`.
