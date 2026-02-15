#!/usr/bin/env python3
"""
Generate demo narration audio with OpenAI-compatible TTS.
Reads demo_assets/narration_script.txt, calls TTS per scene, saves to demo_assets/audio/scene1.mp3 ... scene6.mp3.

Uses Microsoft Edge's online text-to-speech (free) for an OpenAI-like experience without cost. Options:
- Free (default): edge-tts Python package — no API key or Docker. pip install edge-tts.
- Free (server): openai-edge-tts (https://github.com/travisvn/openai-edge-tts). TTS_BASE_URL=http://localhost:5050.
- Paid: OpenAI TTS. Set OPENAI_API_KEY (no TTS_BASE_URL).
"""
from pathlib import Path
from typing import List
import os
import sys
import argparse
import asyncio

# Project root (parent of scripts/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

NARRATION_FILE = PROJECT_ROOT / "demo_assets" / "narration_script.txt"
OUTPUT_DIR = PROJECT_ROOT / "demo_assets" / "audio"
DEFAULT_VOICE = "nova"
DEFAULT_MODEL = "tts-1-hd"
# Free edge-tts uses tts-1; OpenAI supports tts-1-hd
EDGE_TTS_DEFAULT_MODEL = "tts-1"
# Map OpenAI-style voice names to edge-tts voice IDs (Microsoft Edge online TTS)
EDGE_VOICE_MAP = {
    "alloy": "en-US-GuyNeural",
    "echo": "en-US-GuyNeural",
    "fable": "en-GB-SoniaNeural",
    "onyx": "en-US-GuyNeural",
    "nova": "en-US-JennyNeural",
    "shimmer": "en-US-AriaNeural",
    "guy": "en-US-GuyNeural",
    "male": "en-US-ChristopherNeural",
    "christopher": "en-US-ChristopherNeural",
}


def _clean_tts_text(s: str) -> str:
    """Strip markdown bold and normalize whitespace for TTS."""
    import re
    s = re.sub(r"\*\*([^*]+)\*\*", r"\1", s)  # **word** -> word
    s = " ".join(s.split())  # collapse whitespace
    return s.strip()


def parse_demo_script_md(path: Path) -> List[str]:
    """Parse DEMO_SCRIPT.md into narration strings. Finds each ## SCENE/Scene block and extracts **Narration:** quoted text."""
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    sections = [s.strip() for s in text.split("\n## ") if s.strip()]
    scenes = []
    for section in sections:
        if "**Narration" not in section:
            continue
        lines = section.splitlines()
        start_idx = None
        for i, line in enumerate(lines):
            if "**Narration" in line and "**" in line:
                start_idx = i
                break
        if start_idx is None:
            continue
        # Narration block follows; collect until --- or **Action (do not break on blank lines inside quote)
        quote_lines = []
        for i in range(start_idx + 1, len(lines)):
            line = lines[i]
            if line.strip() == "---":
                break
            if line.strip().startswith("**Action"):
                break
            quote_lines.append(line)
        raw = "\n".join(quote_lines)
        # Extract text between first " and last "
        first = raw.find('"')
        last = raw.rfind('"')
        if first == -1 or last == -1 or first >= last:
            continue
        narration = raw[first + 1 : last].strip()
        narration = _clean_tts_text(narration)
        if narration:
            scenes.append(narration)
    return scenes


def parse_voiceover_end_to_end(path: Path) -> List[str]:
    """Parse voiceover_script_end_to_end.txt: blocks separated by ---, alternating SCENE N header and narration text."""
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    segments = [s.strip() for s in text.split("\n---\n") if s.strip()]
    # After optional header (seg 0), blocks alternate: "SCENE N — Title", then narration. Narration at 2, 4, 6, ...
    scenes = []
    for i in range(8):
        idx = 2 * i + 2
        if idx >= len(segments):
            break
        narration = segments[idx]
        lines = [line for line in narration.splitlines() if not line.strip().startswith("#")]
        narration = " ".join(l.strip() for l in lines if l.strip()).strip()
        if narration:
            scenes.append(narration)
    return scenes


def parse_narration(path: Path) -> List[str]:
    """Parse narration_script.txt into 6 scene texts. File uses --- on its own line between each block."""
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    segments = [s.strip() for s in text.split("\n---\n") if s.strip()]
    # After header (seg 0), blocks alternate: SCENE N header, then narration. So scene i narration = seg[2*i].
    need = 1 + 6 * 2  # header + 6 (header+narration) pairs
    if len(segments) < need:
        raise ValueError(f"Expected at least {need} blocks (header + 6 scenes), got {len(segments)}")
    scenes = []
    for i in range(6):
        narration = segments[2 * i + 2]  # scene 1 narration at seg 2, scene 2 at seg 4, etc.
        lines = [line for line in narration.splitlines() if not line.strip().startswith("#")]
        narration = "\n".join(lines).strip()
        if not narration:
            raise ValueError(f"Scene {i + 1} has no narration text")
        scenes.append(narration)
    return scenes


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate demo TTS audio (OpenAI-compatible: edge-tts or OpenAI)")
    ap.add_argument("--voice", default=DEFAULT_VOICE, help="Voice: alloy, echo, fable, onyx, nova, shimmer, guy, male, christopher, or edge-tts ID e.g. en-US-ChristopherNeural (default: nova)")
    ap.add_argument("--model", default=None, help="Model (default: tts-1 for edge-tts, tts-1-hd for OpenAI)")
    ap.add_argument("--speed", type=float, default=1.0, help="Speech speed 0.25–4.0 (default: 1.0)")
    ap.add_argument("--script", default=None, help="Narration script path (default: demo_assets/narration_script.txt)")
    ap.add_argument("--output-dir", default=None, help="Output directory for MP3s (default: demo_assets/audio)")
    args = ap.parse_args()

    narration_file = Path(args.script) if args.script else NARRATION_FILE
    if not narration_file.is_absolute():
        narration_file = PROJECT_ROOT / narration_file
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    base_url = (os.getenv("TTS_BASE_URL") or "").strip().rstrip("/")
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    use_edge_server = bool(base_url)
    use_openai = bool(api_key) and not use_edge_server
    use_edge_package = not use_edge_server and not use_openai

    if not narration_file.exists():
        print(f"Error: Narration file not found: {narration_file}", file=sys.stderr)
        sys.exit(1)

    use_voiceover_parser = "voiceover_script_end_to_end" in narration_file.name
    use_md_parser = narration_file.suffix.lower() == ".md" or "DEMO_SCRIPT" in narration_file.name
    if use_voiceover_parser:
        scenes = parse_voiceover_end_to_end(narration_file)
        if not scenes:
            print("Error: No narration blocks found in voiceover script", file=sys.stderr)
            sys.exit(1)
    elif use_md_parser:
        scenes = parse_demo_script_md(narration_file)
        if not scenes:
            print("Error: No narration blocks found in markdown file", file=sys.stderr)
            sys.exit(1)
    else:
        scenes = parse_narration(narration_file)
        if len(scenes) != 6:
            print(f"Error: Expected 6 scenes for .txt script, got {len(scenes)}", file=sys.stderr)
            sys.exit(1)

    num_scenes = len(scenes)
    output_dir.mkdir(parents=True, exist_ok=True)

    if use_edge_package:
        # Free: edge-tts Python package (Microsoft Edge online TTS, no API key or Docker)
        try:
            import edge_tts
        except ImportError:
            print("Error: edge-tts not installed. Run: pip install edge-tts", file=sys.stderr)
            sys.exit(1)
        # Direct edge-tts voice ID (e.g. en-US-ChristopherNeural) if --voice contains hyphen and "Neural"
        if "-" in args.voice and "Neural" in args.voice:
            voice_id = args.voice
        else:
            voice_id = EDGE_VOICE_MAP.get(args.voice, "en-US-JennyNeural")
        pct = int((args.speed - 1.0) * 100)
        rate = f"{pct:+d}%" if pct != 0 else "+0%"
        print(f"Using free TTS: edge-tts (Microsoft Edge online), voice={voice_id}")

        async def _generate() -> None:
            for i, content in enumerate(scenes, start=1):
                out_path = output_dir / f"scene{i}.mp3"
                print(f"Generating scene {i}/{num_scenes} -> {out_path.name} ...")
                communicate = edge_tts.Communicate(content, voice_id, rate=rate)
                await communicate.save(str(out_path))
                print(f"  Saved {out_path}")

        asyncio.run(_generate())
        print(f"Done. Output: {output_dir}")
        return

    if use_edge_server:
        api_key = (os.getenv("TTS_API_KEY") or "any").strip() or "any"
        model = args.model or EDGE_TTS_DEFAULT_MODEL
        print(f"Using free TTS: {base_url} (openai-edge-tts)")
    else:
        model = args.model or DEFAULT_MODEL
        print("Using OpenAI TTS")

    from openai import OpenAI
    if use_edge_server:
        client = OpenAI(base_url=base_url, api_key=api_key)
    else:
        client = OpenAI()

    for i, content in enumerate(scenes, start=1):
        out_path = output_dir / f"scene{i}.mp3"
        print(f"Generating scene {i}/{num_scenes} -> {out_path.name} ...")
        try:
            response = client.audio.speech.create(
                model=model,
                voice=args.voice,
                input=content,
                response_format="mp3",
                speed=args.speed,
            )
            out_path.write_bytes(response.content)
            print(f"  Saved {out_path}")
        except Exception as e:
            print(f"  Error: {e}", file=sys.stderr)
            sys.exit(1)

    print(f"Done. Output: {output_dir}")


if __name__ == "__main__":
    main()
