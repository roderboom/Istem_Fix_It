# FixBot — Architecture

## Overview

FixBot is a Telegram bot that receives photos of broken items, runs them through a local vision AI model, and returns annotated images with structured repair instructions. Everything runs locally via Podman — no internet connection required at runtime.

---

## Container Architecture

Two containers run via `podman-compose`:

```
┌─────────────────────────────────────────────────────┐
│  Host Machine (Linux, RTX 5070 Ti 16GB, 94GB RAM)   │
│                                                     │
│  ┌─────────────────────┐  ┌──────────────────────┐  │
│  │   fixbot-ollama     │  │    fixbot-bot        │  │
│  │                     │  │                      │  │
│  │  Ollama server      │◄─│  Python 3.12         │  │
│  │  port 11434         │  │  python-telegram-bot │  │
│  │  GPU: CDI all       │  │  httpx, Pillow       │  │
│  │                     │  │                      │  │
│  │  Volume:            │  │  Volume:             │  │
│  │  ollama-models      │  │  repair-history      │  │
│  └─────────────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

`fixbot-bot` waits for `fixbot-ollama` to pass its healthcheck before starting. The bot communicates with Ollama over the internal Docker network at `http://ollama:11434`.

---

## Request Flow

### Photo sent by user

```
User sends photo (+ optional caption)
        │
        ▼
  handle_photo()  [bot.py]
        │
        ├── Detect language from caption (Hebrew / English)
        ├── Album buffering: wait 2.5s for all photos in same media_group_id
        │
        ▼
  _run_analysis()  [bot.py]
        │
        ├── Status: "🔍 Analyzing…"
        │
        ├── vision.py: analyze_image()
        │     ├── Resize images (1 photo → 1120px, 2 photos → 896px each)
        │     ├── Snap dimensions to multiples of 28 (Gemma4 patch grid)
        │     ├── Base64-encode images
        │     ├── Build system prompt (language-aware)
        │     ├── POST /api/chat → Ollama
        │     ├── Extract JSON from response (_extract_json + _repair_json)
        │     └── Validate + normalize JSON (_validate)
        │
        ├── Status: "🖊️ Marking areas…"
        │
        ├── annotator.py: annotate_image()
        │     ├── Apply EXIF rotation (ImageOps.exif_transpose)
        │     ├── Scale image if too small
        │     ├── For each component: convert grid_col/grid_row → x/y pixels
        │     └── Draw numbered dots (dark outline + white fill + number)
        │
        ├── history.py: save_repair()
        │
        └── Send annotated photo + formatted text to user
```

### Text message (follow-up question)

```
User sends text
        │
        ▼
  handle_text()  [bot.py]
        │
        ├── Detect language
        ├── vision.py: ask_followup()
        │     └── POST /api/chat with last image + conversation history
        └── Reply with answer
```

---

## vision.py — Model Interface

### Key functions

**`analyze_image(images, caption, history, lang)`**
- Primary analysis call. Sends 1-2 images with the repair prompt.
- Returns a validated dict with `problem_summary`, `severity`, `safety_warnings`, `tools_needed`, `materials_needed`, `components`, `steps`, `professional_advice`.

**`ask_followup(images, question, history, lang)`**
- Conversational follow-up. Sends the last image with full conversation history.

### Image sizing strategy

| Photos sent | Max side | Purpose |
|-------------|----------|---------|
| 1 | 1120px | Maximum detail, fits in VRAM |
| 2 | 896px each | Both fit without OOM |
| 3+ | 896px, capped at 2 | Prevent VRAM overflow |

All dimensions are snapped to multiples of 28 — this aligns with Gemma4's vision encoder patch size, preventing partial patches at image edges and improving coordinate accuracy.

### JSON schema returned by model

```json
{
  "problem_summary": "One sentence describing the problem",
  "severity": "low|medium|high",
  "safety_warnings": ["warning 1"],
  "tools_needed": ["tool 1"],
  "materials_needed": ["material 1"],
  "components": [
    {
      "id": 1,
      "name": "Part name",
      "grid_col": 1-10,
      "grid_row": 1-10
    }
  ],
  "steps": [
    {
      "step": 1,
      "title": "Step title",
      "instruction": "Paragraph describing what to do, why, and what to watch out for"
    }
  ],
  "professional_advice": "When to call a professional"
}
```

### JSON robustness

Three layers of extraction:
1. Direct `json.loads()` on the raw response
2. Strip markdown code fences, retry parse
3. Regex to find outermost `{...}` block
4. `_repair_json()` — balances unclosed brackets/braces from truncated responses
5. Fallback dict with error message if all fail

### Prompt design

- **Language**: system prompt instructs the model to respond in Hebrew or English based on detected user language. Gemma4 handles both natively.
- **Components vs Steps**: these are fully decoupled. `components` are physical parts visible in the image (each gets a dot). `steps` are repair instructions with no coordinate references.
- **Grid system**: model outputs `grid_col` + `grid_row` on a 10×10 grid instead of raw percentages. Cell centers map to 5%, 15%, 25%... 95% on each axis. This is more reliable than asking models to estimate raw percentages.
- **Location description**: model must first describe the part's location in words before setting grid coordinates — forces spatial reasoning before guessing numbers.
- **CJK sanitizer**: strips any Chinese/Japanese/Korean characters that bleed through from Qwen's training data (not needed for Gemma4 but kept as a safety net).

---

## annotator.py — Image Annotation

### Dot placement

```
grid_col (1-10) → x% = (col × 10) - 5
grid_row (1-10) → y% = (row × 10) - 5

x_pixels = x% / 100 × image_width
y_pixels = y% / 100 × image_height
```

### EXIF rotation

Phone cameras store images in raw sensor orientation with an EXIF tag indicating the intended display rotation. Without correction, a portrait photo is stored as landscape pixels — the model sees it correctly rotated but the annotator would place dots on unrotated coordinates.

`ImageOps.exif_transpose()` physically rotates the pixel data to match the display orientation before any coordinate math, ensuring the model's grid coordinates and the actual pixel positions match.

### Dot rendering

Each dot: dark ring (outline) + white filled circle + centered step number. Size scales with image resolution (`W // 55`).

---

## history.py — Repair History

- Stores last 50 repairs per user in `/data/history/<user_id>.json`
- Mounted as a named Podman volume (`repair-history`) so history persists across container restarts
- Saved fields: `timestamp`, `problem_summary`, `severity`, `tools_needed`, `professional_advice`
- `/history` command shows last 10 repairs formatted for Telegram

---

## bot.py — Session Management

### Per-user session state

```python
{
  "history":       [],    # last 6 exchanges (role + content)
  "last_images":   [],    # raw bytes of last analyzed photos
  "last_analysis": None,  # last analysis dict
  "album_buffer":  {},    # keyed by media_group_id, holds photos + timer task
  "lang":          "en",  # detected from user's last message
}
```

### Album (multi-photo) handling

Telegram sends album photos as separate messages sharing a `media_group_id`. The bot buffers each photo and schedules a 2.5-second timer. If another photo arrives before the timer fires, it cancels and reschedules. After 2.5 seconds of silence, all buffered photos are sent together for analysis.

### Language detection

`detect_language()` in `vision.py` scans the text for Unicode characters in the Hebrew range (`\u05d0`–`\u05ea`). Any Hebrew character → `lang = "he"`. Updated on every user message so switching languages mid-conversation works.

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `TELEGRAM_TOKEN` | Yes | — | Bot token from @BotFather |
| `MODEL_NAME` | No | `gemma4:26b` | Ollama model to use |
| `OLLAMA_HOST` | No | `http://localhost:11434` | Ollama server URL |
| `HISTORY_DIR` | No | `/data/history` | Path for repair history JSON files |

---

## Model Recommendations

| Model | VRAM | Speed | Notes |
|-------|------|-------|-------|
| `gemma4:26b` | ~18 GB | ~90s | Best quality, MoE: only ~4B active per token |
| `gemma4:e4b` | ~10 GB | ~30s | Fully fits 16GB, fast, good quality |
| `qwen2.5vl:32b` | ~21 GB | ~3min | Original model, partially offloads to RAM |
| `qwen2.5vl:7b` | ~5 GB | ~25s | Fast fallback |

Gemma4 is preferred — better multilingual support (Hebrew/English), stronger JSON instruction following, and the MoE architecture means minimal speed penalty despite the large parameter count.
