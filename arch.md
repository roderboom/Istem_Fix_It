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
        ├── Auth check → reject if not on whitelist
        ├── Detect language from caption (Hebrew / English)
        ├── Album buffering: wait 2.5s for all photos in same media_group_id
        │
        ▼
  _run_analysis()  [bot.py]
        │
        ├── Acquire _ollama_sem — queues if another request is running
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
        │     ├── For each component: place dot at point.x/point.y percentages
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
        ├── Auth check → reject if not on whitelist
        ├── Detect language
        ├── vision.py: ask_followup()
        │     └── POST /api/chat with last image + conversation history
        └── Reply with answer
```

---

## vision.py — Model Interface

### Token budget

`num_predict=6000`, `num_ctx=10240` for the analysis call. Hebrew text tokenises at 2–4× the token rate of English. The prompt enforces brevity (max 12 words per list item, max 25 words per step instruction) to keep total output within budget even for detailed repair guides.

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
      "point": { "x": 0-100, "y": 0-100 }
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
4. `_repair_json()` — repairs truncated responses: closes any unterminated string literal first, then balances unclosed brackets and braces
5. Fallback dict with error message if all fail

### Prompt design

- **Language**: system prompt instructs the model to respond in Hebrew or English based on detected user language. Gemma4 handles both natively.
- **Components vs Steps**: these are fully decoupled. `components` are physical parts visible in the image (each gets a dot). `steps` are repair instructions with no coordinate references.
- **Coordinate system**: model outputs direct `x/y` percentages (0–100) for each component. x=0 is the left edge, x=100 is the right edge. A 10×10 grid was tried but caused visible offsets due to coarse 10% snapping. Gemma4's spatial reasoning handles free-form percentages well.
- **Caption priority**: the user's caption is placed at the start of the user message before the analysis instruction. A `PRIORITY` rule in the prompt instructs the model to annotate exactly the parts named in the caption when specified.
- **CJK sanitizer**: strips any Chinese/Japanese/Korean characters that bleed through from Qwen's training data (not needed for Gemma4 but kept as a safety net).

---

## annotator.py — Image Annotation

### Dot placement

```
x_pixels = point.x / 100 × image_width
y_pixels = point.y / 100 × image_height
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

### Global queue

`_ollama_sem = asyncio.Semaphore(1)` — ensures only one Ollama call runs at a time. Multiple users can send photos simultaneously; the bot processes one and shows the others "⏳ In queue, please wait…". The semaphore is acquired inside `_run_analysis()` and released automatically when analysis completes, even on error.

`_queue_waiters` is a counter incremented when a request starts waiting and decremented when it acquires the semaphore.

### Album (multi-photo) handling

Telegram sends album photos as separate messages sharing a `media_group_id`. The bot buffers each photo and schedules a 2.5-second timer. If another photo arrives before the timer fires, it cancels and reschedules. After 2.5 seconds of silence, all buffered photos are sent together for analysis.

### Language detection

`detect_language()` in `vision.py` scans the text for Unicode characters in the Hebrew range (`\u05d0`–`\u05ea`). Any Hebrew character → `lang = "he"`. Updated on every user message so switching languages mid-conversation works.

---

## whitelist.py — Access Control

Stores the set of allowed user IDs in `/data/history/whitelist.json` (same persistent volume as repair history). Loaded on import — seeded from the `ALLOWED_USERS` env var, then merged with the saved JSON file.

### Approval flow

```
New user sends /start
        │
        ├── Not on whitelist → add to _pending_approval set
        ├── Send user: "request sent, please wait"
        └── Send admin: approval message with [✅ Allow] [❌ Deny] buttons
                │
                ▼
        Admin taps Allow
                │
                ├── wl.add_user(uid) → saved to whitelist.json
                ├── Admin message edits to "✅ approved"
                └── User notified: "✅ Access approved, send /start"
```

The `_pending_approval` set lives in memory — it prevents duplicate admin notifications if the user sends multiple messages while waiting. It does not persist across restarts (intentional — user just sends `/start` again).

### Admin commands

| Command | Description |
|---------|-------------|
| `/adduser <id>` | Approve a user directly |
| `/removeuser <id>` | Revoke access (cannot remove admin) |
| `/listusers` | List all approved users |

All commands are gated — non-admins get a 🚫 rejection.

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `TELEGRAM_TOKEN` | Yes | — | Bot token from @BotFather |
| `ADMIN_USER_ID` | Yes | — | Your Telegram user ID — grants admin access and receives approval requests |
| `ALLOWED_USERS` | No | — | Comma-separated user IDs to pre-approve at startup |
| `MODEL_NAME` | No | `gemma4:26b` | Ollama model to use |
| `OLLAMA_HOST` | No | `http://localhost:11434` | Ollama server URL |
| `HISTORY_DIR` | No | `/data/history` | Path for repair history and whitelist JSON files |

---

## Model Recommendations

| Model | VRAM | Speed | Notes |
|-------|------|-------|-------|
| `gemma4:26b` | ~18 GB | ~90s | Best quality, MoE: only ~4B active per token |
| `gemma4:e4b` | ~10 GB | ~30s | Fully fits 16GB, fast, good quality |
| `qwen2.5vl:32b` | ~21 GB | ~3min | Original model, partially offloads to RAM |
| `qwen2.5vl:7b` | ~5 GB | ~25s | Fast fallback |

Gemma4 is preferred — better multilingual support (Hebrew/English), stronger JSON instruction following, and the MoE architecture means minimal speed penalty despite the large parameter count.
