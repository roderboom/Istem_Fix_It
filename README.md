# 🔧 FixBot — AI Repair Assistant for Telegram

A Telegram bot that analyzes photos of broken items and returns step-by-step repair instructions with annotated images. Runs **100% locally** — no cloud services, no API costs, no data leaves your machine.

---

## Active Files

```
fixbot/
├── compose.yaml          — Podman Compose: Ollama + bot containers
├── .env.example          — Environment variable template
└── bot/
    ├── Containerfile     — Container image definition
    ├── requirements.txt  — Python dependencies
    ├── bot.py            — Telegram handlers, session state, photo/album/text routing
    ├── vision.py         — Ollama API client, prompts, JSON parsing, image resizing
    ├── annotator.py      — Draws numbered dots on images (Pillow, EXIF-aware)
    ├── history.py        — Persistent per-user repair history (JSON files)
    └── whitelist.py      — User whitelist with admin approval flow
```

**Safe to delete:**
- `bot/searcher.py` — web search feature (removed)
- `bot/translator.py` — translation layer (removed, model handles Hebrew natively)
- `setup.sh` — outdated setup script

---

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 10 GB | 16 GB |
| RAM | 32 GB | 64 GB+ |
| OS | Linux | Ubuntu 22.04 / 24.04 |
| NVIDIA driver | 520+ | latest |
| Podman | 4.0+ | 4.9+ |

---

## Setup

### 1 — Install Podman

```bash
sudo apt-get update && sudo apt-get install -y podman
```

### 2 — Install podman-compose

```bash
pip install podman-compose
```

### 3 — NVIDIA Container Toolkit

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
```

### 4 — Configure CDI (GPU access for containers)

```bash
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml

# Verify:
podman run --rm --device nvidia.com/gpu=all --security-opt=label=disable \
  docker.io/nvidia/cuda:12.3.0-base-ubuntu22.04 nvidia-smi -L
```

### 5 — Create your Telegram bot

1. Open Telegram → search `@BotFather`
2. Send `/newbot` → follow the prompts
3. Copy the token (`7123456789:AAFxxxxx...`)

### 6 — Configure environment

```bash
cp .env.example .env
nano .env
# Set: TELEGRAM_TOKEN=your_token_here
```

### 7 — Build and start

```bash
podman-compose build
podman-compose up -d ollama
sleep 15
podman exec fixbot-ollama ollama pull gemma4:26b
podman-compose up -d fixbot
```

### 8 — Verify

```bash
podman-compose logs -f fixbot
# Should show: INFO | FixBot running…
```

---

## Bot Commands

| Command | Description |
|---------|-------------|
| `/start` | Welcome message — triggers access request if not approved |
| `/help` | Tips for good repair photos |
| `/history` | Last 10 repairs |
| `/clear_history` | Delete repair history |
| `/new_conversation` | Clear session and start fresh |
| `/adduser <id>` | *(admin)* Approve a user |
| `/removeuser <id>` | *(admin)* Revoke a user |
| `/listusers` | *(admin)* List all approved users |

---

## Access Control

FixBot uses a whitelist — only approved users can interact with it.

**Setup:** set `ADMIN_USER_ID` in your `.env` (get your ID from `@userinfobot` on Telegram).

**Approval flow:**
1. A new user sends `/start`
2. Bot tells them their request was sent, then notifies you (admin) with **✅ Allow** / **❌ Deny** buttons
3. You tap a button — user is instantly approved or denied and notified

**Pre-approve users** at startup via `ALLOWED_USERS=id1,id2` in `.env`.

The whitelist is stored in `/data/history/whitelist.json` and survives container restarts.

---

## Recommended Models (16 GB VRAM)

| Model | Download | Speed | Quality |
|-------|----------|-------|---------|
| `gemma4:26b` | 18 GB | ~90s | Best |
| `gemma4:e4b` | 9.6 GB | ~30s | Good |
| `qwen2.5vl:7b` | 5 GB | ~25s | Good |

Change model in `compose.yaml` → `MODEL_NAME=<model>`.

---

## Useful Commands

```bash
podman-compose logs -f fixbot              # live logs
podman-compose restart fixbot              # restart after copying updated files
podman-compose build fixbot && podman-compose up -d fixbot  # full rebuild
podman exec fixbot-ollama ollama list      # list downloaded models
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Bot doesn't respond | `podman-compose logs fixbot` |
| GPU not accessible | `sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml` |
| Model not found | `podman exec fixbot-ollama ollama pull gemma4:26b` |
| JSON parse / truncated responses | Increase `num_predict` in `vision.py` analyze_image options |
| Hebrew responses cut off | Hebrew uses 2-4x more tokens — `num_predict` is set to 6000 in `vision.py` |
