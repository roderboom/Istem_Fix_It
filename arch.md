# 🛠️ FixBot Architecture & Implementation Guide

This document provides a comprehensive overview of the **FixBot** codebase, its architecture, and instructions for modifying or extending the system.

## 🤖 System Overview

FixBot is a Telegram-based AI assistant that analyzes photos of broken items (appliances, plumbing, vehicles, etc.) to provide detailed, localized repair instructions. It leverages a local **Ollama** instance running a vision-language model (e.g., `qwen2.5-vl` or `llama3.2-vision`) to ensure privacy and zero cost.

### 🏗️ High-Level Architecture

The system operates as a pipeline of specialized Python modules:

1.  **Telegram Interface (`bot/bot.py`)**: Man
    *   Handles Telegram bot interactions (commands, photo uploads, follow-up questions).
    *   Manages user sessions, language detection (Hebrew/English), and conversation history.
    *   Orchestrates the entire analysis pipeline.
2.  **Vision Analysis (`bot/vision.py`)**:
    *   Communicates with the local **Ollama** API.
    *   Performs two main tasks:
        *   `identify_problem`: A fast, low-token pass to identify the object and issue (used for web search queries).
        *   `analyze_image`: A deep analysis pass that returns a structured JSON object containing severity, tools, materials, steps, and part coordinates.
3.  **Web Context Injection (`bot/searcher.py`)**:
    *   Uses **Google Custom Search API** to find real-world repair guides based on the identified problem.
    *   Provides the LLM with factual "web context" to improve instruction accuracy.
4.  **Translation Layer (`bot/translator.py`)**:
    *   Uses the same Ollama instance to translate user captions (Hebrew $\leftrightarrow$ English) and model outputs (English $\to$ Hebrew).
    *   Handles batch translation of complex JSON structures to maintain efficiency.
5.  **Image Annotation (`bot/annotator.py`)**:
    *   Takes the visual coordinates (`grid_col`, `grid_row`) from the AI's JSON output.
    *   Uses **Pillow** to draw numbered, high-visibility markers on the original image to point out specific parts.
6.  **Persistence (`bot/history.py`)**:
    *   Saves completed repair analyses to disk as JSON files, organized by user ID.

---

## 📂 Project Structure

```text
.
├── bot/
│   ├── annotator.py    # Image marking logic (Pillow)
│   ├── bot.py          # Main Telegram bot logic & orchestrator
│   ├── history.py      # JSON-based repair history persistence
│   ├── searcher.py     # Google Custom Search integration
│   ├── translator.py   # LLM-powered translation (Heb/Eng)
│   ├── vision.py       # Ollama API communication & vision prompts
│   └── requirements.txt# Python dependencies
├── compose.yaml        # Container orchestration (Bot + Ollama)
├── .env                # Environment variables (Tokens, API Keys)
└── README.md           # Setup and usage instructions
```

---

## 🛠️ Development & Modification Guide

### 1. Adding New Capabilities
*   **New Analysis Steps**: If you want to add a "cost estimation" feature, modify the JSON schema in `bot/vision.py`, update the `_format` function in `bot/bot.py`, and ensure the `translator.py` handles the new field.
*   **New Communication Channels**: To support Discord or WhatsApp, you would need to replace the `python-telegram-bot` logic in `bot/bot.py` with the respective library, while keeping the `vision.py` and `searcher.py` logic intact.

### 2. Modifying the AI Behavior
*   **Prompt Engineering**: All core logic resides in `bot/vision.py` (System Prompts) and `bot/translator.py`.
*   **Prompt Tuning**: To change how the bot identifies parts, modify the `COMPONENT RULES` section within `bot/vision.py`.
*   **Model Swapping**: If you upgrade the Ollama model, update `MODEL_NAME` in `bot/vision.py`. Ensure the new model supports the required visual features (like coordinate/grid output).

### 3. Handling Languages
*   **Language Detection**: Handled automatically in `bot/bot.py` and `bot/vision.py` by checking for Hebrew Unicode ranges.
*   **Expanding Language Support**: To add Spanish, you must update:
    1.  `bot/bot.py`: Add Spanish strings to `_status` and `help_command`.
    2.  `bot/translator.py`: Update the `_translate` logic and prompt instructions.
    3.  `bot/vision.py`: Update the `_system_prompt` and `_followup_system` to include Spanish instructions and JSON schemas.

### 4. Image Processing & Performance
*   **VRAM Management**: `bot/vision.py` includes logic to resize images based on the number of photos sent (to prevent OOM errors on 16GB GPUs). Do not remove the `MAX_IMAGE_SIDE` constraints unless you have significantly more VRAM.
*   **Annotation Precision**: If the dots are appearing in the wrong place, check the `grid_col`/`grid_row` $\to$ percentage conversion logic in `bot/vision.py` and the `annotate_image` logic in `bot/annotator.py`.

### 5. Environment & Deployment
*   **API Keys**: Never commit `.env`. Always use `.env.example` as a template.
*   **Docker/Podman**: The project is designed to run in isolated containers. If adding new Python dependencies, ensure you update `bot/requirements.txt` and rebuild the container using `podman-compose build`.

---

## ⚠️ Critical Constraints
*   **JSON Integrity**: The `analyze_image` function relies on a strict JSON response. Any modification to the prompt that breaks the JSON structure will cause the bot to fail.
*   **Language Purity**: The `_strip_cjk` and `_sanitize_dict` functions in `bot/vision.py` are safety measures to prevent the model from "leaking" Chinese/Japanese/Korean characters when the user expects Hebrew or English.
*   **Concurrency**: The bot uses `asyncio` to handle multiple users. Avoid blocking the main event loop with heavy synchronous computation; use `run_in_executor` for CPU-bound tasks (like `annotator.py`).
