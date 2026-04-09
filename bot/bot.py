"""
FixBot — Telegram repair assistant powered by a local Ollama vision model.
"""

import asyncio
import logging
import os
from collections import defaultdict

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from vision import analyze_image, ask_followup, detect_language
from annotator import annotate_image
from history import save_repair, get_history_text, clear_history

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]

user_sessions: dict = defaultdict(lambda: {
    "history":      [],
    "last_images":  [],
    "last_analysis": None,
    "album_buffer": {},
    "lang":         "en",   # detected from user's first message
})

MAX_HISTORY = 6
ALBUM_WAIT  = 2.5
NEW_CONV_CB = "new_conversation"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[
        InlineKeyboardButton("🔄 שיחה חדשה / New Conversation", callback_data=NEW_CONV_CB)
    ]])


def _clear_session(uid: int) -> None:
    user_sessions[uid] = {
        "history":       [],
        "last_images":   [],
        "last_analysis": None,
        "album_buffer":  {},
        "lang":          "en",
    }


def _push_history(session: dict, role: str, content: str) -> None:
    session["history"].append({"role": role, "content": content})
    if len(session["history"]) > MAX_HISTORY * 2:
        session["history"] = session["history"][-(MAX_HISTORY * 2):]


# ── Message formatting ────────────────────────────────────────────────────────

def _format(analysis: dict, lang: str) -> str:
    sev_emoji = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(
        analysis.get("severity", "medium"), "🟡"
    )
    lines = []

    # Problem summary
    lines.append(f"{sev_emoji} {analysis.get('problem_summary', '')}\n")

    # Safety warnings
    warnings = analysis.get("safety_warnings", [])
    if warnings:
        lines.append(f"⚠️ *{'אזהרות בטיחות' if lang == 'he' else 'Safety warnings'}*")
        for w in warnings:
            lines.append(f"• {w}")
        lines.append("")

    # Tools + materials
    tools     = analysis.get("tools_needed", [])
    materials = analysis.get("materials_needed", [])
    if tools:
        lines.append(f"🔧 *{'כלים' if lang == 'he' else 'Tools'}:* {', '.join(tools)}")
    if materials:
        lines.append(f"📦 *{'חומרים' if lang == 'he' else 'Materials'}:* {', '.join(materials)}")
    if tools or materials:
        lines.append("")

    # Components (what the dots point to)
    components = analysis.get("components", [])
    if components:
        lines.append(f"📍 *{'חלקים בתמונה' if lang == 'he' else 'Parts in image'}*")
        for c in components:
            lines.append(f"  {c.get('id', '?')}. {c.get('name', '')}")
        lines.append("")

    # Repair steps
    steps = analysis.get("steps", [])
    if steps:
        lines.append(f"📋 *{'שלבי התיקון' if lang == 'he' else 'Repair steps'}*\n")
        for s in steps:
            lines.append(f"*{s.get('step', '?')}. {s.get('title', '')}*")
            lines.append(f"{s.get('instruction', '')}\n")

    # Professional advice
    advice = analysis.get("professional_advice", "")
    if advice:
        lines.append(f"💡 {advice}\n")

    lines.append("❓ " + ("שאלו אותי שאלות המשך." if lang == "he" else "Ask me any follow-up questions."))
    return "\n".join(lines)


def _status(key: str, lang: str) -> str:
    msgs = {
        "analyzing":   ("🔍 מנתח… זה עשוי לקחת 20–40 שניות.",  "🔍 Analyzing… this may take 20–40 seconds."),
        "diagnosing":  ("🧠 מאבחן את הבעיה…",                   "🧠 Diagnosing the problem…"),
        "marking":     ("🖊️ מסמן אזורים על התמונה…",            "🖊️ Marking key areas on the image…"),
        "thinking":    ("💬 חושב…",                               "💬 Thinking…"),

        "followup":    ("❓ שאלות המשך?",                         "❓ Follow-up questions?"),
        "no_photo":    ("📸 שלחו תמונה קודם.",                    "📸 Please send a photo first."),
        "dl_error":    ("❌ לא הצלחתי להוריד את התמונה.",         "❌ Couldn't download your image."),
        "model_error": ("⚠️ המודל אינו זמין.",                    "⚠️ The AI model isn't ready."),
        "send_error":  ("⚠️ שגיאה בשליחה. הנה הניתוח:\n\n",      "⚠️ Couldn't send image. Here's the analysis:\n\n"),
        "qa_error":    ("⚠️ לא הצלחתי לעבד את השאלה.",           "⚠️ Couldn't process your question."),
        "new_conv":    ("🔄 שיחה חדשה! שלחו תמונה להתחלה.",       "🔄 New conversation! Send a photo to begin."),
        "cleared":     ("🔄 שיחה חדשה התחילה.",                   "🔄 New conversation started."),
        "album_cap":   ("📸 {n} תמונות התקבלו — משתמש בשתיים הראשונות.",
                        "📸 Received {n} photos — using the first 2."),
    }
    he, en = msgs[key]
    return he if lang == "he" else en


# ── Commands ──────────────────────────────────────────────────────────────────

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "👋 *FixBot*\n\n"
        "🇮🇱 שלחו תמונה של פריט פגום ואני אאבחן ואתן הוראות תיקון.\n"
        "🇬🇧 Send a photo of a broken item and I'll diagnose and provide repair instructions.\n\n"
        "*/help* • */history* • */new\\_conversation*",
        parse_mode=ParseMode.MARKDOWN,
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    lang = user_sessions[update.effective_user.id]["lang"]
    if lang == "he":
        text = (
            "💡 *טיפים לתוצאות טובות*\n\n"
            "• צלמו תמונות ברורות ומוארות\n"
            "• התקרבו לאזור הבעיה\n"
            "• שלחו כמה תמונות כאלבום לבעיות מורכבות\n"
            "• הוסיפו כיתוב כמו 'דולף מלמטה' או 'משמיע רעש'\n\n"
            "*סוגי תיקון:* מכשירי חשמל • אינסטלציה • רכב • תיקונים כלליים"
        )
    else:
        text = (
            "💡 *Tips for best results*\n\n"
            "• Take clear, well-lit photos\n"
            "• Get close to the problem area\n"
            "• Send multiple photos as an album for complex issues\n"
            "• Add a caption like 'leaking from the bottom' or 'makes a noise'\n\n"
            "*Supported:* Appliances • Plumbing • Car mechanics • General repairs"
        )
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)


async def history_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid  = update.effective_user.id
    lang = user_sessions[uid]["lang"]
    await update.message.reply_text(get_history_text(uid, lang), parse_mode=ParseMode.MARKDOWN)


async def clear_history_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid   = update.effective_user.id
    lang  = user_sessions[uid]["lang"]
    count = clear_history(uid)
    if lang == "he":
        msg = f"🗑 נמחקו {count} רשומות." if count else "📭 ההיסטוריה כבר ריקה."
    else:
        msg = f"🗑 Deleted {count} repair record(s)." if count else "📭 History is already empty."
    await update.message.reply_text(msg)


async def new_conversation_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    lang = user_sessions[update.effective_user.id]["lang"]
    _clear_session(update.effective_user.id)
    await update.message.reply_text(_status("cleared", lang), parse_mode=ParseMode.MARKDOWN)


async def new_conversation_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    lang = user_sessions[query.from_user.id]["lang"]
    _clear_session(query.from_user.id)
    try:
        await query.edit_message_reply_markup(reply_markup=None)
    except Exception:
        pass
    await query.message.reply_text(_status("cleared", lang))


# ── Core analysis pipeline ────────────────────────────────────────────────────

async def _run_analysis(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    image_bytes_list: list[bytes],
    caption: str,
) -> None:
    uid     = update.effective_user.id
    session = user_sessions[uid]
    lang    = session["lang"]

    status_msg = await update.message.reply_text(_status("analyzing", lang))

    try:
        await status_msg.edit_text(_status("diagnosing", lang))
        analysis = await asyncio.get_event_loop().run_in_executor(
            None, analyze_image, image_bytes_list, caption, session["history"], lang
        )
    except Exception as e:
        logger.error(f"Vision analysis failed: {e}")
        await status_msg.edit_text(_status("model_error", lang) + f"\n`{str(e)[:150]}`",
                                   parse_mode=ParseMode.MARKDOWN)
        return



    session["last_images"]   = image_bytes_list
    session["last_analysis"] = analysis
    _push_history(session, "user",      f"[photo]{' — ' + caption if caption else ''}")
    _push_history(session, "assistant", analysis.get("problem_summary", ""))
    save_repair(uid, analysis)

    annotated = None
    if analysis.get("components"):
        try:
            await status_msg.edit_text(_status("marking", lang))
            annotated = await asyncio.get_event_loop().run_in_executor(
                None, annotate_image, image_bytes_list[0], analysis
            )
        except Exception as e:
            logger.error(f"Annotation failed: {e}", exc_info=True)

    reply_text = _format(analysis, lang)
    keyboard   = _kb()

    try:
        await status_msg.delete()
    except Exception:
        pass

    # Always send the text analysis first — never lose it
    try:
        await update.message.reply_text(
            reply_text, parse_mode=ParseMode.MARKDOWN, reply_markup=keyboard
        )
    except Exception as e:
        logger.error(f"Failed to send text reply: {e}")
        # Strip markdown and try plain text as last resort
        try:
            await update.message.reply_text(reply_text, reply_markup=keyboard)
        except Exception:
            pass

    # Then send the annotated image separately (optional)
    if annotated:
        try:
            await update.message.reply_photo(photo=annotated)
        except Exception as e:
            logger.error(f"Failed to send annotated image: {e}")


# ── Photo handler ─────────────────────────────────────────────────────────────

async def _flush_album(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    uid: int,
    media_group_id: str,
) -> None:
    session = user_sessions[uid]
    buf     = session["album_buffer"].pop(media_group_id, None)
    if not buf:
        return

    images  = buf["images"]
    caption = buf["caption"]
    lang    = session["lang"]

    if len(images) > 2:
        await update.message.reply_text(
            _status("album_cap", lang).replace("{n}", str(len(images)))
        )

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
    await _run_analysis(update, context, images, caption)


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid     = update.effective_user.id
    session = user_sessions[uid]

    # Detect language from caption if present
    caption = update.message.caption or ""
    if caption:
        session["lang"] = detect_language(caption)

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)

    photo = update.message.photo[-1]
    try:
        file        = await context.bot.get_file(photo.file_id)
        image_bytes = bytes(await file.download_as_bytearray())
    except Exception as e:
        logger.error(f"Failed to download photo: {e}")
        await update.message.reply_text(_status("dl_error", session["lang"]))
        return

    media_group_id = update.message.media_group_id
    if media_group_id:
        buf = session["album_buffer"]
        if media_group_id not in buf:
            buf[media_group_id] = {"images": [], "caption": caption}
        buf[media_group_id]["images"].append(image_bytes)

        existing = buf[media_group_id].get("task")
        if existing:
            existing.cancel()

        async def delayed_flush():
            await asyncio.sleep(ALBUM_WAIT)
            await _flush_album(update, context, uid, media_group_id)

        buf[media_group_id]["task"] = asyncio.create_task(delayed_flush())
    else:
        await _run_analysis(update, context, [image_bytes], caption)


# ── Text handler ──────────────────────────────────────────────────────────────

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid      = update.effective_user.id
    session  = user_sessions[uid]
    question = update.message.text.strip()

    # Update language from whatever the user just typed
    session["lang"] = detect_language(question)
    lang = session["lang"]

    if not session["last_images"]:
        await update.message.reply_text(_status("no_photo", lang))
        return

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
    status_msg = await update.message.reply_text(_status("thinking", lang))

    try:
        answer = await asyncio.get_event_loop().run_in_executor(
            None, ask_followup, session["last_images"], question, session["history"], lang
        )
    except Exception as e:
        logger.error(f"Follow-up failed: {e}")
        await status_msg.edit_text(_status("qa_error", lang))
        return

    _push_history(session, "user",      question)
    _push_history(session, "assistant", answer)

    await status_msg.edit_text(answer, parse_mode=ParseMode.MARKDOWN, reply_markup=_kb())


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    app = (
        Application.builder()
        .token(TELEGRAM_TOKEN)
        .read_timeout(30)
        .write_timeout(30)
        .connect_timeout(15)
        .pool_timeout(30)
        .build()
    )

    app.add_handler(CommandHandler("start",            start))
    app.add_handler(CommandHandler("help",             help_command))
    app.add_handler(CommandHandler("history",          history_command))
    app.add_handler(CommandHandler("clear_history",    clear_history_command))
    app.add_handler(CommandHandler("new_conversation", new_conversation_command))
    app.add_handler(CallbackQueryHandler(new_conversation_callback, pattern=f"^{NEW_CONV_CB}$"))
    app.add_handler(MessageHandler(filters.PHOTO,                   handle_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    logger.info("FixBot running…")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
