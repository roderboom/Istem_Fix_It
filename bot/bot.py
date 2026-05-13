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

from vision import analyze_image, ask_followup, detect_language, get_annotations_for_image
import whitelist as wl
from whitelist import is_banned, ban_user, unban_user, was_ban_notified, mark_ban_notified
from annotator import annotate_image
from history import save_repair, get_history_text, clear_history

# Global queue — one Ollama call at a time
# Users who sent /start and are waiting for admin approval
_pending_approval: set[int] = set()
APPROVE_CB   = "approve"
DENY_CB      = "deny"
BAN_CB       = "ban"
ALLOWMORE_CB = "allowmore"

# Per-user message counters and pending-alert tracking
_msg_counts:     dict[int, int]  = {}   # uid → messages since last reset
_limit_alerted:  set[int]        = set() # uids where admin was already alerted (awaiting decision)
MSG_LIMIT = 3

_ollama_sem     = asyncio.Semaphore(1)
_queue_waiters  = 0   # number of requests currently waiting

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

async def _check_ban(update: Update) -> bool:
    """Return True if user is banned (and handle the one-time message). Handler should return immediately if True."""
    uid = update.effective_user.id
    if not is_banned(uid):
        return False
    if not was_ban_notified(uid):
        mark_ban_notified(uid)
        try:
            if update.message:
                await update.message.reply_text("🚫 You have been banned from using this bot.")
        except Exception:
            pass
    return True  # silently ignore all future messages


async def _check_limit(update: Update, context: ContextTypes.DEFAULT_TYPE, lang: str) -> bool:
    """Increment message counter. If limit hit, alert admin and block. Return True if blocked."""
    uid  = update.effective_user.id
    if wl.is_admin(uid):
        return False  # admin is never rate-limited

    _msg_counts[uid] = _msg_counts.get(uid, 0) + 1
    count = _msg_counts[uid]

    if count <= MSG_LIMIT:
        return False  # still within limit

    # Over limit — block the message
    if uid not in _limit_alerted and wl.ADMIN_USER_ID:
        # First time over limit — alert admin
        _limit_alerted.add(uid)
        user     = update.effective_user
        name     = user.full_name or 'Unknown'
        username = f' (@{user.username})' if user.username else ''
        keyboard = InlineKeyboardMarkup([[
            InlineKeyboardButton("✅ Allow 3 more", callback_data=f"{ALLOWMORE_CB}:{uid}"),
            InlineKeyboardButton("🚫 Ban",          callback_data=f"{BAN_CB}:{uid}"),
        ]])
        try:
            await context.bot.send_message(
                chat_id=wl.ADMIN_USER_ID,
                text=f"⚠️ *Usage limit reached*\n\n"
                     f"User: {name}{username}\n"
                     f"ID: `{uid}`\n"
                     f"Messages sent: {count}\n\n"
                     f"Allow more or ban?",
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=keyboard,
            )
        except Exception as e:
            logger.warning(f'Could not alert admin about limit: {e}')

    # Tell user they're waiting for admin
    try:
        if update.message:
            msg = ('⏸ הגעת למגבלת ההודעות. ממתין לאישור מנהל.' if lang == 'he'
                   else '⏸ You have reached your message limit. Waiting for admin approval.')
            await update.message.reply_text(msg)
    except Exception:
        pass
    return True


def _unauthorized(uid: int) -> str:
    if uid in _pending_approval:
        return "⏳ Your request has been sent to the admin. Please wait for approval."
    return "🚫 You are not authorized to use this bot."


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

    # Components (dots) and areas (boxes)
    components = analysis.get("components", [])
    areas      = analysis.get("areas", [])
    if components or areas:
        label = "📍 *סימונים בתמונה*" if lang == "he" else "📍 *Annotations in image*"
        lines.append(label)
        for c in components:
            lines.append(f"  🔴 {c.get('id', '?')}. {c.get('name', '')}")
        for a in areas:
            lines.append(f"  🟦 {a.get('id', '?')}. {a.get('name', '')}")
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
    uid  = update.effective_user.id
    user = update.effective_user

    if not wl.is_allowed(uid):
        # Notify admin with approve/deny buttons
        if wl.ADMIN_USER_ID and uid not in _pending_approval:
            _pending_approval.add(uid)
            name     = user.full_name or "Unknown"
            username = f" (@{user.username})" if user.username else ""
            keyboard = InlineKeyboardMarkup([[
                InlineKeyboardButton("✅ Allow",  callback_data=f"{APPROVE_CB}:{uid}"),
                InlineKeyboardButton("❌ Deny",   callback_data=f"{DENY_CB}:{uid}"),
            ]])
            try:
                await context.bot.send_message(
                    chat_id=wl.ADMIN_USER_ID,
                    text=f"🔔 *Access request*\n\n"
                         f"Name: {name}{username}\n"
                         f"ID: `{uid}`\n\n"
                         f"Allow this user?",
                    parse_mode=ParseMode.MARKDOWN,
                    reply_markup=keyboard,
                )
            except Exception as e:
                logger.warning(f"Could not notify admin: {e}")
        await update.message.reply_text(
            "👋 Hi! Your access request has been sent to the admin.\n"
            "You will be notified here once approved."
        )
        return

    is_admin = wl.is_admin(uid)
    if is_admin:
        commands = (
            "📸 Send a photo to start a repair diagnosis\n\n"
            "*Commands:*\n"
            "/help — tips for good photos\n"
            "/history — your last 10 repairs\n"
            "/clear\\_history — delete repair history\n"
            "/new\\_conversation — clear session and start fresh\n\n"
            "*Admin commands:*\n"
            "/adduser <id> — approve a user\n"
            "/removeuser <id> — revoke a user\n"
            "/listusers — list all approved users\n"
            "/ban <id> — ban a user\n"
            "/unban <id> — unban a user"
        )
    else:
        commands = (
            "📸 Send a photo to start a repair diagnosis\n\n"
            "*Commands:*\n"
            "/help — tips for good photos\n"
            "/history — your last 10 repairs\n"
            "/clear\\_history — delete repair history\n"
            "/new\\_conversation — clear session and start fresh"
        )
        if wl.is_admin(uid):
            text += (
                "\n\n*Admin commands:*\n"
                "/adduser <id> — approve a user\n"
                "/removeuser <id> — revoke a user\n"
                "/listusers — list approved users\n"
                "/ban <id> — ban a user\n"
                "/unban <id> — unban a user"
            )
    await update.message.reply_text(
        "👋 *FixBot*\n\n"
        "🇮🇱 שלחו תמונה של פריט פגום ואני אאבחן ואתן הוראות תיקון.\n"
        "🇬🇧 Send a photo of a broken item and I'll diagnose and provide repair instructions.\n\n"
        + commands,
        parse_mode=ParseMode.MARKDOWN,
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid  = update.effective_user.id
    if not wl.is_allowed(uid):
        await update.message.reply_text(_unauthorized(uid))
        return
    lang = user_sessions[update.effective_user.id]["lang"]
    if lang == "he":
        text = (
            "💡 *טיפים לתוצאות טובות*\n\n"
            "• צלמו תמונות ברורות ומוארות\n"
            "• התקרבו לאזור הבעיה\n"
            "• שלחו כמה תמונות כאלבום לבעיות מורכבות\n"
            "• הוסיפו כיתוב כמו 'דולף מלמטה' או 'משמיע רעש'\n\n"
            "*סוגי תיקון:* מכשירי חשמל • אינסטלציה • רכב • תיקונים כלליים\n\n"
            "*פקודות:*\n"
            "/help — טיפים לצילום\n"
            "/history — 10 תיקונים אחרונים\n"
            "/clear\\_history — מחיקת היסטוריה\n"
            "/new\\_conversation — התחלה מחדש"
        )
        if wl.is_admin(uid):
            text += (
                "\n\n*פקודות מנהל:*\n"
                "/adduser <id> — אישור משתמש\n"
                "/removeuser <id> — הסרת משתמש\n"
                "/listusers — רשימת משתמשים מאושרים\n"
                "/ban <id> — חסימת משתמש\n"
                "/unban <id> — ביטול חסימה"
            )
    else:
        text = (
            "💡 *Tips for best results*\n\n"
            "• Take clear, well-lit photos\n"
            "• Get close to the problem area\n"
            "• Send multiple photos as an album for complex issues\n"
            "• Add a caption like 'leaking from the bottom' or 'makes a noise'\n\n"
            "*Supported:* Appliances • Plumbing • Car mechanics • General repairs\n\n"
            "*Commands:*\n"
            "/help — tips for good photos\n"
            "/history — your last 10 repairs\n"
            "/clear\\_history — delete repair history\n"
            "/new\\_conversation — clear session and start fresh"
        )
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)


async def history_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid  = update.effective_user.id
    if not wl.is_allowed(uid):
        await update.message.reply_text(_unauthorized(uid))
        return
    lang = user_sessions[uid]["lang"]
    await update.message.reply_text(get_history_text(uid, lang), parse_mode=ParseMode.MARKDOWN)


async def clear_history_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid   = update.effective_user.id
    if not wl.is_allowed(uid):
        await update.message.reply_text(_unauthorized(uid))
        return
    lang  = user_sessions[uid]["lang"]
    count = clear_history(uid)
    if lang == "he":
        msg = f"🗑 נמחקו {count} רשומות." if count else "📭 ההיסטוריה כבר ריקה."
    else:
        msg = f"🗑 Deleted {count} repair record(s)." if count else "📭 History is already empty."
    await update.message.reply_text(msg)


async def new_conversation_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid  = update.effective_user.id
    if not wl.is_allowed(uid):
        await update.message.reply_text(_unauthorized(uid))
        return
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


# ── Ban / Allow-more callback ───────────────────────────────────────────────

async def ban_allowmore_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query    = update.callback_query
    admin_id = query.from_user.id
    await query.answer()

    if not wl.is_admin(admin_id):
        return

    action, target_str = query.data.split(':', 1)
    target = int(target_str)

    if action == BAN_CB:
        ban_user(target)
        _limit_alerted.discard(target)
        await query.edit_message_text(
            f'🚫 User `{target}` has been banned.',
            parse_mode=ParseMode.MARKDOWN,
        )
        try:
            if not was_ban_notified(target):
                mark_ban_notified(target)
                await context.bot.send_message(
                    chat_id=target,
                    text='🚫 You have been banned from using this bot.',
                )
        except Exception as e:
            logger.warning(f'Could not notify banned user {target}: {e}')

    elif action == ALLOWMORE_CB:
        # Reset counter so user gets MSG_LIMIT more messages
        _msg_counts[target] = 0
        _limit_alerted.discard(target)
        await query.edit_message_text(
            f'✅ User `{target}` allowed {MSG_LIMIT} more messages.',
            parse_mode=ParseMode.MARKDOWN,
        )
        try:
            await context.bot.send_message(
                chat_id=target,
                text=f'✅ You can send {MSG_LIMIT} more messages.',
            )
        except Exception as e:
            logger.warning(f'Could not notify user {target}: {e}')


# ── Approval callback ───────────────────────────────────────────────────────

async def approval_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query    = update.callback_query
    admin_id = query.from_user.id
    await query.answer()

    if not wl.is_admin(admin_id):
        return

    action, target_str = query.data.split(":", 1)
    target = int(target_str)

    if action == APPROVE_CB:
        # Recover name from the approval message text
        msg_text = query.message.text or ""
        import re as _re
        name_match = _re.search(r"Name: (.+?)\n", msg_text)
        display_name = name_match.group(1).strip() if name_match else ""
        wl.add_user(target, display_name)
        _pending_approval.discard(target)
        # Edit the admin message to confirm
        await query.edit_message_text(
            f"✅ User `{target}` approved.",
            parse_mode=ParseMode.MARKDOWN,
        )
        # Notify the user
        try:
            await context.bot.send_message(
                chat_id=target,
                text="✅ Your access has been approved! Send /start to begin.",
            )
        except Exception as e:
            logger.warning(f"Could not notify user {target}: {e}")

    elif action == DENY_CB:
        _pending_approval.discard(target)
        await query.edit_message_text(
            f"❌ User `{target}` denied.",
            parse_mode=ParseMode.MARKDOWN,
        )
        try:
            await context.bot.send_message(
                chat_id=target,
                text="❌ Your access request was not approved.",
            )
        except Exception as e:
            logger.warning(f"Could not notify user {target}: {e}")


# ── Admin commands ──────────────────────────────────────────────────────────

async def adduser_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid = update.effective_user.id
    if not wl.is_admin(uid):
        await update.message.reply_text("🚫 Admin only.")
        return
    if not context.args:
        await update.message.reply_text("Usage: /adduser <user_id>")
        return
    try:
        target = int(context.args[0])
    except ValueError:
        await update.message.reply_text("Invalid user ID.")
        return
    added = wl.add_user(target)
    msg = f"✅ User {target} added." if added else f"ℹ️ User {target} already allowed."
    await update.message.reply_text(msg)


async def removeuser_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid = update.effective_user.id
    if not wl.is_admin(uid):
        await update.message.reply_text("🚫 Admin only.")
        return
    if not context.args:
        await update.message.reply_text("Usage: /removeuser <user_id>")
        return
    try:
        target = int(context.args[0])
    except ValueError:
        await update.message.reply_text("Invalid user ID.")
        return
    removed = wl.remove_user(target)
    if not removed and target == wl.ADMIN_USER_ID:
        await update.message.reply_text("⛔ Cannot remove the admin.")
    elif removed:
        await update.message.reply_text(f"✅ User {target} removed.")
    else:
        await update.message.reply_text(f"ℹ️ User {target} was not in the list.")


async def ban_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid = update.effective_user.id
    if not wl.is_admin(uid):
        await update.message.reply_text('🚫 Admin only.')
        return
    if not context.args:
        await update.message.reply_text('Usage: /ban <user_id>')
        return
    try:
        target = int(context.args[0])
    except ValueError:
        await update.message.reply_text('Invalid user ID.')
        return
    if ban_user(target):
        await update.message.reply_text(f'🚫 User {target} banned.')
    else:
        await update.message.reply_text(f'ℹ️ User {target} is already banned or is the admin.')


async def unban_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid = update.effective_user.id
    if not wl.is_admin(uid):
        await update.message.reply_text('🚫 Admin only.')
        return
    if not context.args:
        await update.message.reply_text('Usage: /unban <user_id>')
        return
    try:
        target = int(context.args[0])
    except ValueError:
        await update.message.reply_text('Invalid user ID.')
        return
    if unban_user(target):
        await update.message.reply_text(f'✅ User {target} unbanned.')
        try:
            await context.bot.send_message(chat_id=target,
                text='✅ Your ban has been lifted. You can use the bot again.')
        except Exception:
            pass
    else:
        await update.message.reply_text(f'ℹ️ User {target} was not banned.')


async def listusers_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid = update.effective_user.id
    if not wl.is_admin(uid):
        await update.message.reply_text("🚫 Admin only.")
        return
    users = wl.list_users()
    if not users:
        await update.message.reply_text("No users in whitelist.")
        return
    lines = [f"👥 *Allowed users* ({len(users)}):"]
    for uid_entry, display_name in users:
        tag  = " _(admin)_" if uid_entry == wl.ADMIN_USER_ID else ""
        name = f" — {display_name}" if display_name else ""
        lines.append(f"• `{uid_entry}`{name}{tag}")
    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)


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

    global _queue_waiters

    # Counter-based queue: increment before acquiring, so others can see how many are waiting
    _queue_waiters += 1
    position = _queue_waiters

    if position > 1:
        # Someone else is already processing — show queue position
        if lang == "he":
            queue_msg = f"⏳ אתה מספר {position} בתור. ממתין…"
        else:
            queue_msg = f"⏳ You are #{position} in queue. Please wait…"
        status_msg = await update.message.reply_text(queue_msg)
    else:
        status_msg = await update.message.reply_text(_status("analyzing", lang))

    async with _ollama_sem:
        _queue_waiters -= 1
        try:
            await status_msg.edit_text(_status("diagnosing", lang))
            analysis = await asyncio.get_event_loop().run_in_executor(
                None, analyze_image, image_bytes_list, caption, session["history"], lang
            )
        except Exception as e:
            logger.error(f"Vision analysis failed: {e}")
            await status_msg.edit_text(_status("model_error", lang) + f"\n`{str(e)[:150]}`",
                                       parse_mode=ParseMode.MARKDOWN)
            _queue_waiters = max(0, _queue_waiters)
            return



    session["last_images"]   = image_bytes_list
    session["last_analysis"] = analysis
    _push_history(session, "user",      f"[photo]{' — ' + caption if caption else ''}")
    _push_history(session, "assistant", analysis.get("problem_summary", ""))
    save_repair(uid, analysis)

    # Annotate each photo
    # Single photo: use analysis components/areas directly
    # Multi-photo: run a separate small annotation call per image for precise coordinates
    annotated_photos = []
    if analysis.get("components") or analysis.get("areas"):
        try:
            await status_msg.edit_text(_status("marking", lang))
        except Exception:
            pass  # status update is cosmetic — never block annotation on it
        try:
            if len(image_bytes_list) == 1:
                # Single image — use analysis annotations directly
                ann = await asyncio.get_event_loop().run_in_executor(
                    None, annotate_image, image_bytes_list[0], analysis, 1
                )
                annotated_photos.append(ann)
            else:
                # Multiple images — brief pause so Ollama frees its connection,
                # then get fresh per-image coordinates for each photo
                await asyncio.sleep(1.5)
                problem = analysis.get("problem_summary", "")
                for idx, img_bytes in enumerate(image_bytes_list, start=1):
                    per_img = await asyncio.get_event_loop().run_in_executor(
                        None, get_annotations_for_image, img_bytes, problem, lang
                    )
                    img_analysis = dict(analysis)
                    img_analysis["components"] = per_img["components"]
                    img_analysis["areas"]      = per_img["areas"]
                    ann = await asyncio.get_event_loop().run_in_executor(
                        None, annotate_image, img_bytes, img_analysis, 1
                    )
                    annotated_photos.append(ann)
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
        try:
            await update.message.reply_text(reply_text, reply_markup=keyboard)
        except Exception:
            pass

    # Send annotated photos (one per original image)
    for ann in annotated_photos:
        try:
            await update.message.reply_photo(photo=ann)
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
    if await _check_ban(update): return
    if not wl.is_allowed(uid):
        await update.message.reply_text(_unauthorized(uid))
        return
    session = user_sessions[uid]
    if await _check_limit(update, context, session['lang']): return

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
    if await _check_ban(update): return
    if not wl.is_allowed(uid):
        await update.message.reply_text(_unauthorized(uid))
        return
    session  = user_sessions[uid]
    if await _check_limit(update, context, session['lang']): return
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
    app.add_handler(CommandHandler("adduser",          adduser_command))
    app.add_handler(CommandHandler("removeuser",       removeuser_command))
    app.add_handler(CommandHandler("listusers",        listusers_command))
    app.add_handler(CommandHandler("ban",              ban_command))
    app.add_handler(CommandHandler("unban",            unban_command))
    app.add_handler(CallbackQueryHandler(ban_allowmore_callback,
                    pattern=f'^({BAN_CB}|{ALLOWMORE_CB}):'))
    app.add_handler(CallbackQueryHandler(new_conversation_callback, pattern=f"^{NEW_CONV_CB}$"))
    app.add_handler(CallbackQueryHandler(approval_callback, pattern=f"^({APPROVE_CB}|{DENY_CB}):"))
    app.add_handler(MessageHandler(filters.PHOTO,                   handle_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    logger.info("FixBot running…")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
