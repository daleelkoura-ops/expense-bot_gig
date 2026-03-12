"""
💸 Expense Bot — Telegram Agentic Workflow
==========================================
Sends text or voice via Telegram → logs expense to your JSONBin dashboard.

Voice pipeline : Telegram .ogg → pydub/ffmpeg → .wav → Vosk (FREE, offline)
Text pipeline  : Telegram text → Gemini 2.5 Flash AI → parsed JSON → JSONBin
"""

import os
import json
import wave
import requests
import tempfile
import asyncio
from datetime import datetime
from dotenv import load_dotenv

from vosk import Model, KaldiRecognizer
from pydub import AudioSegment

from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    filters, ContextTypes
)

# FIX 1: Replaced deprecated `google.generativeai` with `google.genai`
from google import genai
from google.genai import types

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — loads from .env file automatically
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv()

TELEGRAM_TOKEN     = os.getenv("TELEGRAM_TOKEN")
GEMINI_API_KEY     = os.getenv("GEMINI_API_KEY")
JSONBIN_MASTER_KEY = os.getenv("JSONBIN_MASTER_KEY")
JSONBIN_BIN_ID     = os.getenv("JSONBIN_BIN_ID")

# Validate all keys are present
missing = [k for k, v in {
    "TELEGRAM_TOKEN":     TELEGRAM_TOKEN,
    "GEMINI_API_KEY":     GEMINI_API_KEY,
    "JSONBIN_MASTER_KEY": JSONBIN_MASTER_KEY,
    "JSONBIN_BIN_ID":     JSONBIN_BIN_ID,
}.items() if not v]

if missing:
    raise EnvironmentError(
        f"❌ Missing environment variables: {', '.join(missing)}\n"
        "Check your .env file."
    )

JSONBIN_URL = f"https://api.jsonbin.io/v3/b/{JSONBIN_BIN_ID}"
CATEGORIES  = ["Housing","Food","Transport","Health","Entertainment","Shopping","Utilities","Other"]
MONTHS      = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

# ─────────────────────────────────────────────────────────────────────────────
# CLIENTS
# ─────────────────────────────────────────────────────────────────────────────

# FIX 1 (continued): New google.genai client instantiation
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# FIX 2: Graceful Vosk model loading — clear error if model folder is missing
_VOSK_MODEL_PATH = os.path.join(os.path.dirname(__file__), "vosk-model-en-us-0.22-lgraph")
if not os.path.isdir(_VOSK_MODEL_PATH):
    raise RuntimeError(
        f"❌ Vosk model not found at: {_VOSK_MODEL_PATH}\n\n"
        "Download it by running:\n"
        "  wget https://alphacephei.com/vosk/models/vosk-model-en-us-0.22-lgraph.zip\n"
        "  unzip vosk-model-en-us-0.22-lgraph.zip -d /app/\n\n"
        "Or add these lines to your Dockerfile:\n"
        "  RUN wget https://alphacephei.com/vosk/models/vosk-model-en-us-0.22-lgraph.zip && \\\n"
        "      unzip vosk-model-en-us-0.22-lgraph.zip -d /app/ && \\\n"
        "      rm vosk-model-en-us-0.22-lgraph.zip"
    )
VOSK_MODEL = Model(_VOSK_MODEL_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# JSONBIN — Cloud Database Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_expenses() -> list:
    """Fetch all expenses from JSONBin cloud."""
    try:
        res = requests.get(
            f"{JSONBIN_URL}/latest",
            headers={"X-Master-Key": JSONBIN_MASTER_KEY},
            timeout=10
        )
        res.raise_for_status()
        return res.json().get("record", {}).get("expenses", [])
    except requests.exceptions.Timeout:
        raise Exception("JSONBin request timed out. Check your internet connection.")
    except requests.HTTPError as e:
        code = e.response.status_code
        if code == 401:
            raise Exception("JSONBin auth failed — check your JSONBIN_MASTER_KEY.")
        elif code == 404:
            raise Exception("Bin not found — check your JSONBIN_BIN_ID.")
        raise Exception(f"JSONBin error {code}")


def save_expenses(expenses: list) -> None:
    """Write updated expenses list back to JSONBin."""
    try:
        res = requests.put(
            JSONBIN_URL,
            json={"expenses": expenses},
            headers={
                "Content-Type": "application/json",
                "X-Master-Key": JSONBIN_MASTER_KEY
            },
            timeout=10
        )
        res.raise_for_status()
    except requests.exceptions.Timeout:
        raise Exception("Save timed out. Try again.")
    except requests.HTTPError as e:
        raise Exception(f"JSONBin save failed ({e.response.status_code})")


# ─────────────────────────────────────────────────────────────────────────────
# VOICE → TEXT (Vosk — Free, Offline, No API Key Needed)
# ─────────────────────────────────────────────────────────────────────────────

async def transcribe_voice(ogg_path: str) -> str:
    """
    Convert Telegram .ogg voice file → .wav → text using Vosk.
    Completely free and works offline. No API key required.
    Requires ffmpeg installed on the system.
    """
    loop = asyncio.get_event_loop()

    def _run():
        wav_path = ogg_path.replace(".ogg", ".wav")
        try:
            # Step 1: Convert ogg/opus → wav with improved audio settings
            audio = AudioSegment.from_file(ogg_path, format="ogg")
            audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
            audio = audio.normalize()
            audio.export(wav_path, format="wav", parameters=["-ar", "16000"])

            # Step 2: Transcribe with Vosk (offline, free)
            wf  = wave.open(wav_path, "rb")
            rec = KaldiRecognizer(VOSK_MODEL, wf.getframerate())
            rec.SetWords(True)

            results = []
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if rec.AcceptWaveform(data):
                    results.append(json.loads(rec.Result())["text"])
            results.append(json.loads(rec.FinalResult())["text"])
            wf.close()

            transcript = " ".join(r for r in results if r).strip()
            if not transcript:
                raise ValueError(
                    "Could not understand the audio.\n\n"
                    "Tips:\n"
                    "• Speak clearly and slowly\n"
                    "• Hold phone close to your mouth\n"
                    "• Avoid background noise\n"
                    "• Try sending a text message instead"
                )
            return transcript

        finally:
            # Always clean up wav file
            if os.path.exists(wav_path):
                os.remove(wav_path)

    return await loop.run_in_executor(None, _run)


# ─────────────────────────────────────────────────────────────────────────────
# GEMINI AI — Parse Natural Language → Structured Expense
# ─────────────────────────────────────────────────────────────────────────────

def parse_expense(text: str) -> dict:
    """
    Use Gemini 2.5 Flash to extract structured expense data from natural language.
    Input : "I spent 45 bucks on groceries"
    Output: { "desc": "Grocery run", "amount": 45.0, "category": "Food", "month": 2 }
    """
    today = datetime.now()

    prompt = f"""Extract expense details from the message and return ONLY valid JSON.
No explanation. No markdown fences. No extra text. Pure JSON only.

Message: "{text}"
Today's date: {today.strftime("%B %d, %Y")}
Current month index (0=Jan, 11=Dec): {today.month - 1}
Valid categories: {', '.join(CATEGORIES)}

Return exactly this JSON shape:
{{
  "desc": "short clear description",
  "amount": 123.45,
  "category": "one of the valid categories",
  "month": {today.month - 1}
}}

Rules:
- amount must be a plain number, no $ sign, no commas
- month is 0-indexed; use current month unless user says otherwise
- category: pick the single closest match from the valid list
- desc: concise (max 5 words), e.g. "Monthly rent", "Grocery run", "Netflix sub"
- If no amount found, return amount as 0
"""

    # FIX 1 (continued): Updated to new google.genai API call syntax
    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0,
        )
    )
    raw = response.text.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


# ─────────────────────────────────────────────────────────────────────────────
# CORE WORKFLOW — Parse + Save + Confirm
# ─────────────────────────────────────────────────────────────────────────────

async def process_expense(update: Update, text: str) -> None:
    """
    Full pipeline:
    text input → Gemini parse → JSONBin append → Telegram confirmation
    """
    try:
        # Step 1: Parse with Gemini
        status_msg = await update.message.reply_text("🤖 Parsing your expense…")
        expense    = parse_expense(text)

        # Validate amount
        if not expense.get("amount") or expense["amount"] <= 0:
            await status_msg.edit_text(
                "❌ I couldn't find an amount in that message.\n\n"
                "Try: _'Spent $45 on groceries'_ or _'Rent 1200'_",
                parse_mode="Markdown"
            )
            return

        # Step 2: Give expense a unique ID
        expense["id"] = int(datetime.now().timestamp() * 1000)

        # Step 3: Fetch current, append new, save back
        await status_msg.edit_text("☁️ Saving to cloud…")
        expenses = get_expenses()
        expenses.append(expense)
        save_expenses(expenses)

        # Step 4: Confirm to user
        month_name = MONTHS[expense["month"]]
        await status_msg.edit_text(
            f"✅ *Expense logged!*\n\n"
            f"📌 *{expense['desc']}*\n"
            f"💰 Amount  : ${expense['amount']:,.2f}\n"
            f"🏷️ Category: {expense['category']}\n"
            f"📅 Month   : {month_name}\n\n"
            f"_Open your dashboard to see it live_ 🚀",
            parse_mode="Markdown"
        )

    except json.JSONDecodeError:
        await update.message.reply_text(
            "❌ Couldn't parse that as an expense.\n\n"
            "*Try these formats:*\n"
            "• _Spent $45 on groceries_\n"
            "• _Paid $1,200 rent_\n"
            "• _Netflix $15_\n"
            "• _Coffee 4.50_",
            parse_mode="Markdown"
        )
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {str(e)}")


# ─────────────────────────────────────────────────────────────────────────────
# TELEGRAM COMMAND HANDLERS
# ─────────────────────────────────────────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    keyboard     = [["/summary", "/recent", "/help"]]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    await update.message.reply_text(
        "👋 *Welcome to your Expense Bot!*\n\n"
        "I log your expenses automatically — just tell me what you spent.\n\n"
        "*📝 Text examples:*\n"
        "• _Spent $50 on groceries_\n"
        "• _Paid $1200 rent_\n"
        "• _Coffee $4.50_\n"
        "• _Electricity bill $95 last month_\n\n"
        "*🎙️ Voice:* Just hold the mic button and speak naturally!\n\n"
        "*Commands:*\n"
        "/summary — This month's spending breakdown\n"
        "/recent  — Last 5 expenses\n"
        "/help    — Show this message",
        parse_mode="Markdown",
        reply_markup=reply_markup
    )


async def cmd_summary(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        await update.message.reply_text("📊 Fetching your summary…")
        expenses      = get_expenses()
        current_month = datetime.now().month - 1
        filtered      = [e for e in expenses if e.get("month") == current_month]
        month_name    = MONTHS[current_month]

        if not filtered:
            await update.message.reply_text(
                f"📭 No expenses logged for *{month_name}* yet.\n\n"
                f"Send me a message like _'Spent $50 on groceries'_ to get started!",
                parse_mode="Markdown"
            )
            return

        by_cat = {}
        for e in filtered:
            cat = e.get("category", "Other")
            by_cat[cat] = by_cat.get(cat, 0) + e["amount"]

        total = sum(e["amount"] for e in filtered)
        lines = [f"📊 *{month_name} Breakdown*\n"]
        for cat, amt in sorted(by_cat.items(), key=lambda x: -x[1]):
            bar = "█" * min(int(amt / total * 10), 10)
            lines.append(f"`{bar:<10}` {cat}: *${amt:,.2f}*")

        lines.append(f"\n💰 *Total Spent : ${total:,.2f}*")
        lines.append(f"📝 Transactions: {len(filtered)}")
        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

    except Exception as e:
        await update.message.reply_text(f"❌ Error: {str(e)}")


async def cmd_recent(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        expenses = get_expenses()
        if not expenses:
            await update.message.reply_text("📭 No expenses logged yet.")
            return
        recent = expenses[-5:][::-1]
        lines  = ["🕐 *Last 5 Expenses*\n"]
        for e in recent:
            lines.append(
                f"• {e['desc']} — *${e['amount']:,.2f}*\n"
                f"  {e['category']} · {MONTHS[e['month']]}"
            )
        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {str(e)}")


# ─────────────────────────────────────────────────────────────────────────────
# TELEGRAM MESSAGE HANDLERS
# ─────────────────────────────────────────────────────────────────────────────

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle plain text expense messages."""
    await process_expense(update, update.message.text)


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle voice messages: download → transcribe (Vosk) → process."""
    status_msg = await update.message.reply_text("🎙️ Voice received! Transcribing…")
    tmp_path   = None

    try:
        voice_file = await context.bot.get_file(update.message.voice.file_id)

        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
            tmp_path = tmp.name

        await voice_file.download_to_drive(tmp_path)

        transcript = await transcribe_voice(tmp_path)

        await status_msg.edit_text(
            f"📝 *I heard:*\n_{transcript}_",
            parse_mode="Markdown"
        )

        await process_expense(update, transcript)

    except ValueError as e:
        await status_msg.edit_text(f"❌ {str(e)}")
    except Exception as e:
        await status_msg.edit_text(f"❌ Voice error: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — Start the Bot
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("🤖 Expense Bot starting…")
    print(f"📦 Bin ID   : {JSONBIN_BIN_ID}")
    print(f"🤖 AI Model : Gemini 2.5 Flash")
    print(f"🎙️ Voice    : Vosk en-us-0.22-lgraph")

    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start",   cmd_start))
    app.add_handler(CommandHandler("help",    cmd_start))
    app.add_handler(CommandHandler("summary", cmd_summary))
    app.add_handler(CommandHandler("recent",  cmd_recent))

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))

    print("✅ Bot is live! Open Telegram and message your bot.")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
