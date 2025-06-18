import logging, json
from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ConversationHandler,
    ContextTypes,
    filters,
)
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get BOT_TOKEN with proper error handling
BOT_TOKEN = os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN not found in environment variables. Please check your .env file.")

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Conversation states
MENU, SIDE_PHOTO, SIDE_RESULT, SIDE_CUSTOM, TREAD_PHOTO, TREAD_RESULT, TREAD_CUSTOM = range(7)

# Callback data identifiers
CB_MENU, CB_SIDE, CB_TREAD = "menu", "side", "tread"
CB_SIDE_OK, CB_SIDE_CUSTOM = "side_ok", "side_custom"
CB_TREAD_OK, CB_TREAD_CUSTOM = "tread_ok", "tread_custom"


def build_main_menu() -> InlineKeyboardMarkup:
    """Return the main menu keyboard."""
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("Марка и модель шины", callback_data=CB_SIDE)],
        [InlineKeyboardButton("Глубина протектора", callback_data=CB_TREAD)],
    ])


async def send_main_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Always send the main menu as a new message."""
    # Acknowledge any callback so the "loading…" spinner goes away
    if update.callback_query:
        await update.callback_query.answer()
        chat_id = update.callback_query.message.chat_id
    else:
        chat_id = update.message.chat_id

    await context.bot.send_message(
        chat_id=chat_id,
        text="Чем я могу помочь?",
        reply_markup=build_main_menu()
    )
    return MENU


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handler for /start — show main menu."""
    return await send_main_menu(update, context)


async def menu_choice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle main‐menu button presses."""
    query = update.callback_query
    await query.answer()

    if query.data == CB_SIDE:
        await context.bot.send_message(query.message.chat_id, "Загрузите фотографию боковой стороны шины")
        return SIDE_PHOTO

    if query.data == CB_TREAD:
        await context.bot.send_message(
            query.message.chat_id,
            "Загрузите фотографию протектора шины.\nУбедитесь, что шина полностью в кадре"
        )
        return TREAD_PHOTO

    # CB_MENU — redisplay the menu
    return await send_main_menu(update, context)


async def side_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Process side‐view photo (placeholder)."""
    await update.message.reply_text("Обработка фотографии...")
    # TODO: image processing here
    await update.message.reply_text(
        "Марка: …\nМодель: …\nДиаметр: …",
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("OK", callback_data=CB_SIDE_OK)],
            [InlineKeyboardButton("Свой вариант", callback_data=CB_SIDE_CUSTOM)],
        ])
    )
    return SIDE_RESULT


async def side_result(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle side‐view result buttons."""
    query = update.callback_query
    await query.answer()

    if query.data == CB_SIDE_OK:
        # TODO: send results to server
        await context.bot.send_message(query.message.chat_id, "Хорошего дня!")
        return await send_main_menu(update, context)

    # CB_SIDE_CUSTOM
    await context.bot.send_message(query.message.chat_id, "Введите свой вариант марки, модели и диаметра шины:")
    return SIDE_CUSTOM


async def side_custom(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle free‐text for side‐view."""
    user_text = update.message.text
    # TODO: use user_text and send to server
    await update.message.reply_text("Спасибо! Благодаря вам модель станет лучше")
    return await send_main_menu(update, context)


async def tread_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Process tread photo (placeholder)."""
    await update.message.reply_text("Обработка фотографии...")
    # TODO: image processing here
    await update.message.reply_text(
        "Глубина протектора: …\nКоличество шин: …",
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("OK", callback_data=CB_TREAD_OK)],
            [InlineKeyboardButton("Свой вариант", callback_data=CB_TREAD_CUSTOM)],
        ])
    )
    return TREAD_RESULT


async def tread_result(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle tread result buttons."""
    query = update.callback_query
    await query.answer()

    if query.data == CB_TREAD_OK:
        # TODO: send results to server
        await context.bot.send_message(query.message.chat_id, "Хорошего дня!")
        return await send_main_menu(update, context)

    # CB_TREAD_CUSTOM
    await context.bot.send_message(query.message.chat_id, "Введите свой вариант глубины протектора и количества шин:")
    return TREAD_CUSTOM


async def tread_custom(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle free‐text for tread."""
    user_text = update.message.text
    # TODO: use user_text and send to server
    await update.message.reply_text("Спасибо! Благодаря вам модель станет лучше")
    return await send_main_menu(update, context)


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle /cancel to end conversation."""
    await update.message.reply_text("До свидания!")
    return ConversationHandler.END


def main() -> None:
    app = Application.builder().token(BOT_TOKEN).build()

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            MENU: [CallbackQueryHandler(menu_choice)],
            SIDE_PHOTO: [MessageHandler(filters.PHOTO, side_photo)],
            SIDE_RESULT: [CallbackQueryHandler(side_result)],
            SIDE_CUSTOM: [MessageHandler(filters.TEXT & ~filters.COMMAND, side_custom)],
            TREAD_PHOTO: [MessageHandler(filters.PHOTO, tread_photo)],
            TREAD_RESULT: [CallbackQueryHandler(tread_result)],
            TREAD_CUSTOM: [MessageHandler(filters.TEXT & ~filters.COMMAND, tread_custom)],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
        per_user=True,
    )

    app.add_handler(conv_handler)
    app.run_polling()


if __name__ == "__main__":
    main()
