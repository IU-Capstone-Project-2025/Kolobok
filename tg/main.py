import logging
import base64
import requests
import io
import os
from PIL import Image

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

BOT_TOKEN = os.environ["BOT_TOKEN"]
APP_URL = "ml:8000"

TREAD_ANALYSIS_URL = f"http://{APP_URL}/api/v1/analyze_thread"
TIRE_READING_URL = f"http://{APP_URL}/api/v1/identify_tire"
TIRE_INFO_EXTRACTION_URL = f"http://{APP_URL}/api/v1/extract_information"

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Conversation states
(
    PASSWORD,
    INCORRECT_PASSWORD,
    MENU,
    SIDE_PHOTO,
    SIDE_RESULT,
    SIDE_CUSTOM,
    TREAD_PHOTO,
    TREAD_RESULT,
    TREAD_CUSTOM,
) = range(9)

# Callback data identifiers
CB_MENU, CB_SIDE, CB_TREAD = "menu", "side", "tread"
CB_SIDE_OK, CB_SIDE_CUSTOM = "side_ok", "side_custom"
CB_TREAD_OK, CB_TREAD_CUSTOM = "tread_ok", "tread_custom"


def build_main_menu() -> InlineKeyboardMarkup:
    """Return the main menu keyboard."""
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "Глубина протектора и анализ шипов", callback_data=CB_TREAD
                )
            ],
            [InlineKeyboardButton("Марка и модель шины", callback_data=CB_SIDE)],
        ]
    )


async def send_main_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Sends the main menu as a new message."""
    # Acknowledge any callback so the “loading…” spinner goes away
    if update.callback_query:
        await update.callback_query.answer()
        chat_id = update.callback_query.message.chat_id
    else:
        chat_id = update.message.chat_id

    await context.bot.send_message(
        chat_id=chat_id, text="Чем я могу помочь?", reply_markup=build_main_menu()
    )
    return MENU


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("Введите пароль:")
    return PASSWORD


async def get_password(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    token = update.message.text
    true_token = os.environ["API_TOKEN"]
    if token != true_token:
        return await incorrect_password(update, context)
    else:
        context.user_data["token"] = f"Bearer {token}"
        return await send_main_menu(update, context)


async def incorrect_password(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("Неверный пароль, попробуйте ещё раз:")
    return PASSWORD


async def menu_choice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle main‐menu button presses."""
    query = update.callback_query
    await query.answer()

    if query.data == CB_SIDE:
        await context.bot.send_message(
            query.message.chat_id, "Загрузите фотографию боковой стороны шины"
        )
        return SIDE_PHOTO

    if query.data == CB_TREAD:
        await context.bot.send_message(
            query.message.chat_id,
            "Загрузите фотографию протектора шины.\nУбедитесь, что шина полностью в кадре",
        )
        return TREAD_PHOTO

    # CB_MENU — redisplay the menu
    return await send_main_menu(update, context)


async def side_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("Обработка фотографии...")

    photo = update.message.photo[-1]
    tg_file = await photo.get_file()

    bio = io.BytesIO()
    await tg_file.download_to_memory(out=bio)
    bio.seek(0)

    b64 = base64.b64encode(bio.read()).decode("utf-8")
    header = {"Authorization": context.user_data["token"]}
    payload = {"image": b64}

    resp = requests.post(TIRE_INFO_EXTRACTION_URL, headers=header, json=payload)
    resp.raise_for_status()
    print(resp.json())

    data = resp.json()
    manufacturer = data.get("manufacturer") or "Не определено"
    model = data.get("model") or "Не определено"
    tire_size = data.get("tire_size_string") or "Не определено"

    await update.message.reply_text(
        f"Производитель: {manufacturer}\nМодель: {model}\nРазмер: {tire_size}",
        reply_markup=InlineKeyboardMarkup(
            [
                [InlineKeyboardButton("OK", callback_data=CB_SIDE_OK)],
                [InlineKeyboardButton("Свой вариант", callback_data=CB_SIDE_CUSTOM)],
            ]
        ),
    )
    return SIDE_RESULT


async def side_result(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle side‐view result buttons."""
    query = update.callback_query
    await query.answer()

    if query.data == CB_SIDE_OK:
        await context.bot.send_message(query.message.chat_id, "Хорошего дня!")
        return await send_main_menu(update, context)

    # CB_SIDE_CUSTOM
    await context.bot.send_message(
        query.message.chat_id,
        "Введите свой вариант производителя, модели и размера шины:",
    )
    return SIDE_CUSTOM


async def side_custom(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle free‐text for side‐view."""
    user_text = update.message.text
    await update.message.reply_text("Спасибо! Благодаря вам модель станет лучше")
    return await send_main_menu(update, context)


async def tread_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Process tread photo"""
    await update.message.reply_text("Обработка фотографии...")

    photo = update.message.photo[-1]
    tg_file = await photo.get_file()

    bio = io.BytesIO()
    await tg_file.download_to_memory(out=bio)
    bio.seek(0)

    b64 = base64.b64encode(bio.read()).decode("utf-8")
    header = {"Authorization": context.user_data["token"]}
    payload = {"image": b64, "token": context.user_data["token"]}

    resp = requests.post(TREAD_ANALYSIS_URL, headers=header, json=payload)
    resp.raise_for_status()

    data = resp.json()

    # logger.log(msg=f"keys: {data.keys()}")

    tread_depth = data["thread_depth"]
    spikes = data["spikes"]
    annotated_image_base64 = data["image"]

    num_bad = sum(spike["class"] == 1 for spike in spikes)
    num_good = len(spikes) - num_bad

    bio = io.BytesIO(base64.b64decode(annotated_image_base64))
    bio.seek(0)
    image = Image.open(bio)
    image.save(bio, format="PNG")
    bio.seek(0)
    await update.message.reply_photo(bio)

    await update.message.reply_text(
        f"Глубина протектора: {tread_depth:.2f}\n"
        + f"Количество плохих шипов: {num_bad}\n"
        + f"Количество хороших шипов: {num_good}",
        reply_markup=InlineKeyboardMarkup(
            [
                [InlineKeyboardButton("OK", callback_data=CB_TREAD_OK)],
                # [InlineKeyboardButton("Свой вариант", callback_data=CB_TREAD_CUSTOM)],
            ]
        ),
    )
    return TREAD_RESULT


async def tread_result(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle tread result buttons."""
    query = update.callback_query
    await query.answer()

    if query.data == CB_TREAD_OK:
        await context.bot.send_message(query.message.chat_id, "Хорошего дня!")
        return await send_main_menu(update, context)

    # CB_TREAD_CUSTOM
    await context.bot.send_message(
        query.message.chat_id,
        "Введите свой вариант глубины протектора и количества шин:",
    )
    return TREAD_CUSTOM


async def tread_custom(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle free‐text for tread."""
    user_text = update.message.text
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
            PASSWORD: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_password)],
            MENU: [CallbackQueryHandler(menu_choice)],
            SIDE_PHOTO: [MessageHandler(filters.PHOTO, side_photo)],
            SIDE_RESULT: [CallbackQueryHandler(side_result)],
            SIDE_CUSTOM: [MessageHandler(filters.TEXT & ~filters.COMMAND, side_custom)],
            TREAD_PHOTO: [MessageHandler(filters.PHOTO, tread_photo)],
            TREAD_RESULT: [CallbackQueryHandler(tread_result)],
            TREAD_CUSTOM: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, tread_custom)
            ],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
        per_user=True,
    )

    app.add_handler(conv_handler)
    app.run_polling()


if __name__ == "__main__":
    main()
