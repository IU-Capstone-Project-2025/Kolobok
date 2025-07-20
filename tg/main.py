import logging
import base64
import requests
import io
import os
from PIL import Image
import time
from functools import wraps

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
API_TOKEN = os.environ["API_TOKEN"]

TREAD_ANALYSIS_URL = f"http://{APP_URL}/api/v1/analyze_thread"
TIRE_INFO_EXTRACTION_URL = f"http://{APP_URL}/api/v1/extract_information"

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger("tg-bot")


class DropHTTPReqFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "HTTP Request:" not in record.getMessage()


handler = logging.getLogger().handlers[0]
handler.addFilter(DropHTTPReqFilter())

ALLOWED_USERS = os.environ["ALLOWED_USERS"].split(",")


def restricted(func):
    @wraps(func)
    async def wrapper(
        update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs
    ):
        user_id = update.effective_user.id
        if str(user_id) not in ALLOWED_USERS:
            # Optionally log attempt
            logger.info(f"Unauthorized access denied for {user_id}")
            await update.message.reply_text(
                "У вас нет доступа к функциям этого бота. Обратитесь к администратору"
            )
            return  # do not call the handler
        return await func(update, context, *args, **kwargs)

    return wrapper


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
    if update.callback_query:
        await update.callback_query.answer()
        chat_id = update.callback_query.message.chat_id
    else:
        chat_id = update.message.chat_id

    await context.bot.send_message(
        chat_id=chat_id, text="Чем я могу помочь?", reply_markup=build_main_menu()
    )
    return MENU


@restricted
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    username = update.effective_user.username
    logger.info(f"User {username} started conversation")
    return await send_main_menu(update, context)


async def menu_choice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle main‐menu button presses."""
    username = update.effective_user.username
    query = update.callback_query
    await query.answer()

    if query.data == CB_SIDE:
        logging.info(f"User {username} picked OCR")
        await query.edit_message_text(text="Загрузите фотографию боковой стороны шины")
        return SIDE_PHOTO

    if query.data == CB_TREAD:
        logging.info(f"User {username} picked tread")
        await query.edit_message_text(
            text="Загрузите фотографию протектора шины",
        )
        return TREAD_PHOTO

    # CB_MENU — redisplay the menu
    return await send_main_menu(update, context)


async def side_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    start = time.perf_counter()
    username = update.effective_user.username
    logger.info(f"User {username} uploaded photo for OCR")
    await update.message.reply_text("Обработка фотографии...")

    photo = update.message.photo[-1]
    tg_file = await photo.get_file()

    bio = io.BytesIO()
    await tg_file.download_to_memory(out=bio)
    bio.seek(0)
    logger.info(f"Successfully downloaded photo of {username} to memory")

    b64 = base64.b64encode(bio.read()).decode("utf-8")
    header = {"Authorization": f"Bearer {API_TOKEN}"}
    payload = {"image": b64}

    logger.info(f"Posted response for OCR for {username}")
    resp = requests.post(TIRE_INFO_EXTRACTION_URL, headers=header, json=payload)
    resp.raise_for_status()

    data = resp.json()

    # Extract the index results from the new response format
    index_results = data.get("index_results", [])
    raw_strings = data.get("strings", [])
    tire_size = data.get("tire_size", "Не определено")

    logger.info(f"User {username} got the OCR result with {len(index_results)} matches")

    if not index_results:
        await update.message.reply_text(
            "❌ **Результат не найден**\n\n"
            "К сожалению, не удалось определить марку и модель шины на фотографии.\n\n"
            "💡 **Возможные причины:**\n"
            "• Низкое качество фотографии\n"
            "• Плохое освещение\n"
            "• Текст на шине нечёткий или повреждён\n"
            "• Неподдерживаемая марка шины\n\n"
            "📝 Попробуйте загрузить фото лучшего качества или введите данные вручную.",
            reply_markup=InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton(
                            "✏️ Ввести вручную", callback_data=CB_SIDE_CUSTOM
                        )
                    ],
                    [InlineKeyboardButton("🏠 В меню", callback_data=CB_MENU)],
                ]
            ),
            parse_mode="Markdown",
        )
        return SIDE_RESULT

    message_parts = ["🔍 **Найденные совпадения:**\n"]

    # Find the maximum confidence score
    max_confidence = (
        max(match.get("combined_score", 0) for match in index_results)
        if index_results
        else 0
    )

    for i, match in enumerate(index_results, 1):
        brand = match.get("brand_name", "Не определено")
        model = match.get("model_name", "Не определено")
        # Use the tire_size from the top-level response instead of individual matches
        match_tire_size = tire_size
        score = match.get("combined_score", 0)

        confidence_emoji = "🟢" if score > 0.8 else "🟡" if score > 0.6 else "🔴"
        confidence_text = (
            "Высокая" if score > 0.8 else "Средняя" if score > 0.6 else "Низкая"
        )

        message_parts.append(
            f"{confidence_emoji} **Результат {i}:**\n"
            f"Линейка (Бренд): {brand}\n"
            f"Модель: {model}\n"
            f"Размер: {match_tire_size}\n"
            f"Точность: {confidence_text} ({score:.1%})\n"
        )

    if max_confidence < 0.8 and raw_strings:
        formatted_strings = ", ".join(f'"{string}"' for string in raw_strings)
        message_parts.append(f"\n🔍 **Сырые OCR данные:**\n`{formatted_strings}`\n")

    message_parts.append("📝 Выберите подходящий результат или введите свой вариант:")

    logger.info(
        f"OCR result for {username}: {len(index_results)} matches found, max confidence: {max_confidence:.3f}"
    )
    await update.message.reply_text(
        "\n".join(message_parts),
        reply_markup=InlineKeyboardMarkup(
            [
                [InlineKeyboardButton("✅ Подходит", callback_data=CB_SIDE_OK)],
                [InlineKeyboardButton("✏️ Свой вариант", callback_data=CB_SIDE_CUSTOM)],
            ]
        ),
        parse_mode="Markdown",
    )
    end = time.perf_counter()
    logger.info(f"OCR for {username} complete in {end - start}")
    return SIDE_RESULT


async def side_result(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle side‐view result buttons."""
    username = update.effective_user.username
    query = update.callback_query
    await query.answer()

    if query.data == CB_SIDE_OK:
        logger.info(f"User {username} agreed with OCR result")
        await context.bot.send_message(
            query.message.chat_id,
            "✅ **Результат принят!**\n\nСпасибо за подтверждение. Данные сохранены.",
            parse_mode="Markdown",
        )
        return await send_main_menu(update, context)

    # CB_SIDE_CUSTOM
    logger.info(f"User {username} edit the OCR result")
    await context.bot.send_message(
        query.message.chat_id,
        "✏️ **Введите данные вручную**\n\n"
        "📋 **Формат:**\nПроизводитель Модель Размер\n\n"
        "💡 **Пример:**\nNokian Hakka Blue 225/60R17",
        parse_mode="Markdown",
    )
    return SIDE_CUSTOM


async def side_custom(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle free‐text for side‐view."""
    username = update.effective_user.username
    user_text = update.message.text
    logging.info(f"User {username} edits OCR result: {user_text}")
    await update.message.reply_text(
        "🙏 **Спасибо за коррекцию!**\n\nБлагодаря вам модель станет точнее.",
        parse_mode="Markdown",
    )
    return await send_main_menu(update, context)


async def tread_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Process tread photo"""
    start = time.perf_counter()
    username = update.effective_user.username
    logger.info(f"User {username} uploaded photo for tread")
    await update.message.reply_text("Обработка фотографии...")

    photo = update.message.photo[-1]
    tg_file = await photo.get_file()

    bio = io.BytesIO()
    await tg_file.download_to_memory(out=bio)
    bio.seek(0)
    logger.info(f"Successfully downloaded photo of {username} to memory")

    b64 = base64.b64encode(bio.read()).decode("utf-8")
    header = {"Authorization": f"Bearer {API_TOKEN}"}
    payload = {"image": b64, "token": API_TOKEN}

    logger.info(f"Posted response for tread for {username}")
    resp = requests.post(TREAD_ANALYSIS_URL, headers=header, json=payload)
    resp.raise_for_status()

    data = resp.json()
    logger.info(f"User {username} got the tread result: {resp.json().keys()}")
    # logger.info(data)

    if resp.json()["success"] == 0:
        error_detail = resp.json().get("detail", "Неизвестная ошибка")
        await update.message.reply_text(
            f"❌ **Ошибка анализа протектора**\n\n"
            f"Не удалось обработать фотографию протектора.\n\n"
            f"🔍 **Детали ошибки:**\n{error_detail}\n\n"
            f"💡 **Попробуйте:**\n"
            f"• Сделать фото лучшего качества\n"
            f"• Убедиться, что протектор хорошо виден\n"
            f"• Улучшить освещение\n"
            f"• Избегать бликов и теней",
            reply_markup=InlineKeyboardMarkup(
                [
                    [InlineKeyboardButton("🏠 В меню", callback_data=CB_MENU)],
                ]
            ),
            parse_mode="Markdown",
        )
        return MENU

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

    logger.info(
        f"Tread detection result for {username}: tread depth: {tread_depth:.2f}, bad: {num_bad}, good: {num_good}"
    )

    total_spikes = len(spikes)
    good_percentage = (num_good / total_spikes * 100) if total_spikes > 0 else 0
    bad_percentage = (num_bad / total_spikes * 100) if total_spikes > 0 else 0

    # Determine emoji based on condition
    if tread_depth >= 4.0:
        depth_emoji = "✅"
    elif tread_depth >= 2.0:
        depth_emoji = "⚠️"
    else:
        depth_emoji = "❌"

    if bad_percentage <= 10:
        spike_emoji = "✅"
    elif bad_percentage <= 30:
        spike_emoji = "⚠️"
    else:
        spike_emoji = "❌"

    formatted_message = (
        f"📊 **Результаты анализа протектора:**\n\n"
        f"{depth_emoji} **Глубина протектора:** {tread_depth:.2f} мм\n\n"
        f"{spike_emoji} **Анализ шипов:**\n"
        f"Всего шипов: {total_spikes}\n"
        f"Хорошие: {num_good} ({good_percentage:.1f}%)\n"
        f"Поврежденные: {num_bad} ({bad_percentage:.1f}%)"
    )

    # Add critical warnings only when necessary
    if tread_depth < 1.6:
        formatted_message += "\n\n❌ **Требуется немедленная замена!**"
    elif tread_depth < 2.0 or bad_percentage > 30:
        formatted_message += "\n\n⚠️ **Рекомендуется замена в ближайшее время**"

    await update.message.reply_text(
        formatted_message,
        reply_markup=InlineKeyboardMarkup(
            [
                [InlineKeyboardButton("✅ Принято", callback_data=CB_TREAD_OK)],
                [InlineKeyboardButton("✏️ Свой вариант", callback_data=CB_TREAD_CUSTOM)],
            ]
        ),
        parse_mode="Markdown",
    )
    end = time.perf_counter()
    logger.info(f"Tread for {username} complete in {end - start}")
    return TREAD_RESULT


async def tread_result(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle tread result buttons."""
    username = update.effective_user.username
    query = update.callback_query
    await query.answer()

    if query.data == CB_TREAD_OK:
        logger.info(f"User {username} agreed with tread result")
        await context.bot.send_message(
            query.message.chat_id,
            "✅ **Анализ завершён!**\n\nСпасибо за использование бота. Хорошего дня!",
            parse_mode="Markdown",
        )
        return await send_main_menu(update, context)

    # CB_TREAD_CUSTOM
    logger.info(f"User {username} edits tread result")
    await context.bot.send_message(
        query.message.chat_id,
        "✏️ **Введите свои данные**\n\n"
        "📋 **Укажите:**\n"
        "• Глубина протектора (мм)\n"
        "• Количество хороших шипов\n"
        "• Количество поврежденных шипов\n\n"
        "💡 **Пример:**\n3.5 мм, хорошие: 45, плохие: 5",
        parse_mode="Markdown",
    )
    return TREAD_CUSTOM


async def tread_custom(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle free‐text for tread."""
    user_text = update.message.text
    username = update.effective_user.username
    logger.info(f"{username} edit for tread: {user_text}")
    await update.message.reply_text(
        "🙏 **Спасибо за коррекцию!**\n\nБлагодаря вам модель станет лучше.",
        parse_mode="Markdown",
    )
    return await send_main_menu(update, context)


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle /cancel to end conversation."""
    username = update.effective_user.username
    await update.message.reply_text("До свидания!")
    logger.info(f"User {username} ended the conversation")
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
