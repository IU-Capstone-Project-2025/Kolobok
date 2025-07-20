import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch, MagicMock

import tg.main as bot

@pytest.fixture(autouse=True)
def set_env(monkeypatch):
    monkeypatch.setenv("BOT_TOKEN", "test-bot-token")
    monkeypatch.setenv("API_TOKEN", "test-api-token")
    monkeypatch.setenv("ALLOWED_USERS", "12345,67890")

@pytest_asyncio.fixture
def fake_context():
    ctx = MagicMock()
    ctx.bot = AsyncMock()
    return ctx

@pytest_asyncio.fixture
def fake_update():
    upd = MagicMock()
    upd.effective_user.id = 12345
    upd.effective_user.username = "testuser"
    upd.message = AsyncMock()
    upd.message.chat_id = 1
    upd.message.reply_text = AsyncMock()
    upd.callback_query = None
    return upd

@pytest.mark.asyncio
async def test_start_allowed(fake_update, fake_context):
    res = await bot.start(fake_update, fake_context)
    assert res == bot.MENU
    fake_context.bot.send_message.assert_called()

@pytest.mark.asyncio
async def test_start_restricted(fake_update, fake_context):
    fake_update.effective_user.id = 99999  # not in ALLOWED_USERS
    await bot.start(fake_update, fake_context)
    fake_update.message.reply_text.assert_called_with(
        "У вас нет доступа к функциям этого бота. Обратитесь к администратору"
    )

@pytest.mark.asyncio
async def test_menu_choice_side(fake_update, fake_context):
    fake_update.callback_query = MagicMock()
    fake_update.callback_query.data = bot.CB_SIDE
    fake_update.callback_query.answer = AsyncMock()
    fake_update.callback_query.edit_message_text = AsyncMock()
    fake_update.effective_user.username = "testuser"
    res = await bot.menu_choice(fake_update, fake_context)
    assert res == bot.SIDE_PHOTO
    fake_update.callback_query.edit_message_text.assert_called_with(
        text="Загрузите фотографию боковой стороны шины"
    )

@pytest.mark.asyncio
async def test_menu_choice_tread(fake_update, fake_context):
    fake_update.callback_query = MagicMock()
    fake_update.callback_query.data = bot.CB_TREAD
    fake_update.callback_query.answer = AsyncMock()
    fake_update.callback_query.edit_message_text = AsyncMock()
    fake_update.effective_user.username = "testuser"
    res = await bot.menu_choice(fake_update, fake_context)
    assert res == bot.TREAD_PHOTO
    fake_update.callback_query.edit_message_text.assert_called_with(
        text="Загрузите фотографию протектора шины"
    )

@pytest.mark.asyncio
async def test_cancel(fake_update, fake_context):
    fake_update.effective_user.username = "testuser"
    res = await bot.cancel(fake_update, fake_context)
    assert res == bot.ConversationHandler.END
    fake_update.message.reply_text.assert_called_with("До свидания!")

@pytest.mark.asyncio
@patch("tg.main.requests.post")
async def test_side_photo_success(mock_post, fake_update, fake_context):
    # Мокаем telegram file download
    photo_mock = MagicMock()
    photo_mock.get_file = AsyncMock(return_value=AsyncMock(download_to_memory=AsyncMock()))
    fake_update.message.photo = [photo_mock, photo_mock]
    # Мокаем requests.post
    mock_post.return_value.json.return_value = {
        "manufacturer": "TestBrand",
        "model": "TestModel",
        "tire_size_string": "205/55R16"
    }
    mock_post.return_value.raise_for_status = lambda: None
    fake_update.message.reply_text = AsyncMock()
    res = await bot.side_photo(fake_update, fake_context)
    assert res == bot.SIDE_RESULT
    fake_update.message.reply_text.assert_any_call(
        "Производитель: TestBrand\nМодель: TestModel\nРазмер: 205/55R16",
        reply_markup=pytest.helpers.anything()
    )

@pytest.mark.asyncio
async def test_side_result_ok(fake_update, fake_context):
    fake_update.callback_query = MagicMock()
    fake_update.callback_query.data = bot.CB_SIDE_OK
    fake_update.callback_query.answer = AsyncMock()
    fake_update.callback_query.message.chat_id = 1
    fake_update.effective_user.username = "testuser"
    fake_context.bot.send_message = AsyncMock()
    res = await bot.side_result(fake_update, fake_context)
    assert res == bot.MENU
    fake_context.bot.send_message.assert_called_with(1, "Хорошего дня!")

@pytest.mark.asyncio
async def test_side_result_custom(fake_update, fake_context):
    fake_update.callback_query = MagicMock()
    fake_update.callback_query.data = bot.CB_SIDE_CUSTOM
    fake_update.callback_query.answer = AsyncMock()
    fake_update.callback_query.message.chat_id = 1
    fake_update.effective_user.username = "testuser"
    fake_context.bot.send_message = AsyncMock()
    res = await bot.side_result(fake_update, fake_context)
    assert res == bot.SIDE_CUSTOM
    fake_context.bot.send_message.assert_called_with(
        1, "Введите свой вариант производителя, модели и размера шины:")

@pytest.mark.asyncio
async def test_side_custom(fake_update, fake_context):
    fake_update.message.text = "Мой вариант"
    fake_update.effective_user.username = "testuser"
    fake_update.message.reply_text = AsyncMock()
    res = await bot.side_custom(fake_update, fake_context)
    assert res == bot.MENU
    fake_update.message.reply_text.assert_called_with("Спасибо! Благодаря вам модель станет лучше")

@pytest.mark.asyncio
async def test_tread_result_ok(fake_update, fake_context):
    fake_update.callback_query = MagicMock()
    fake_update.callback_query.data = bot.CB_TREAD_OK
    fake_update.callback_query.answer = AsyncMock()
    fake_update.callback_query.message.chat_id = 1
    fake_update.effective_user.username = "testuser"
    fake_context.bot.send_message = AsyncMock()
    res = await bot.tread_result(fake_update, fake_context)
    assert res == bot.MENU
    fake_context.bot.send_message.assert_called_with(1, "Хорошего дня!")

@pytest.mark.asyncio
async def test_tread_result_custom(fake_update, fake_context):
    fake_update.callback_query = MagicMock()
    fake_update.callback_query.data = bot.CB_TREAD_CUSTOM
    fake_update.callback_query.answer = AsyncMock()
    fake_update.callback_query.message.chat_id = 1
    fake_update.effective_user.username = "testuser"
    fake_context.bot.send_message = AsyncMock()
    res = await bot.tread_result(fake_update, fake_context)
    assert res == bot.TREAD_CUSTOM
    fake_context.bot.send_message.assert_called_with(
        1, "Введите свой вариант глубины протектора и количества шин:")

@pytest.mark.asyncio
async def test_tread_custom(fake_update, fake_context):
    fake_update.message.text = "Мой вариант"
    fake_update.effective_user.username = "testuser"
    fake_update.message.reply_text = AsyncMock()
    res = await bot.tread_custom(fake_update, fake_context)
    assert res == bot.MENU
    fake_update.message.reply_text.assert_called_with("Спасибо! Благодаря вам модель станет лучше")

@pytest.mark.asyncio
@patch("tg.main.requests.post")
@patch("tg.main.Image.open")
async def test_tread_photo_success(mock_image_open, mock_post, fake_update, fake_context):
    # Мокаем telegram file download
    photo_mock = MagicMock()
    photo_mock.get_file = AsyncMock(return_value=AsyncMock(download_to_memory=AsyncMock()))
    fake_update.message.photo = [photo_mock, photo_mock]
    # Мокаем requests.post
    mock_post.return_value.json.return_value = {
        "success": 1,
        "thread_depth": 7.5,
        "spikes": [{"class": 1}, {"class": 0}, {"class": 1}],
        "image": "dGVzdA=="  # base64 'test'
    }
    mock_post.return_value.raise_for_status = lambda: None
    fake_update.message.reply_text = AsyncMock()
    fake_update.message.reply_photo = AsyncMock()
    mock_image_open.return_value.save = MagicMock()
    res = await bot.tread_photo(fake_update, fake_context)
    assert res == bot.TREAD_RESULT
    fake_update.message.reply_text.assert_any_call(
        "Глубина протектора: 7.50\nКоличество плохих шипов: 2\nКоличество хороших шипов: 1",
        reply_markup=pytest.helpers.anything()
    )
    fake_update.message.reply_photo.assert_called()
