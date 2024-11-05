import os
import pandas as pd
from aiogram import Bot, Dispatcher, types, Router
from aiogram.filters import Command
import asyncio
from aiogram import F
from ourmodel import FinalModel


# Введите токен вашего бота
TOKEN = '7982846313:AAG_ZgtSWpyDcY062JWNUfnNokZ1jGeMLZE'


# Инициализация бота и диспетчера
bot = Bot(token=TOKEN)
dp = Dispatcher()
model = FinalModel(6)
router = Router()
print("Preparation complete")
# Хэндлер для команды /start
@dp.message(Command("start"))
async def start(message: types.Message):
    await message.answer("Привет! Я бот для определения вероятности рака кожи. Пришлите мне jpg изображение.")

@router.message(F.photo)
async def handle_document(message: types.Message):
    
    photos = [message.photo[-1]]  # Берем фотографию наибольшего размера
    file_paths = []
    for photo in photos:
        file_info = await bot.get_file(photo.file_id)
        file_path = file_info.file_path
        downloaded_file = await bot.download_file(file_path)
        file_name = file_info.file_path.split("/")[-1]
        file_path = "./downloads/" + file_name
        file_paths.append(file_path)
        with open(file_path, "wb") as f:
            f.write(downloaded_file.read())
    # file_info = await bot.get_file(photo.file_id)
    # file_path = file_info.file_path
    # downloaded_file = await bot.download_file(file_path)
    # file_name = file_info.file_path.split("/")[-1]
    # file_path = "./downloads/" + file_name
    # with open(file_path, "wb") as f:
    #     f.write(downloaded_file.read())
    try:
        # Обработка файла CSV с помощью pandas

        prediction = model.calibrated_predict(image_paths=file_paths)
        # Сохраняем новый CSV файл
        reply = ""
        for i in range(len(prediction["binary_probs"])):
            os.remove(file_paths[i])
            reply += f"Фото {i+1}: Вероятность злокачественности на фото: {prediction['binary_probs'][i] * 100:.1f}%. Данное фото имеет наибольшее сходство с {prediction['multiclass_preds'][i]} \n"
        await message.reply(reply)


    except Exception as e:
        await message.reply(f"Ошибка при обработке файла: {e}")

# Основная функция для запуска бота
async def main():
    dp.include_router(router)
    # Запускаем бота
    await dp.start_polling(bot)

if __name__ == '__main__':
    # Создаем папку для временных файлов, если ее нет
    if not os.path.exists('downloads'):
        os.makedirs('downloads')

    # Запуск бота с помощью asyncio
    asyncio.run(main())
