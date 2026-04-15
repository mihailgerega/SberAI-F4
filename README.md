# SberAI-F4

## Краткое описание

SberAI-F4 - это веб-прототип для обработки фото и видео с покадровой отправкой в backend.

Сейчас приложение делает следующее:
- принимает изображение или видео через web-интерфейс;
- для видео извлекает кадры и отправляет их по одному в backend;
- backend возвращает для каждого кадра:
  - сам кадр,
  - маски-заглушки в формате YOLO Segmentation.

Текущие заглушки масок:
- `track_markup_stub` (разметка трассы),
- `wheel_segmentation_stub` (колесо).

## Быстрый запуск

Требования:
- Python 3.11+
- pip

Команды для Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Если в системе есть `python`, можно использовать его вместо `python3`.

После запуска откройте в браузере:

```text
http://127.0.0.1:8000
```

## Как проверить работу

1. Нажмите "Выбрать медиафайл" и выберите фото или видео.
2. Нажмите "Запустить обработку".
3. В интерфейсе появится кадр с масками, а справа - YOLO Segmentation строки.

## API (кратко)

Endpoint:

```text
POST /api/infer/frame
```

Form-data параметры:
- `frame` (файл кадра),
- `frame_index` (номер кадра, int).