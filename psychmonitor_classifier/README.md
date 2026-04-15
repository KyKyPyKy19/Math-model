# PsychMonitor — Блок C: Классификатор триггеров стресса

Демонстрационный модуль блока `C` математической модели PsychMonitor.
Задача — классификация текстового комментария клиента (на русском языке)
по 6 группам триггеров стрессового эпизода.

> Полный отчёт о проделанной работе и метрики качества — в [REPORT.md](REPORT.md).

## Классы триггеров

| k | Группа               | Примеры                                  |
|---|----------------------|------------------------------------------|
| 1 | Работа / дедлайны    | «Сложный звонок с начальником»           |
| 2 | Семья / отношения    | «Ссора с мужем из-за ребёнка»            |
| 3 | Здоровье             | «Болит голова с утра»                    |
| 4 | Друзья / общение     | «Неприятный разговор с подругой»         |
| 5 | Финансы              | «Пришли счета, задолженность растёт»     |
| 6 | Не распознано        | «Просто стало тревожно, не знаю почему»  |

## Результаты на тестовой выборке

| Метрика | Значение | Цель (PDF) |
|---|---|---|
| Accuracy | 0.821 | — |
| Macro F1 | **0.810** | ≥ 0.70 ✓ |
| Weighted F1 | 0.820 | — |
| AUC-ROC (macro OvR) | **0.964** | ≥ 0.85 ✓ |
| Log Loss | 0.574 | — |
| ECE (10 bins) | 0.058 | — |

Обе жёстких цели математической модели превышены с запасом.

## Структура проекта

```
psychmonitor_classifier/
├── config.yaml                     # гиперпараметры, пути, пороги
├── requirements.txt
├── README.md                       # этот файл
├── REPORT.md                       # подробный отчёт о работе
├── data/
│   ├── raw/SAD_v1.csv              # исходный англ. датасет (Stress Annotated Dataset)
│   └── processed/
│       ├── sad_mapped.csv          # после фильтрации is_stressor==1 и маппинга 9→6 классов
│       ├── sad_ru.csv              # после русификации через Google Translate
│       ├── train.csv, val.csv, test.csv   # стратифицированный split 80/10/10
│       └── class_weights.json      # веса классов для CrossEntropyLoss
├── src/
│   ├── config.py                   # загрузчик конфига
│   ├── label_map.py                # 9 классов SAD → 6 классов PsychMonitor
│   ├── translate.py                # EN → RU (deep_translator + GoogleTranslator)
│   ├── prepare_dataset.py          # split + class_weights
│   ├── train.py                    # fine-tuning rubert-tiny2 (HuggingFace Trainer)
│   ├── evaluate.py                 # метрики, confusion matrix, графики
│   └── predict.py                  # демо-инференс (CLI, интерактивный REPL)
├── models/
│   └── rubert_trigger_classifier/  # сохранённый чекпоинт
└── reports/
    ├── train_val_metrics.json
    ├── test_metrics.json
    ├── confusion_matrix.csv / .png
    └── confidence_histogram.png
```

## Установка

```bash
# Зависимости
pip install -r requirements.txt

# PyTorch с CUDA (для Python 3.13, GPU NVIDIA)
pip install torch --index-url https://download.pytorch.org/whl/cu124

# Проверка GPU
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

## Пайплайн

Команды выполняются из корня `psychmonitor_classifier/`.

```bash
# 1. Фильтр шума + маппинг меток 9 → 6 классов (секунды)
python src/label_map.py
# → data/processed/sad_mapped.csv (6476 строк)

# 2. Русификация EN → RU (~30–50 мин)
python src/translate.py
# → data/processed/sad_ru.csv
# Если оборвётся: python src/translate.py --resume

# 3. Стратифицированный split + class weights
python src/prepare_dataset.py
# → train.csv, val.csv, test.csv, class_weights.json

# 4. Fine-tuning rubert-tiny2 (~12 мин на GTX 1050 Ti)
python src/train.py
# → models/rubert_trigger_classifier/

# 5. Полная оценка на test
python src/evaluate.py
# → reports/test_metrics.json, confusion_matrix.png, confidence_histogram.png

# 6. Демо-инференс
python src/predict.py --text "Опять дедлайн горит, начальник требует отчёт к утру"
python src/predict.py --interactive
```

## Модель

- **Базовая сеть:** `cointegrated/rubert-tiny2` (29M параметров)
- **Max length:** 128 токенов
- **Loss:** weighted CrossEntropy (компенсация дисбаланса)
- **Optimizer:** AdamW, lr=5e-5, weight_decay=0.01
- **Batch size:** 32 (train) / 64 (eval), FP16
- **Early stopping:** по Macro-F1 на validation, patience=2
- **Порог уверенности:** `τ_conf = 0.4` → при меньшей conf fallback в класс 6 («Не распознано»)

## Формат ответа модуля

На каждый комментарий `predict.py` возвращает JSON, соответствующий схеме записи
в БД PsychMonitor (§2.2.5 математической модели):

```json
{
  "input": "Опять дедлайн горит, начальник требует отчёт к утру",
  "predicted_class_id": 1,
  "predicted_class_name_ru": "Работа / дедлайны",
  "confidence": 0.9412,
  "trigger_group": 1,
  "user_description": "Опять дедлайн горит, начальник требует отчёт к утру",
  "probabilities": {
    "Работа / дедлайны":       0.9412,
    "Семья / отношения":       0.0138,
    "Здоровье / самочувствие": 0.0092,
    "Друзья / общение":        0.0211,
    "Финансы":                 0.0074,
    "Не распознано":           0.0073
  },
  "reason": "argmax"
}
```

Оригинальный текст (`user_description`) сохраняется независимо от результата
классификации — требование §2.2.5.

## Соответствие разделам математической модели

- **§2.2.2** — 6 классов триггеров → `config.yaml: classes`
- **§2.2.3** — RuBERT + softmax head, `τ_conf = 0.4` → `predict.py`, `evaluate.py`
- **§2.2.4** — AdamW, lr, batch, epochs, weighted CE, early stopping → `train.py`
- **§2.2.5** — обработка null-комментариев, сохранение оригинала → `predict.py`
- **§3.2** — Accuracy, Macro F1, Weighted F1, Confusion Matrix → `evaluate.py`
- **§3.3** — AUC-ROC, Log Loss, Calibration Error (ECE) → `evaluate.py`
