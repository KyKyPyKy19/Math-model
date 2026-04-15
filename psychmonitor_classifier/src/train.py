"""
Fine-tuning RuBERT-tiny2 для классификации триггеров стресса (блок C PsychMonitor).

Реализует §2.2.3–2.2.4 математической модели:
  - Входные токены → BERT → [CLS] → Linear(768→6) → softmax
  - Функция потерь: categorical cross-entropy (ф. 2.14), опционально взвешенная
  - AdamW, lr=5e-5 (tiny), warmup, ранняя остановка по Macro-F1 на val
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    set_seed,
)

from config import load_config, PROJECT_ROOT


def load_split(path: Path) -> Dataset:
    df = pd.read_csv(path)
    return Dataset.from_pandas(
        df[["sentence_ru", "label_id"]].rename(columns={"sentence_ru": "text", "label_id": "labels"}),
        preserve_index=False,
    )


def compute_metrics_fn(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0,
    )
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1,
        "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
        "precision_macro": precision,
        "recall_macro": recall,
    }


class WeightedTrainer(Trainer):
    """Trainer с взвешенной CrossEntropy — компенсирует дисбаланс классов."""

    def __init__(self, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if self.class_weights is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        else:
            loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def main() -> None:
    cfg = load_config()
    set_seed(cfg["training"]["seed"])

    model_name = cfg["model"]["name"]
    max_length = cfg["model"]["max_length"]
    num_labels = cfg["model"]["num_labels"]

    output_dir = PROJECT_ROOT / cfg["paths"]["model_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = PROJECT_ROOT / cfg["paths"]["reports_dir"]
    reports_dir.mkdir(parents=True, exist_ok=True)

    # === Данные ===
    train_ds = load_split(PROJECT_ROOT / cfg["data"]["train_path"])
    val_ds = load_split(PROJECT_ROOT / cfg["data"]["val_path"])

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text"])
    val_ds = val_ds.map(tokenize, batched=True, remove_columns=["text"])

    # === Class weights ===
    class_weights = None
    if cfg["training"]["use_class_weights"]:
        weights_path = PROJECT_ROOT / "data/processed/class_weights.json"
        with open(weights_path, "r", encoding="utf-8") as f:
            w = json.load(f)
        class_weights = torch.tensor(
            [w[str(i)] for i in range(num_labels)], dtype=torch.float32,
        )
        print(f"Class weights: {class_weights.tolist()}")

    # === Модель ===
    id2label = {i: cfg["classes"][i + 1] for i in range(num_labels)}
    label2id = {v: k for k, v in id2label.items()}
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    # === Аргументы обучения ===
    use_cuda = torch.cuda.is_available()
    print(f"CUDA available: {use_cuda}")
    if use_cuda:
        print(f"Device: {torch.cuda.get_device_name(0)}")

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=cfg["training"]["num_epochs"],
        per_device_train_batch_size=cfg["training"]["batch_size"],
        per_device_eval_batch_size=cfg["training"]["eval_batch_size"],
        learning_rate=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
        warmup_ratio=cfg["training"]["warmup_ratio"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=50,
        report_to=[],
        fp16=cfg["training"]["fp16"] and use_cuda,
        seed=cfg["training"]["seed"],
        dataloader_num_workers=0,   # Windows-friendly
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
        class_weights=class_weights,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=cfg["training"]["early_stopping_patience"],
        )],
    )

    # === Обучение ===
    train_result = trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # === Финальная оценка на val ===
    val_metrics = trainer.evaluate()
    val_metrics["train_loss"] = float(train_result.training_loss)

    with open(reports_dir / "train_val_metrics.json", "w", encoding="utf-8") as f:
        json.dump(val_metrics, f, indent=2, ensure_ascii=False)

    print("\n=== Финальные метрики на val ===")
    for k, v in val_metrics.items():
        if isinstance(v, float):
            print(f"  {k:25s}: {v:.4f}")

    print(f"\nМодель сохранена: {output_dir}")


if __name__ == "__main__":
    main()
