"""
Подготовка train/val/test из переведённого датасета SAD.

- Загружает data/processed/sad_ru.csv
- Фильтрует строки с пустыми или слишком короткими переводами
- Стратифицированный split 80/10/10 (стратификация по label)
- Считает class_weights (inverse frequency) для взвешенной кросс-энтропии
- Сохраняет train.csv, val.csv, test.csv и class_weights.json
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import load_config, PROJECT_ROOT


MIN_LEN = 3  # минимальная длина переведённого текста (символы)


def main() -> None:
    cfg = load_config()
    src = PROJECT_ROOT / cfg["data"]["translated_path"]
    train_p = PROJECT_ROOT / cfg["data"]["train_path"]
    val_p = PROJECT_ROOT / cfg["data"]["val_path"]
    test_p = PROJECT_ROOT / cfg["data"]["test_path"]
    weights_p = PROJECT_ROOT / "data/processed/class_weights.json"

    df = pd.read_csv(src)
    print(f"Загружено: {len(df)}")

    df["sentence_ru"] = df["sentence_ru"].fillna("").astype(str).str.strip()
    df = df[df["sentence_ru"].str.len() >= MIN_LEN].copy()
    print(f"После фильтра длины (>= {MIN_LEN} симв.): {len(df)}")

    # PyTorch требует метки 0..N-1, а у нас 1..6 → сдвигаем
    df["label_id"] = df["label"].astype(int) - 1

    seed = cfg["training"]["seed"]
    test_frac = cfg["data"]["split"]["test"]
    val_frac = cfg["data"]["split"]["val"]

    # Сначала отделяем test, затем val из оставшегося train+val
    trainval, test = train_test_split(
        df, test_size=test_frac, stratify=df["label_id"], random_state=seed,
    )
    rel_val = val_frac / (1.0 - test_frac)
    train, val = train_test_split(
        trainval, test_size=rel_val, stratify=trainval["label_id"], random_state=seed,
    )

    cols = ["sID", "sentence", "sentence_ru", "top_label", "label", "label_id", "class_name"]
    train[cols].to_csv(train_p, index=False, encoding="utf-8")
    val[cols].to_csv(val_p, index=False, encoding="utf-8")
    test[cols].to_csv(test_p, index=False, encoding="utf-8")

    print(f"\nSplit (seed={seed}):")
    for name, part in [("train", train), ("val", val), ("test", test)]:
        dist = part["label"].value_counts().sort_index().to_dict()
        print(f"  {name:5s} = {len(part):5d}   {dist}")

    # Class weights: inverse frequency, нормированные к среднему 1
    counts = train["label_id"].value_counts().sort_index().values.astype(float)
    inv = 1.0 / counts
    weights = inv / inv.mean()
    weights_dict = {int(i): float(w) for i, w in enumerate(weights)}

    with open(weights_p, "w", encoding="utf-8") as f:
        json.dump(weights_dict, f, indent=2, ensure_ascii=False)

    print(f"\nClass weights (для CrossEntropyLoss):")
    for i, w in weights_dict.items():
        print(f"  class {i+1} ({cfg['classes'][i+1]:8s}): w = {w:.3f}")

    print(f"\nСохранено:")
    print(f"  {train_p}")
    print(f"  {val_p}")
    print(f"  {test_p}")
    print(f"  {weights_p}")


if __name__ == "__main__":
    main()
