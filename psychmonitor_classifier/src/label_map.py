"""
Маппинг меток SAD v1 (9 классов) → 6 классов PsychMonitor.

Правила:
  Work, School                                         → 1 (work)      Работа / дедлайны
  Family Issues                                        → 2 (family)    Семья / отношения
  Health, Fatigue, or Physical Pain                    → 3 (health)    Здоровье
  Social Relationships                                 → 4 (friends)   Друзья / общение
  Financial Problem                                    → 5 (finance)   Финансы
  Emotional Turmoil, Everyday Decision Making, Other   → 6 (unknown)   Не распознано

Также отбрасывает строки с is_stressor == 0 (шум: «?», «a big project», без контекста).
Работает на исходном английском CSV — до перевода.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import pandas as pd

from config import load_config, PROJECT_ROOT


SAD_TO_PSYCH = {
    "Work": 1,
    "School": 1,
    "Family Issues": 2,
    "Health, Fatigue, or Physical Pain": 3,
    "Social Relationships": 4,
    "Financial Problem": 5,
    "Emotional Turmoil": 6,
    "Everyday Decision Making": 6,
    "Other": 6,
}

CLASS_NAMES = {
    1: "work",
    2: "family",
    3: "health",
    4: "friends",
    5: "finance",
    6: "unknown",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None, help="исходный SAD_v1.csv")
    parser.add_argument("--output", default=None, help="CSV только с нужными колонками и метками")
    args = parser.parse_args()

    cfg = load_config()
    inp = Path(args.input) if args.input else PROJECT_ROOT / cfg["data"]["raw_path"]
    out = Path(args.output) if args.output else PROJECT_ROOT / cfg["data"]["mapped_path"]
    out.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(inp)
    before = len(df)

    df = df[df["is_stressor"] == 1].copy()
    after_filter = len(df)

    df["label"] = df["top_label"].map(SAD_TO_PSYCH)
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)
    df["class_name"] = df["label"].map(CLASS_NAMES)

    out_df = df[["sID", "sentence", "top_label", "label", "class_name"]].reset_index(drop=True)
    out_df.to_csv(out, index=False, encoding="utf-8")

    print(f"Исходно:         {before}")
    print(f"После is_stressor==1: {after_filter}")
    print(f"После маппинга:  {len(out_df)}")
    print()
    print("Распределение по классам PsychMonitor:")
    dist = out_df["label"].value_counts().sort_index()
    for k, n in dist.items():
        print(f"  {k} ({CLASS_NAMES[k]:8s}): {n:5d}  ({n/len(out_df)*100:.1f}%)")
    print(f"\nСохранено: {out}")


if __name__ == "__main__":
    main()
