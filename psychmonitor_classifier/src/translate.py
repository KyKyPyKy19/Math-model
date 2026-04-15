"""
Русификация SAD v1 через GoogleTranslator (deep_translator).

Переводит колонку `sentence` (англ.) → `sentence_ru` (рус.).
Работает на уже размеченном CSV (после label_map.py).

Особенности:
  - Потоковая запись: сохраняет CSV каждые --save-every строк,
    чтобы при обрыве связи не терять прогресс.
  - --resume: продолжает с места, где остановились (по sID).
  - Предложения короткие (<=185 симв.), разбивать на чанки не нужно.
  - Пауза между запросами, чтобы не словить rate-limit.
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from deep_translator import GoogleTranslator

from config import load_config, PROJECT_ROOT


def translate_one(translator: GoogleTranslator, text: str, retries: int = 3) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    for attempt in range(retries):
        try:
            return translator.translate(text)
        except Exception as e:
            if attempt == retries - 1:
                print(f"[err] «{text[:60]}...»: {e}")
                return ""
            time.sleep(2 ** attempt)
    return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None, help="CSV с колонкой sentence (после label_map.py)")
    parser.add_argument("--output", default=None)
    parser.add_argument("--resume", action="store_true", help="продолжить с сохранённого файла")
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--limit", type=int, default=None, help="для теста — первые N строк")
    parser.add_argument("--sleep-ms", type=int, default=50, help="пауза между запросами, мс")
    args = parser.parse_args()

    cfg = load_config()
    inp = Path(args.input) if args.input else PROJECT_ROOT / cfg["data"]["mapped_path"]
    out = Path(args.output) if args.output else PROJECT_ROOT / cfg["data"]["translated_path"]
    out.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(inp)
    if args.limit:
        df = df.head(args.limit).copy()

    if "sentence_ru" not in df.columns:
        df["sentence_ru"] = ""

    if args.resume and out.exists():
        prev = pd.read_csv(out)
        if "sentence_ru" in prev.columns:
            ru_map = dict(zip(prev["sID"], prev["sentence_ru"].fillna("")))
            df["sentence_ru"] = df["sID"].map(ru_map).fillna("")
            done = int((df["sentence_ru"].astype(str).str.len() > 0).sum())
            print(f"[resume] уже переведено: {done}/{len(df)}")

    translator = GoogleTranslator(source="en", target="ru")
    sleep_s = max(args.sleep_ms, 0) / 1000.0

    todo_mask = df["sentence_ru"].astype(str).str.len() == 0
    todo_idx = df.index[todo_mask].tolist()

    pbar = tqdm(todo_idx, desc="EN→RU")
    for i, idx in enumerate(pbar):
        df.at[idx, "sentence_ru"] = translate_one(translator, df.at[idx, "sentence"])
        if sleep_s:
            time.sleep(sleep_s)
        if (i + 1) % args.save_every == 0:
            df.to_csv(out, index=False, encoding="utf-8")
            pbar.set_postfix(saved=i + 1)

    df.to_csv(out, index=False, encoding="utf-8")

    empty = int((df["sentence_ru"].astype(str).str.len() == 0).sum())
    print(f"\nГотово. Всего: {len(df)}, пустых переводов: {empty}")
    print(f"Сохранено: {out}")


if __name__ == "__main__":
    main()
