"""
Демо-инференс: текстовый комментарий → класс триггера + уверенность.

Реализует инференс блока C (§2.2.3 мат. модели):
  ŷ = argmax_k P(y=k|c)     (ф. 2.12)
  conf(c) = max_k P(y=k|c)  (ф. 2.13)

При conf(c) < τ_conf = 0.4 → результат переопределяется в класс 6 (unknown).
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import load_config, PROJECT_ROOT


CLASS_LABELS_RU = {
    1: "Работа / дедлайны",
    2: "Семья / отношения",
    3: "Здоровье / самочувствие",
    4: "Друзья / общение",
    5: "Финансы",
    6: "Не распознано",
}


class TriggerClassifier:
    def __init__(self, model_dir: Path, max_length: int, tau_conf: float):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(self.device)
        self.model.eval()
        self.max_length = max_length
        self.tau_conf = tau_conf
        self.num_labels = self.model.config.num_labels

    @torch.no_grad()
    def predict(self, text: str) -> dict:
        text = (text or "").strip()
        if not text:
            return {
                "input": text,
                "predicted_class_id": None,
                "predicted_class_name_ru": None,
                "confidence": None,
                "trigger_group": None,           # null-эпизод (§2.2.5)
                "user_description": None,
                "probabilities": {},
                "reason": "empty_input",
            }

        enc = self.tokenizer(
            text, truncation=True, max_length=self.max_length,
            padding=True, return_tensors="pt",
        ).to(self.device)
        logits = self.model(**enc).logits[0].cpu().numpy()
        probs = np.exp(logits - logits.max())
        probs = probs / probs.sum()

        pred_id = int(probs.argmax())
        conf = float(probs[pred_id])
        final_class = pred_id + 1  # 0..5 → 1..6

        reason = "argmax"
        if conf < self.tau_conf:
            final_class = 6
            reason = f"conf<{self.tau_conf} → fallback to 6 (unknown)"

        return {
            "input": text,
            "predicted_class_id": final_class,
            "predicted_class_name_ru": CLASS_LABELS_RU[final_class],
            "confidence": round(conf, 4),
            "trigger_group": final_class,
            "user_description": text,            # оригинал сохраняется (§2.2.5)
            "probabilities": {
                CLASS_LABELS_RU[i + 1]: round(float(probs[i]), 4)
                for i in range(self.num_labels)
            },
            "reason": reason,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, default=None, help="Один комментарий для классификации")
    parser.add_argument("--interactive", action="store_true", help="Интерактивный REPL")
    args = parser.parse_args()

    cfg = load_config()
    model_dir = PROJECT_ROOT / cfg["paths"]["model_dir"]
    clf = TriggerClassifier(
        model_dir=model_dir,
        max_length=cfg["model"]["max_length"],
        tau_conf=cfg["inference"]["confidence_threshold"],
    )

    if args.interactive or args.text is None:
        print("PsychMonitor — блок C (классификатор триггеров)")
        print(f"Модель: {cfg['model']['name']}  |  τ_conf = {clf.tau_conf}  |  device = {clf.device}")
        print("Введите комментарий ('exit' для выхода):\n")
        while True:
            try:
                text = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if text.lower() in {"exit", "quit", ""}:
                break
            result = clf.predict(text)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            print()
    else:
        result = clf.predict(args.text)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
