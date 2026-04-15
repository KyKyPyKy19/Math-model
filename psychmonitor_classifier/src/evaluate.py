"""
Полная оценка обученной модели на test-сплите.

Считает метрики из §3.2–3.3 математической модели PsychMonitor:
  - Accuracy (§3.2.1)
  - Macro F1 / Weighted F1 (§3.2.2, §3.2.3)
  - Confusion Matrix (§3.2.4) + визуализация
  - AUC-ROC macro (one-vs-rest, §3.3.1)
  - Log Loss (§3.3.2)
  - Expected Calibration Error (ECE, §3.3.3)
  - Per-class precision/recall/f1
  - Анализ эффекта порога τ_conf = 0.4 (§2.2.3)
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import load_config, PROJECT_ROOT


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """ECE (§3.3.3): сумма |acc(B_m) − conf(B_m)| по бинам уверенности."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    N = len(labels)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf >= lo) & (conf < hi if hi < 1 else conf <= hi)
        if mask.sum() == 0:
            continue
        acc_m = correct[mask].mean()
        conf_m = conf[mask].mean()
        ece += (mask.sum() / N) * abs(acc_m - conf_m)
    return float(ece)


@torch.no_grad()
def predict_logits(model, tokenizer, texts: list[str], device: torch.device, max_length: int, batch_size: int = 64) -> np.ndarray:
    model.eval()
    all_logits = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(batch, truncation=True, max_length=max_length, padding=True, return_tensors="pt").to(device)
        logits = model(**enc).logits.cpu().numpy()
        all_logits.append(logits)
    return np.concatenate(all_logits, axis=0)


def main() -> None:
    cfg = load_config()
    model_dir = PROJECT_ROOT / cfg["paths"]["model_dir"]
    reports_dir = PROJECT_ROOT / cfg["paths"]["reports_dir"]
    reports_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(device)

    test_df = pd.read_csv(PROJECT_ROOT / cfg["data"]["test_path"])
    texts = test_df["sentence_ru"].astype(str).tolist()
    y_true = test_df["label_id"].values.astype(int)

    logits = predict_logits(model, tokenizer, texts, device, cfg["model"]["max_length"])
    probs = softmax(logits)
    y_pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)

    num_labels = cfg["model"]["num_labels"]
    class_names = [cfg["classes"][i + 1] for i in range(num_labels)]

    # === Метрики ===
    metrics = {
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "log_loss": float(log_loss(y_true, probs, labels=list(range(num_labels)))),
        "ece_10bins": expected_calibration_error(probs, y_true, n_bins=10),
        "auc_roc_macro_ovr": float(
            roc_auc_score(y_true, probs, multi_class="ovr", average="macro", labels=list(range(num_labels)))
        ),
    }

    # === Эффект порога уверенности τ_conf = 0.4 ===
    tau = cfg["inference"]["confidence_threshold"]
    low_conf_mask = conf < tau
    y_pred_thresh = y_pred.copy()
    y_pred_thresh[low_conf_mask] = num_labels - 1  # класс 6 (unknown) → label_id = 5
    metrics["tau_conf"] = tau
    metrics["low_conf_rate"] = float(low_conf_mask.mean())
    metrics["accuracy_with_tau"] = float(accuracy_score(y_true, y_pred_thresh))
    metrics["f1_macro_with_tau"] = float(
        f1_score(y_true, y_pred_thresh, average="macro", zero_division=0)
    )

    # === Per-class отчёт ===
    report = classification_report(
        y_true, y_pred, target_names=class_names, digits=4, zero_division=0, output_dict=True,
    )
    with open(reports_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"summary": metrics, "per_class": report}, f, indent=2, ensure_ascii=False)

    # === Confusion Matrix ===
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_labels)))
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(reports_dir / "confusion_matrix.csv", encoding="utf-8")

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=True)
    plt.title("Confusion Matrix — блок C PsychMonitor (test)")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(reports_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    # === Распределение уверенности ===
    plt.figure(figsize=(8, 4))
    plt.hist(conf, bins=20, color="steelblue", edgecolor="white")
    plt.axvline(tau, color="red", linestyle="--", label=f"τ_conf = {tau}")
    plt.xlabel("max softmax probability (confidence)")
    plt.ylabel("count")
    plt.title("Распределение уверенности модели на test")
    plt.legend()
    plt.tight_layout()
    plt.savefig(reports_dir / "confidence_histogram.png", dpi=150)
    plt.close()

    # === Вывод ===
    print("\n=== Метрики на test ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:25s}: {v:.4f}")
        else:
            print(f"  {k:25s}: {v}")

    print("\n=== Per-class F1 ===")
    for name in class_names:
        r = report[name]
        print(f"  {name:10s}  P={r['precision']:.3f}  R={r['recall']:.3f}  F1={r['f1-score']:.3f}  (n={int(r['support'])})")

    print(f"\nОтчёты сохранены в: {reports_dir}")
    print("  - test_metrics.json")
    print("  - confusion_matrix.csv / .png")
    print("  - confidence_histogram.png")

    # === Проверка целевых порогов PDF ===
    print("\n=== Сверка с целевыми порогами мат. модели ===")
    checks = [
        ("Macro F1 ≥ 0.70 (§3.2.2)", metrics["f1_macro"], 0.70),
        ("AUC-ROC ≥ 0.85 (§3.3.1)", metrics["auc_roc_macro_ovr"], 0.85),
    ]
    for name, val, target in checks:
        ok = "✓" if val >= target else "✗"
        print(f"  {ok} {name}: {val:.4f}")


if __name__ == "__main__":
    main()
