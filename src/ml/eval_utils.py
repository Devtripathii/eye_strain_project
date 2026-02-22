from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class ClassificationReport:
    accuracy: float
    precision_pos: float
    recall_pos: float
    f1_pos: float
    tn: int
    fp: int
    fn: int
    tp: int

    def to_dict(self) -> Dict[str, float | int]:
        return {
            "accuracy": self.accuracy,
            "precision_pos": self.precision_pos,
            "recall_pos": self.recall_pos,
            "f1_pos": self.f1_pos,
            "tn": self.tn,
            "fp": self.fp,
            "fn": self.fn,
            "tp": self.tp,
        }


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b != 0 else 0.0


@torch.no_grad()
def evaluate_binary_classifier(model: torch.nn.Module, loader, device="cpu") -> ClassificationReport:
    model.eval()

    tn = fp = fn = tp = 0
    correct = 0
    total = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        logits = model(xb)
        pred = torch.argmax(logits, dim=1)

        correct += int((pred == yb).sum().item())
        total += int(yb.numel())

        for p, y in zip(pred.view(-1), yb.view(-1)):
            p = int(p.item())
            y = int(y.item())
            if y == 0 and p == 0:
                tn += 1
            elif y == 0 and p == 1:
                fp += 1
            elif y == 1 and p == 0:
                fn += 1
            elif y == 1 and p == 1:
                tp += 1

    acc = _safe_div(correct, total)
    precision = _safe_div(tp, (tp + fp))
    recall = _safe_div(tp, (tp + fn))
    f1 = _safe_div(2 * precision * recall, (precision + recall))

    return ClassificationReport(
        accuracy=acc,
        precision_pos=precision,
        recall_pos=recall,
        f1_pos=f1,
        tn=tn,
        fp=fp,
        fn=fn,
        tp=tp,
    )