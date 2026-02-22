from __future__ import annotations

from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from src.ml.eval_utils import evaluate_binary_classifier

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "mrleyes" / "data"

MODELS_DIR = PROJECT_ROOT / "models"
BEST_PATH = MODELS_DIR / "eye_model_best.pth"   # <-- your saved best
BATCH_SIZE = 32
NUM_WORKERS = 0
INPUT_SIZE = 224

# MUST match your training
GRAYSCALE = True
MEAN = (0.5,)
STD = (0.5,)

def build_model():
    if GRAYSCALE:
        model = models.resnet18(weights=None)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    else:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    model.fc = nn.Linear(model.fc.in_features, 2)
    return model

def build_transform():
    t = [
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    ]
    if GRAYSCALE:
        t.append(transforms.Grayscale(num_output_channels=1))
    t += [
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ]
    return transforms.Compose(t)

def main():
    if not BEST_PATH.exists():
        raise FileNotFoundError(f"Missing model: {BEST_PATH}")

    test_dir = DATA_DIR / "test"
    if not test_dir.exists():
        raise FileNotFoundError(f"Missing test folder: {test_dir}")

    test_ds = datasets.ImageFolder(str(test_dir), transform=build_transform())
    print("Classes:", test_ds.classes)
    print("Test size:", len(test_ds))

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    model = build_model()

    ckpt = torch.load(BEST_PATH, map_location="cpu")
    # supports either raw state_dict or {"state_dict":...}
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"], strict=True)
    else:
        model.load_state_dict(ckpt, strict=True)

    rep = evaluate_binary_classifier(model, test_loader, device="cpu")

    print("\n===== TEST REPORT =====")
    print(f"Accuracy:   {rep.accuracy*100:.2f}%")
    print(f"Precision(1=sleepy): {rep.precision_pos:.4f}")
    print(f"Recall(1=sleepy):    {rep.recall_pos:.4f}")
    print(f"F1(1=sleepy):        {rep.f1_pos:.4f}")
    print(f"Confusion Matrix [[TN, FP],[FN, TP]] = [[{rep.tn}, {rep.fp}], [{rep.fn}, {rep.tp}]]")
    print("=======================\n")

    # Key safety signal:
    if rep.fn > rep.fp:
        print("⚠️ Warning: FN > FP (missing sleepy more often than false alarms). Consider threshold tuning.")

if __name__ == "__main__":
    main()