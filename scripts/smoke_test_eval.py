from __future__ import annotations

from pathlib import Path
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "mrleyes" / "data"
MODELS_DIR = PROJECT_ROOT / "models"
BEST_PATH = MODELS_DIR / "eye_model_best.pth"

BATCH_SIZE = 32
NUM_WORKERS = 0
INPUT_SIZE = 224

GRAYSCALE = True
MEAN = (0.5,)
STD = (0.5,)


def build_model():
    if GRAYSCALE:
        m = models.resnet18(weights=None)
        m.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    else:
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    m.fc = nn.Linear(m.fc.in_features, 2)
    return m


def build_transform():
    t = [transforms.Resize((INPUT_SIZE, INPUT_SIZE))]
    if GRAYSCALE:
        t.append(transforms.Grayscale(num_output_channels=1))
    t += [transforms.ToTensor(), transforms.Normalize(mean=MEAN, std=STD)]
    return transforms.Compose(t)


def main():
    print("CWD:", Path.cwd())
    print("Model path:", BEST_PATH)
    print("Model exists:", BEST_PATH.exists())

    test_dir = DATA_DIR / "test"
    print("Test dir:", test_dir)
    print("Test dir exists:", test_dir.exists())

    ds = datasets.ImageFolder(str(test_dir), transform=build_transform())
    print("Classes:", ds.classes)
    print("Test size:", len(ds))

    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    print("Batches:", len(loader))

    # ---- load model with timing ----
    model = build_model()

    t0 = time.time()
    print("Loading checkpoint... (torch.load)")
    ckpt = torch.load(BEST_PATH, map_location="cpu")
    print(f"torch.load done in {time.time() - t0:.2f}s")

    t1 = time.time()
    print("Loading state_dict...")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"], strict=True)
    elif isinstance(ckpt, dict):
        model.load_state_dict(ckpt, strict=True)
    else:
        raise RuntimeError("Checkpoint format not supported (expected dict).")
    print(f"load_state_dict done in {time.time() - t1:.2f}s")

    model.eval()

    # ---- run exactly 1 batch ----
    print("Fetching 1 batch...")
    xb, yb = next(iter(loader))
    print("Batch shapes:", xb.shape, yb.shape, "dtype:", xb.dtype)

    print("Forward pass...")
    with torch.no_grad():
        logits = model(xb)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1)

    print("logits shape:", logits.shape)
    print("probs[0]:", probs[0].tolist())
    print("pred counts:", torch.bincount(pred, minlength=2).tolist())
    print("✅ Smoke test OK (model + dataloader + forward pass working).")


if __name__ == "__main__":
    main()