from __future__ import annotations

from pathlib import Path
import time
import traceback

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# ---- Make Windows CPU evaluation stable ----
torch.set_num_threads(2)  # prevents CPU thread thrash on some machines

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "mrleyes" / "data"
TEST_DIR = DATA_DIR / "test"

MODELS_DIR = PROJECT_ROOT / "models"
BEST_PATH = MODELS_DIR / "eye_model_best.pth"

BATCH_SIZE = 32
NUM_WORKERS = 0
INPUT_SIZE = 224

# MUST match training
GRAYSCALE = True
MEAN = (0.5,)
STD = (0.5,)


def build_model():
    model = models.resnet18(weights=None)
    if GRAYSCALE:
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def build_transform():
    t = [transforms.Resize((INPUT_SIZE, INPUT_SIZE))]
    if GRAYSCALE:
        t.append(transforms.Grayscale(num_output_channels=1))
    t += [transforms.ToTensor(), transforms.Normalize(mean=MEAN, std=STD)]
    return transforms.Compose(t)


class SafeImageFolder(datasets.ImageFolder):
    """
    ImageFolder that NEVER hangs the whole eval due to a single bad image.
    If an image fails to load/transform, it logs it and returns None.
    """
    def __init__(self, root, transform=None):
        super().__init__(root=root, transform=transform)
        self.bad_files: list[str] = []

    def __getitem__(self, index):
        path, target = self.samples[index]
        try:
            sample = self.loader(path)         # PIL load
            if self.transform is not None:
                sample = self.transform(sample)
            return sample, target, path
        except Exception:
            self.bad_files.append(path)
            # return a sentinel that collate will drop
            return None


def safe_collate(batch):
    # Drop None items
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    xs, ys, paths = zip(*batch)
    return torch.stack(xs, dim=0), torch.tensor(ys, dtype=torch.long), list(paths)


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader):
    model.eval()

    tn = fp = fn = tp = 0
    correct = 0
    total = 0

    t0 = time.time()

    for i, batch in enumerate(loader):
        if batch is None:
            # whole batch was bad images (rare)
            print(f"Batch {i}: all images failed to load -> skipped", flush=True)
            continue

        xb, yb, paths = batch
        logits = model(xb)
        pred = torch.argmax(logits, dim=1)

        correct += int((pred == yb).sum().item())
        total += int(yb.numel())

        # Positive class = 1 (sleepy)
        for p, y in zip(pred.tolist(), yb.tolist()):
            if y == 0 and p == 0: tn += 1
            elif y == 0 and p == 1: fp += 1
            elif y == 1 and p == 0: fn += 1
            else: tp += 1

        # Progress prints (so you SEE it moving)
        if i % 25 == 0:
            acc = (correct / total) if total else 0.0
            mins = (time.time() - t0) / 60.0
            print(f"Batch {i}/{len(loader)} | acc={acc*100:.2f}% | {mins:.1f} min", flush=True)

    acc = correct / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return acc, precision, recall, f1, tn, fp, fn, tp


def main():
    if not TEST_DIR.exists():
        raise FileNotFoundError(f"Missing test folder: {TEST_DIR}")
    if not BEST_PATH.exists():
        raise FileNotFoundError(f"Missing model: {BEST_PATH}")

    ds = SafeImageFolder(str(TEST_DIR), transform=build_transform())
    print("Classes:", ds.classes, flush=True)
    print("Test size:", len(ds), flush=True)

    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=safe_collate,
        pin_memory=False,
    )

    model = build_model()

    print("Loading checkpoint...", flush=True)
    ckpt = torch.load(BEST_PATH, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"], strict=True)
    else:
        model.load_state_dict(ckpt, strict=True)

    print("Checkpoint loaded. Starting evaluation...", flush=True)

    acc, precision, recall, f1, tn, fp, fn, tp = evaluate(model, loader)

    print("\n===== TEST REPORT =====", flush=True)
    print(f"Accuracy:   {acc*100:.2f}%", flush=True)
    print(f"Precision(1=sleepy): {precision:.4f}", flush=True)
    print(f"Recall(1=sleepy):    {recall:.4f}", flush=True)
    print(f"F1(1=sleepy):        {f1:.4f}", flush=True)
    print(f"Confusion Matrix [[TN, FP],[FN, TP]] = [[{tn}, {fp}], [{fn}, {tp}]]", flush=True)
    print("=======================\n", flush=True)

    if ds.bad_files:
        print("⚠️ BAD IMAGES FOUND (delete these files and re-run):", flush=True)
        # print up to 50 paths (enough)
        for p in ds.bad_files[:50]:
            print(" -", p, flush=True)
        if len(ds.bad_files) > 50:
            print(f"... and {len(ds.bad_files)-50} more", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.", flush=True)
    except Exception:
        print("\nFATAL ERROR:\n", flush=True)
        traceback.print_exc()