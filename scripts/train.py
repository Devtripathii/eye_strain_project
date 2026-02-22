import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import get_model


def io_sanity_check(dataset, n=50):
    """Quickly load n samples to see if disk I/O is the bottleneck."""
    n = min(n, len(dataset))
    t0 = time.time()
    for i in range(n):
        _ = dataset[i]  # loads image + transform
    dt = time.time() - t0
    print(f"🧪 I/O check: loaded {n} samples in {dt:.2f}s ({n/max(dt,1e-6):.1f} samples/s)")


def accuracy_from_logits(logits, labels):
    _, predicted = torch.max(logits, 1)
    correct = (predicted == labels).sum().item()
    return correct, labels.size(0)


def evaluate(model, loader, device):
    model.eval()
    correct_total = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            c, t = accuracy_from_logits(outputs, labels)
            correct_total += c
            total += t
    return 100.0 * correct_total / max(1, total)


def main():
    # CPU tuning (optional but helps on some machines)
    torch.set_num_threads(max(1, os.cpu_count() // 2))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("CUDA available:", torch.cuda.is_available())
    print("CPU threads:", torch.get_num_threads())

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    train_dir = "mrleyes/data/train"
    val_dir = "mrleyes/data/val"
    test_dir = "mrleyes/data/test"

    for p in [train_dir, val_dir, test_dir]:
        if not os.path.isdir(p):
            raise FileNotFoundError(f"Missing folder: {p}")

    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=transform)

    print("Classes:", train_dataset.classes)
    print("Train size:", len(train_dataset))
    print("Validation size:", len(val_dataset))
    print("Test size:", len(test_dataset))

    # ✅ Windows-safe DataLoader settings (no multiprocessing)
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,          # smaller batch for CPU stability
        shuffle=True,
        num_workers=0,          # IMPORTANT for Windows reliability
        pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    # I/O sanity check: if this is very slow, training will "feel stuck"
    io_sanity_check(train_dataset, n=50)

    model = get_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    epochs = 2  # keep 2 for debugging; increase later
    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        epoch_t0 = time.time()
        last_print_t = time.time()

        for i, (images, labels) in enumerate(train_loader):
            batch_t0 = time.time()

            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            c, t = accuracy_from_logits(outputs, labels)
            correct += c
            total += t

            batch_dt = time.time() - batch_t0

            # Print every 10 batches OR every ~5 seconds (whichever comes first)
            if (i % 10 == 0) or ((time.time() - last_print_t) > 5):
                avg_loss = running_loss / max(1, (i + 1))
                train_acc = 100.0 * correct / max(1, total)
                print(f"Epoch {epoch+1}/{epochs} | Batch {i}/{len(train_loader)} "
                      f"| Loss {avg_loss:.4f} | Train Acc {train_acc:.2f}% | BatchTime {batch_dt:.2f}s")
                last_print_t = time.time()

            # DEBUG: if you want to confirm progress fast, stop after 200 batches
            # if i == 200:
            #     print("Stopping early for debug (200 batches).")
            #     break

        epoch_dt = time.time() - epoch_t0
        train_acc = 100.0 * correct / max(1, total)
        val_acc = evaluate(model, val_loader, device)

        print(f"✅ Epoch {epoch+1}/{epochs} done in {epoch_dt/60:.1f} min | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "eye_model_best.pth")
            print(f"💾 Saved best: eye_model_best.pth (Val Acc {best_val_acc:.2f}%)")

    torch.save(model.state_dict(), "eye_model_last.pth")
    print("💾 Saved last: eye_model_last.pth")

    test_acc = evaluate(model, test_loader, device)
    print(f"🏁 Test Accuracy: {test_acc:.2f}%")
    print("✅ Done.")


if __name__ == "__main__":
    main()