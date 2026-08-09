from __future__ import annotations

from pathlib import Path

import torch
from ultralytics.utils import TQDM as tqdm
from .train_utils import (
    prepare_batch,
    evaluate_iou,
    save_epoch_visualizations,
    save_checkpoint,
)


def fit(
    model,
    dataset,
    epochs: int = 100,
    lr: float = 1e-3,
    optimizer=None,
    device=None,
    output_dir: str | Path = "./runs",
):
    """Train the model on the dataset."""
    if "train" not in dataset:
        raise ValueError("dataset must contain 'train'.")
    if "test" not in dataset:
        raise ValueError("dataset must contain 'test'.")

    train_loader = dataset["train"]
    test_loader = dataset["test"]

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    model.to(device)
    model.detector.head.stride = model.detector.head_stride.to(device)
    model.criterion = model.init_criterion()

    if optimizer is None:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=1e-4,
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training device: {device}")
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(device))

    for epoch in range(1, epochs + 1):
        model.train()

        epoch_total_loss = 0.0
        epoch_box_loss = 0.0
        epoch_cls_loss = 0.0
        epoch_dfl_loss = 0.0

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")

        for images, labels in progress:
            batch = prepare_batch(images, labels, device)
            optimizer.zero_grad(set_to_none=True)

            predictions = model(batch["img"])
            loss, loss_items = model.criterion(predictions, batch)
            total_loss = loss.sum()

            total_loss.backward()
            optimizer.step()

            if not isinstance(loss_items, dict):
                raise TypeError(
                    "Expected v8DetectionLoss to return a dict, "
                    f"but received {type(loss_items).__name__}."
                )

            for key in ["box_loss", "cls_loss", "dfl_loss"]:
                if key not in loss_items:
                    raise KeyError(
                        f"v8DetectionLoss output does not contain '{key}'. "
                        f"Available keys: {list(loss_items.keys())}"
                    )

            box_loss = float(loss_items["box_loss"])
            cls_loss = float(loss_items["cls_loss"])
            dfl_loss = float(loss_items["dfl_loss"])

            current_total_loss = total_loss.detach().item()

            epoch_total_loss += current_total_loss
            epoch_box_loss += box_loss
            epoch_cls_loss += cls_loss
            epoch_dfl_loss += dfl_loss

            progress.set_postfix(
                total=f"{current_total_loss:.4f}",
                box=f"{box_loss:.4f}",
                cls=f"{cls_loss:.4f}",
                dfl=f"{dfl_loss:.4f}",
            )

        num_batches = max(len(train_loader), 1)

        avg_loss = epoch_total_loss / num_batches
        avg_box_loss = epoch_box_loss / num_batches
        avg_cls_loss = epoch_cls_loss / num_batches
        avg_dfl_loss = epoch_dfl_loss / num_batches

        print(
            f"Epoch [{epoch}/{epochs}] "
            f"total={avg_loss:.4f} "
            f"box={avg_box_loss:.4f} "
            f"cls={avg_cls_loss:.4f} "
            f"dfl={avg_dfl_loss:.4f}"
        )

        if epoch % 5 == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                output_dir=output_dir,
                epoch=epoch,
                avg_loss=avg_loss,
                avg_box_loss=avg_box_loss,
                avg_cls_loss=avg_cls_loss,
                avg_dfl_loss=avg_dfl_loss,
            )

            mean_iou = evaluate_iou(model, test_loader, device=device)
            print(f"Epoch [{epoch}/{epochs}] Mean IoU: {mean_iou:.2f}%")

            save_epoch_visualizations(model, test_loader, epoch, output_dir, count=10)

    return model
