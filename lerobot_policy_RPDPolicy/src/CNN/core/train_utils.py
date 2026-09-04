from __future__ import annotations

import random
from pathlib import Path

import torch
from pathlib import Path
from PIL import Image
from torchvision.io import read_image
from torchvision.transforms import v2
from torchvision.ops import box_iou
from ultralytics.utils import TQDM as tqdm

def prepare_batch(
    images: torch.Tensor,
    labels,
    device: torch.device,
):
    images = images.to(device, non_blocking=True,).float()
    batch_idx = []
    cls = []
    bboxes = []

    for image_idx, targets in enumerate(labels):
        if not torch.is_tensor(targets):
            targets = torch.as_tensor(targets)

        if targets.numel() == 0:
            continue

        targets = targets.to(device)
        n = targets.shape[0]
        batch_idx.append(torch.full((n,), image_idx, device=device, dtype=torch.long, ))
        cls.append(targets[:, 0:1].float())
        bboxes.append(targets[:, 1:5].float())

    if batch_idx:
        batch_idx = torch.cat(batch_idx)
        cls = torch.cat(cls)
        bboxes = torch.cat(bboxes)

    else:
        batch_idx = torch.empty(
            0,
            device=device,
            dtype=torch.long,
        )

        cls = torch.empty(
            (0, 1),
            device=device,
            dtype=torch.float32,
        )

        bboxes = torch.empty(
            (0, 4),
            device=device,
            dtype=torch.float32,
        )

    return {
        "img": images,
        "batch_idx": batch_idx,
        "cls": cls,
        "bboxes": bboxes,
    }


def xywhn_to_xyxy(
    boxes: torch.Tensor,
    height: int,
    width: int,
):
    xyxy = torch.empty_like(boxes)

    x = boxes[:, 0] * width
    y = boxes[:, 1] * height
    w = boxes[:, 2] * width
    h = boxes[:, 3] * height

    xyxy[:, 0] = x - w / 2
    xyxy[:, 1] = y - h / 2
    xyxy[:, 2] = x + w / 2
    xyxy[:, 3] = y + h / 2

    return xyxy


@torch.no_grad()
def save_epoch_visualizations(
    model,
    test_loader,
    epoch: int,
    output_dir: Path,
    count: int = 10,
):
    dataset = test_loader.dataset
    if len(dataset) == 0:
        return

    count = min(count, len(dataset),)

    indices = random.sample(range(len(dataset)), count, )
    epoch_dir = (output_dir / f"epoch_{epoch}" )
    epoch_dir.mkdir(parents=True, exist_ok=True, )

    for i, index in enumerate(indices):
        image, _ = dataset[index]
        viz(model, image, output_dir=epoch_dir, filename=f"{i + 1}.jpg")


@torch.no_grad()
def evaluate_iou(
    model,
    test_loader,
    device,
):
    model.eval()
    total_iou = 0.0
    total_boxes = 0
    progress = tqdm(
        test_loader,
        desc="Evaluating IoU",
        leave=False,
    )
    for images, labels in progress:
        images = images.to(device, non_blocking=True,).float()

        detections = model.detect(images)
        _, _, height, width = images.shape

        for prediction, targets in zip(detections, labels,):
            if not torch.is_tensor(targets):
                targets = torch.as_tensor(targets)
            targets = targets.to(device)

            if targets.numel() == 0:
                continue

            gt_classes = (targets[:, 0].long())
            gt_boxes = xywhn_to_xyxy(targets[:, 1:5].float(), height, width, )
            total_boxes += gt_boxes.shape[0]

            if prediction.numel() == 0:
                continue

            pred_boxes = prediction[:, :4]
            pred_classes = (prediction[:, 5].long())
            ious = box_iou(gt_boxes, pred_boxes, )
            same_class = (gt_classes[:, None] == pred_classes[None, :])
            ious = ious.masked_fill(~same_class,0.0,)
            best_iou = (ious.max(dim=1).values)
            total_iou += (best_iou.sum().item())

    if total_boxes == 0:
        return 0.0

    return (total_iou / total_boxes) * 100.0

def save_checkpoint(
    model, 
    optimizer, 
    output_dir: str | Path, 
    epoch: int, 
    avg_loss: float, 
    avg_box_loss: float, 
    avg_cls_loss: float, 
    avg_dfl_loss: float
):
    """
        Save a complete training checkpoint. 

        The chekcpoint contains: 
            model_state_dict - Model weights
            feature_extractor_state_dict = Weights of the feature extractor only 
            detect_state_dict - Weights of ultralytics detection head 
            optimizer_state_dict - optimiser settings for resuming training 
        
        The feature and detection state dict are saved seperately, 
        so that they can be used seperately in the Policy. 
    """
    output_dir = Path(output_dir) 

    checkpoint_dir = output_dir / "checkpoints" 

    checkpoint_dir.mkdir(
        parents=True, 
        exist_ok=True, 
    )

    checkpoint = {
        "model_state_dict": model.state_dict(), 
        "feature_extractor_state_dict": model.feature_extractor.state_dict(),
        "detector_state_dict": model.detector.state_dict(),
        "optimizer_state_dict" : (
            optimizer.state_dict() 
            if optimizer is not None 
            else None 
        ), 
        "model_config": model.model_config,

        "epoch" : epoch, 
        "loss" : avg_loss, 
        "box_loss": avg_box_loss, 
        "cls_loss": avg_cls_loss, 
        "dfl_loss": avg_dfl_loss, 

        "num_classes": model.num_classes, 
        "names": model.names, 
        "image_size": model.image_size,
    }

    torch.save(
        checkpoint, 
        checkpoint_dir / "last.pt"
    )

    torch.save(
        checkpoint,
        checkpoint_dir / f"epoch_{epoch}.pt"
    )


def load_image(model, source):
    """Load and preprocess image from path or tensor."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        image = read_image(str(path))
        path_name = str(path)
    elif torch.is_tensor(source):
        image = source.detach().cpu()
        path_name = "image"
    else:
        raise TypeError("source must be a file path or a torch.Tensor")

    if image.ndim == 4:
        if image.shape[0] != 1:
            raise ValueError("Tensor source must contain one image.")
        image = image.squeeze(0)

    if image.ndim != 3:
        raise ValueError("Image must have shape [C, H, W].")

    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)

    if image.shape[0] == 4:
        image = image[:3]

    if image.dtype == torch.uint8:
        original = image.clone()
        tensor = image.float() / 255.0
    else:
        tensor = image.float()
        if tensor.max() <= 1.0:
            original = (tensor.clamp(0, 1) * 255.0).round().to(torch.uint8)
        else:
            original = (tensor.clamp(0, 255.0)).round().to(torch.uint8)
            tensor = tensor / 255.0

    tensor = v2.Resize(model.image_size)(tensor).unsqueeze(0)
    return original, tensor, path_name


@torch.no_grad()
def viz(model, source, output_dir: str | Path = "./", filename: str | None = None,
        conf_threshold: float | None = None, iou_threshold: float | None = None):
    """Visualize predictions on image and save."""
    result = model.predict(source, conf_threshold=conf_threshold, iou_threshold=iou_threshold)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        source_path = Path(result.path)
        filename = "prediction.jpg" if source_path.name == "image" else f"{source_path.stem}_prediction.jpg"

    output_path = output_dir / filename
    Image.fromarray(result.plot()).save(output_path)
    return output_path