from __future__ import annotations

import torch
from .train_utils import load_image


@torch.no_grad()
def predict(
    model,
    source,
    conf_threshold: float | None = None,
    iou_threshold: float | None = None,
):
    """Run inference on image and return detections."""
    original, tensor, path = load_image(model, source)
    tensor = tensor.to(model.device)

    detections = model.detect(
        tensor,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
    )

    detection = detections[0].detach().cpu().clone()

    original_h, original_w = original.shape[-2:]
    input_h, input_w = tensor.shape[-2:]

    if detection.numel():
        detection[:, [0, 2]] *= (original_w / input_w)
        detection[:, [1, 3]] *= (original_h / input_h)
        detection[:, [0, 2]].clamp_(0, original_w)
        detection[:, [1, 3]].clamp_(0, original_h)

    image = original.permute(1, 2, 0).contiguous().numpy()

    return model.detector.results(
        detections=detection,
        image=image,
        path=path,
        names=model.names,
    )
