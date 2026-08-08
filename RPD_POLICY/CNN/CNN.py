from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from PIL import Image
import random

import torch
import torch.nn as nn
from torchvision.io import read_image
from torchvision.ops import box_iou
from torchvision.transforms import v2

from ultralytics.nn.modules.head import Detect
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.utils.nms import non_max_suppression
from ultralytics.engine.results import Boxes, Results
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils import TQDM as tqdm 

from .features import FeatureExtractor

class CNN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        backbone_name: str = "resnet50",
        pretrained: bool = True,
        neck: str = "fpn",
        feature_channels: int = 256,
        bifpn_layers: int = 3,
        use_cbam: bool = False,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_detections: int = 300,
        image_size: tuple[int, int] = (640, 640),
        class_names: dict | list | None = None,
    ):
        super().__init__()

        self.num_classes = num_classes 
        self.conf_threshold = conf_threshold 
        self.iou_threshold = iou_threshold 
        self.max_detection = max_detections 
        self.image_size = image_size 

        if class_names is None: 
            self.names = {
                i: str(i)
                for i in range(num_classes) 
            }
        elif isinstance(class_names, list): 
            self.names = { 
                i: name
                for i, name in enumerate(class_names) 
            }
        else: 
            self.names = class_names 

        self.feature_extractor = FeatureExtractor(
            backbone_name=backbone_name,
            pretrained=pretrained,
            neck=neck,
            out_channels=feature_channels,
            bifpn_layers=bifpn_layers,
            use_cbam=use_cbam,
        )

        num_levels = self.feature_extractor.num_output_levels()
        channels = (feature_channels,) * num_levels

        self.detect_head = Detect(
            nc=num_classes,
            ch=channels,
        )

        self.detect_head.stride = torch.tensor(
            self.feature_extractor.out_strides,
            dtype=torch.float32,
        )

        self.detect_head.bias_init()

        self.args = SimpleNamespace(
            box=7.5,
            cls=0.5,
            dfl=1.5,
        )

        self.criterion = None

        self.trackers = {
            "bytetrack": BYTETracker(
                args=self._bytetrack_args()
            ),
            "botsort": BOTSORT(
                args=self._botsort_args()
            ),
        }

    @property
    def model(self):
        return [
            self.feature_extractor,
            self.detect_head,
        ]

    @property
    def device(self):
        return next(self.parameters()).device

    def init_criterion(self):
        return v8DetectionLoss(self)

    @staticmethod
    def _bytetrack_args():
        return SimpleNamespace(
            tracker_type="bytetrack",
            track_high_thresh=0.25,
            track_low_thresh=0.10,
            new_track_thresh=0.25,
            track_buffer=30,
            match_thresh=0.80,
            fuse_score=True,
        )

    @staticmethod
    def _botsort_args():
        return SimpleNamespace(
            tracker_type="botsort",
            track_high_thresh=0.25,
            track_low_thresh=0.10,
            new_track_thresh=0.25,
            track_buffer=30,
            match_thresh=0.80,
            fuse_score=True,
            gmc_method="sparseOptFlow",
            proximity_thresh=0.5,
            appearance_thresh=0.8,
            with_reid=False,
            model="auto",
        )

    def forward(self, x: torch.Tensor):
        return self.detect_head(list(self.feature_extractor(x)))

    @torch.no_grad() 
    def detect(
        self, 
        x: torch.Tensor, 
        conf_threshold: float | None = None, 
        iou_threshold: float | None = None 
    ):
        self.eval() 

        if conf_threshold is None: 
            conf_threshold = self.conf_threshold 

        if iou_threshold is None: 
            iou_threshold = self.iou_threshold 

        predictions = self.forward(x) 

        if isinstance(predictions, tuple): 
            predictions = predictions[0] 

        return non_max_suppression(
            prediction = predictions, 
            conf_thres=conf_threshold, 
            iou_thres=iou_threshold, 
            max_det=self.max_detection,
            nc = self.num_classes, 
        )

    def _load_image(self, source): 
        if isinstance(source, (str, Path)): 
            path = Path(source) 
            image = read_image(str(path))
            path_name = str(path) 

        elif torch.is_tensor(source): 
            image = source.detach().cpu() 
            path_name = "image" 

        else: 
            raise TypeError(
                "source must be a file path or a torch.Tensor"
            )


        if image.ndim == 4:
            if image.shape[0] != 1:
                raise ValueError(
                    "Tensor source must contain one image."
                )

            image = image.squeeze(0)

        if image.ndim != 3:
            raise ValueError(
                "Image must have shape [C, H, W]."
            )

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
                original = (
                    tensor.clamp(0, 1) * 255.0
                ).round().to(torch.uint8) 
            else: 
                original = (
                    tensor.clamp(0, 255.0)
                ).round().to(torch.uint8) 
                tensor = tensor / 255.0 

        tensor = v2.Resize(
            self.image_size
        )(tensor).unsqueeze(0)

        return original, tensor, path_name


    @torch.no_grad()
    def predict(
        self,
        source,
        conf_threshold: float | None = None,
        iou_threshold: float | None = None,
    ):
        original, tensor, path = self._load_image(source)
    
        tensor = tensor.to(self.device)
    
        detections = self.detect(
            tensor,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
        )
    
        detection = detections[0].detach().cpu().clone()
    
        original_h, original_w = original.shape[-2:]
        input_h, input_w = tensor.shape[-2:]
    
        if detection.numel():
            detection[:, [0, 2]] *= (
                original_w / input_w
            )
    
            detection[:, [1, 3]] *= (
                original_h / input_h
            )
    
            detection[:, [0, 2]].clamp_(
                0,
                original_w,
            )
    
            detection[:, [1, 3]].clamp_(
                0,
                original_h,
            )
    
        image = (
            original
            .permute(1, 2, 0)
            .contiguous()
            .numpy()
        )
        
        return Results(
            orig_img=image,
            path=path,
            names=self.names,
            boxes=detection,
        )

    @torch.no_grad()
    def viz(
        self,
        source,
        output_dir: str | Path = "./",
        filename: str | None = None,
        conf_threshold: float | None = None,
        iou_threshold: float | None = None,
    ):
        result = self.predict(
            source,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
        )
    
        output_dir = Path(output_dir)
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
    
        if filename is None:
            source_path = Path(result.path)
    
            if source_path.name == "image":
                filename = "prediction.jpg"
            else:
                filename = f"{source_path.stem}_prediction.jpg"
    
        output_path = output_dir / filename
    
        plotted = result.plot()
    
        Image.fromarray(
            plotted
        ).save(
            output_path
        )
    
        return output_path

    @torch.no_grad()
    def track(
        self,
        x: torch.Tensor,
        tracker: str = "bytetrack",
        img=None,
    ):
        if x.ndim != 4 or x.shape[0] != 1:
            raise ValueError(
                "track() expects x with shape [1, C, H, W]."
            )

        tracker = tracker.lower()

        if tracker not in self.trackers:
            raise ValueError(
                f"Unknown tracker '{tracker}'. "
                f"Available trackers: "
                f"{list(self.trackers.keys())}"
            )

        detections = self.detect(x)

        detection = (
            detections[0]
            .detach()
            .cpu()
        )

        results = Boxes(
            detection,
            orig_shape=x.shape[-2:],
        )

        return self.trackers[tracker].update(
            results,
            img=img,
        )

    def reset_tracker(
        self,
        tracker: str | None = None,
    ):
        if tracker is None:
            for tracking_algorithm in self.trackers.values():
                tracking_algorithm.reset()

            return

        tracker = tracker.lower()

        if tracker not in self.trackers:
            raise ValueError(
                f"Unknown tracker '{tracker}'. "
                f"Available trackers: "
                f"{list(self.trackers.keys())}"
            )

        self.trackers[tracker].reset()

    def _prepare_batch(
        self,
        images: torch.Tensor,
        labels,
        device: torch.device,
    ):
        images = images.to(
            device,
            non_blocking=True,
        ).float()

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

            batch_idx.append(
                torch.full(
                    (n,),
                    image_idx,
                    device=device,
                    dtype=torch.long,
                )
            )

            cls.append(
                targets[:, 0:1].float()
            )

            bboxes.append(
                targets[:, 1:5].float()
            )

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

    @staticmethod
    def _xywhn_to_xyxy(
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
    def _save_epoch_visualizations(
        self,
        test_loader,
        epoch: int,
        output_dir: Path,
        count: int = 10,
    ):
        dataset = test_loader.dataset

        if len(dataset) == 0:
            return

        count = min(
            count,
            len(dataset),
        )

        indices = random.sample(
            range(len(dataset)),
            count,
        )

        epoch_dir = (
            output_dir
            / f"epoch_{epoch}"
        )

        epoch_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for i, index in enumerate(indices):
            image, _ = dataset[index]

            self.viz(
                image,
                output_dir=epoch_dir,
                filename=f"{i + 1}.jpg",
            )

    @torch.no_grad()
    def evaluate_iou(
        self,
        test_loader,
        device,
    ):
        self.eval()

        total_iou = 0.0
        total_boxes = 0

        progress = tqdm(
            test_loader,
            desc="Evaluating IoU",
            leave=False,
        )

        for images, labels in progress:
            images = images.to(
                device,
                non_blocking=True,
            ).float()

            detections = self.detect(images)

            _, _, height, width = images.shape

            for prediction, targets in zip(
                detections,
                labels,
            ):
                if not torch.is_tensor(targets):
                    targets = torch.as_tensor(targets)

                targets = targets.to(device)

                if targets.numel() == 0:
                    continue

                gt_classes = (
                    targets[:, 0]
                    .long()
                )

                gt_boxes = self._xywhn_to_xyxy(
                    targets[:, 1:5].float(),
                    height,
                    width,
                )

                total_boxes += gt_boxes.shape[0]

                if prediction.numel() == 0:
                    continue

                pred_boxes = prediction[:, :4]

                pred_classes = (
                    prediction[:, 5]
                    .long()
                )

                ious = box_iou(
                    gt_boxes,
                    pred_boxes,
                )

                same_class = (
                    gt_classes[:, None]
                    == pred_classes[None, :]
                )

                ious = ious.masked_fill(
                    ~same_class,
                    0.0,
                )

                best_iou = (
                    ious
                    .max(dim=1)
                    .values
                )

                total_iou += (
                    best_iou
                    .sum()
                    .item()
                )

        if total_boxes == 0:
            return 0.0

        return (
            total_iou
            / total_boxes
        ) * 100.0

    def _save_checkpoint(
        self,
        output_dir: Path,
        epoch: int,
        optimizer,
        avg_loss: float,
        avg_bb_loss: float,
        avg_cls_loss: float,
        avg_dfl_loss: float,
    ):
        checkpoint_dir = (
            output_dir
            / "checkpoints"
        )

        checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": (
                optimizer.state_dict()
                if optimizer is not None
                else None
            ),
            "loss": avg_loss,
            "box_loss": avg_bb_loss,
            "cls_loss": avg_cls_loss,
            "dfl_loss": avg_dfl_loss,
            "num_classes": self.num_classes,
            "names": self.names,
            "image_size": self.image_size,
        }

        torch.save(
            checkpoint,
            checkpoint_dir / "latest.pt",
        )

        torch.save(
            checkpoint,
            checkpoint_dir / f"epoch_{epoch}.pt",
        )

    def fit(
        self,
        dataset,
        epochs: int = 100,
        lr: float = 1e-3,
        optimizer=None,
        device=None,
        output_dir: str | Path = "./runs",
    ):
        if "train" not in dataset:
            raise ValueError(
                "dataset must contain 'train'."
            )

        if "test" not in dataset:
            raise ValueError(
                "dataset must contain 'test'."
            )

        train_loader = dataset["train"]
        test_loader = dataset["test"]

        if device is None:
            device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        else:
            device = torch.device(device)

        self.to(device)

        self.detect_head.stride = (
            self.detect_head.stride.to(device)
        )

        self.criterion = self.init_criterion()

        if optimizer is None:
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=lr,
                weight_decay=1e-4,
            )

        output_dir = Path(output_dir)

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        print(f"Training device: {device}")

        if device.type == "cuda":
            print(
                "GPU:",
                torch.cuda.get_device_name(device),
            )

        for epoch in range(1, epochs + 1):
            self.train()

            epoch_total_loss = 0.0
            epoch_box_loss = 0.0
            epoch_cls_loss = 0.0
            epoch_dfl_loss = 0.0

            progress = tqdm(
                train_loader,
                desc=f"Epoch {epoch}/{epochs}",
            )

            for images, labels in progress:
                batch = self._prepare_batch(
                    images,
                    labels,
                    device,
                )

                optimizer.zero_grad(
                    set_to_none=True
                )

                predictions = self(
                    batch["img"]
                )

                loss, loss_items = self.criterion(
                    predictions,
                    batch,
                )

                total_loss = loss.sum()

                total_loss.backward()

                optimizer.step()

                if not isinstance(loss_items, dict):
                    raise TypeError(
                        "Expected v8DetectionLoss to return "
                        "a dictionary of loss components, "
                        f"but received {type(loss_items).__name__}."
                    )

                if "box_loss" not in loss_items:
                    raise KeyError(
                        "v8DetectionLoss output does not contain "
                        "'box_loss'. Available keys: "
                        f"{list(loss_items.keys())}"
                    )

                if "cls_loss" not in loss_items:
                    raise KeyError(
                        "v8DetectionLoss output does not contain "
                        "'cls_loss'. Available keys: "
                        f"{list(loss_items.keys())}"
                    )

                if "dfl_loss" not in loss_items:
                    raise KeyError(
                        "v8DetectionLoss output does not contain "
                        "'dfl_loss'. Available keys: "
                        f"{list(loss_items.keys())}"
                    )

                box_loss = float(
                    loss_items["box_loss"]
                )

                cls_loss = float(
                    loss_items["cls_loss"]
                )

                dfl_loss = float(
                    loss_items["dfl_loss"]
                )

                current_total_loss = (
                    total_loss
                    .detach()
                    .item()
                )

                epoch_total_loss += (
                    current_total_loss
                )

                epoch_box_loss += box_loss
                epoch_cls_loss += cls_loss
                epoch_dfl_loss += dfl_loss

                progress.set_postfix(
                    total=f"{current_total_loss:.4f}",
                    box=f"{box_loss:.4f}",
                    cls=f"{cls_loss:.4f}",
                    dfl=f"{dfl_loss:.4f}",
                )

            num_batches = max(
                len(train_loader),
                1,
            )

            avg_loss = (
                epoch_total_loss
                / num_batches
            )

            avg_box_loss = (
                epoch_box_loss
                / num_batches
            )

            avg_cls_loss = (
                epoch_cls_loss
                / num_batches
            )

            avg_dfl_loss = (
                epoch_dfl_loss
                / num_batches
            )

            print(
                f"Epoch [{epoch}/{epochs}] "
                f"total={avg_loss:.4f} "
                f"box={avg_box_loss:.4f} "
                f"cls={avg_cls_loss:.4f} "
                f"dfl={avg_dfl_loss:.4f}"
            )

            if epoch % 5 == 0:
                self._save_checkpoint(
                    output_dir=output_dir,
                    epoch=epoch,
                    optimizer=optimizer,
                    avg_loss=avg_loss,
                    avg_bb_loss=avg_box_loss,
                    avg_cls_loss=avg_cls_loss,
                    avg_dfl_loss=avg_dfl_loss,
                )

                mean_iou = self.evaluate_iou(
                    test_loader,
                    device=device,
                )

                print(
                    f"Epoch [{epoch}/{epochs}] "
                    f"Mean IoU: {mean_iou:.2f}%"
                )

                self._save_epoch_visualizations(
                    test_loader,
                    epoch,
                    output_dir,
                    count=10,
                )

        return self
