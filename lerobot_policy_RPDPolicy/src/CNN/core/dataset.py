from pathlib import Path 

import torch 
from torch.utils.data import Dataset as TorchDataset 
from torch.utils.data import DataLoader 
from torchvision.io import read_image 
from torchvision.transforms import v2

class DetectionDataset(TorchDataset): 
    def __init__(
            self,
            root: Path,
            size: tuple[int, int],
            one_indexed_labels: bool = True,
    ):
        self.one_indexed_labels = one_indexed_labels
        self.images_dir = root / "images"
        self.labels_dir = root / "labels"


        if not self.images_dir.exists():
            raise FileNotFoundError(
                f"Images directory not found: {self.images_dir}"
            )

        extensions = {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".webp",
        }

        self.images = sorted(
            path
            for path in self.images_dir.iterdir()
            if path.suffix.lower() in extensions
        )

        self.transform = v2.Compose([
            v2.Resize(size),
            v2.ToDtype(
                torch.float32,
                scale=True,
            ),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        label_path = self.labels_dir / f"{image_path.stem}.txt"

        image = read_image(str(image_path))
        image = self.transform(image)

        labels = []

        if label_path.exists():
            with label_path.open("r") as file:
                for line in file:
                    values = line.strip().split()

                    if not values:
                        continue
                    class_id, x, y, w, h = map(float, values)

                    if self.one_indexed_labels:
                        class_id = class_id - 1

                    labels.append([
                        class_id,
                        x,
                        y,
                        w,
                        h,
                    ])

        labels = (
            torch.tensor(
                labels,
                dtype=torch.float32,
            )
            if labels
            else torch.empty(
                (0, 5),
                dtype=torch.float32,
            )
        )

        return image, labels


class Dataset:
    def __init__(
        self,
        root: str,
        size: tuple[int, int] = (640, 640),
        batch_size: int = 8,
        num_workers: int = 0,
        one_indexed_labels: bool = True,
    ):
        root = Path(root)

        train = DetectionDataset(
            root / "train",
            size,
            one_indexed_labels=one_indexed_labels,
        )

        test = DetectionDataset(
            root / "test",
            size,
            one_indexed_labels=one_indexed_labels,
        )

        self.data = {
            "train": DataLoader(
                train,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
                collate_fn=self._collate_fn,
            ),
            "test": DataLoader(
                test,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
                collate_fn=self._collate_fn,
            ),
        }

    @staticmethod
    def _collate_fn(batch):
        images, labels = zip(*batch)

        return (
            torch.stack(images),
            list(labels),
        )

    def __getitem__(self, split):
        return self.data[split]

    def __contains__(self, split):
        return split in self.data