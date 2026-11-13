"""
dataset.py - Modulo Dataset e DataLoader per Pascal VOC e MS COCO

Funzionalità:
1. Supporto Pascal VOC (2007+2012) come default (20 classi) tramite torchvision.datasets.VOCDetection
   e fallback automatico per parsing XML locale.
2. Supporto MS COCO (80 classi) e dataset in formato YOLO (immagini + file .txt).
3. Trasformazioni detection-safe:
   - Letterbox resize con preservazione dell'aspect ratio e padding grigio (114, 114, 114).
   - Random horizontal flip con ricalcolo esatto delle coordinate dei bounding box (cx = 1.0 - cx).
   - Color jitter (luminosità, contrasto, saturazione).
   - Normalizzazione ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).
4. Collate function per formattare i target in un tensore unificato (N_total, 6):
   [batch_idx, class_id, cx, cy, w, h] con coordinate normalizzate in [0, 1].
"""

import os
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import xml.etree.ElementTree as ET

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from torchvision.datasets import VOCDetection
from PIL import Image

# 20 classi Pascal VOC
VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]
VOC_CLASS_TO_IDX = {name: i for i, name in enumerate(VOC_CLASSES)}

# 80 classi MS COCO
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]
COCO_CLASS_TO_IDX = {name: i for i, name in enumerate(COCO_CLASSES)}

# Statistiche standard ImageNet per VGG16-BN
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def letterbox_image(
    image: Image.Image,
    boxes: torch.Tensor,
    target_size: int = 416,
) -> Tuple[Image.Image, torch.Tensor]:
    """
    Ridimensiona l'immagine mantenendo le proporzioni (aspect ratio) e aggiunge padding grigio (114, 114, 114).
    Aggiorna coerentemente le coordinate dei bounding box (in formato xmin, ymin, xmax, ymax in pixel).
    Ritorna:
        - Immagine PIL ridimensionata (target_size, target_size)
        - Tensore box in formato normalizzato [cx, cy, w, h] in [0, 1]
    """
    orig_w, orig_h = image.size
    scale = min(target_size / orig_w, target_size / orig_h)
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))

    # Ridimensionamento bilineare dell'immagine
    resized_img = image.resize((new_w, new_h), Image.BILINEAR)

    # Creazione del canvas con padding grigio neutro
    padded_img = Image.new("RGB", (target_size, target_size), (114, 114, 114))
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    padded_img.paste(resized_img, (pad_x, pad_y))

    if boxes.numel() == 0:
        return padded_img, torch.zeros((0, 4), dtype=torch.float32)

    # Aggiornamento coordinate pixel (xmin, ymin, xmax, ymax)
    boxes = boxes.clone()
    boxes[:, 0] = boxes[:, 0] * scale + pad_x  # xmin
    boxes[:, 1] = boxes[:, 1] * scale + pad_y  # ymin
    boxes[:, 2] = boxes[:, 2] * scale + pad_x  # xmax
    boxes[:, 3] = boxes[:, 3] * scale + pad_y  # ymax

    # Conversione in [cx, cy, w, h] normalizzati in [0, 1]
    cx = ((boxes[:, 0] + boxes[:, 2]) / 2.0) / target_size
    cy = ((boxes[:, 1] + boxes[:, 3]) / 2.0) / target_size
    w = (boxes[:, 2] - boxes[:, 0]) / target_size
    h = (boxes[:, 3] - boxes[:, 1]) / target_size

    # Clamp di sicurezza nei limiti [0, 1]
    cx = torch.clamp(cx, 0.0, 1.0)
    cy = torch.clamp(cy, 0.0, 1.0)
    w = torch.clamp(w, 0.0, 1.0)
    h = torch.clamp(h, 0.0, 1.0)

    norm_boxes = torch.stack([cx, cy, w, h], dim=1)
    return padded_img, norm_boxes


class DetectionAugmentation:
    """Pipeline di trasformazioni detection-safe con supporto a flip orizzontale e jitter colore."""

    def __init__(
        self,
        target_size: int = 416,
        is_train: bool = True,
        hflip_prob: float = 0.5,
        color_jitter_params: Optional[Dict[str, float]] = None,
    ):
        self.target_size = target_size
        self.is_train = is_train
        self.hflip_prob = hflip_prob

        if color_jitter_params and is_train:
            self.color_jitter = transforms.ColorJitter(
                brightness=color_jitter_params.get("brightness", 0.2),
                contrast=color_jitter_params.get("contrast", 0.2),
                saturation=color_jitter_params.get("saturation", 0.2),
            )
        else:
            self.color_jitter = None

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def __call__(
        self,
        image: Image.Image,
        boxes: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Applica augmentations coerenti a immagine e bounding box.
        boxes: tensore (N, 4) in pixel [xmin, ymin, xmax, ymax]
        labels: tensore (N,) con class_ids
        """
        # 1. Color Jitter opzionale (non altera la geometria)
        if self.color_jitter is not None:
            image = self.color_jitter(image)

        # 2. Letterbox resize con preservazione aspect ratio e padding
        image, norm_boxes = letterbox_image(image, boxes, self.target_size)

        # 3. Random Horizontal Flip (se training)
        if self.is_train and random.random() < self.hflip_prob:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            if norm_boxes.numel() > 0:
                # Inverti coordinata orizzontale cx
                norm_boxes[:, 0] = 1.0 - norm_boxes[:, 0]

        # 4. ToTensor e Normalizzazione ImageNet richiesta da VGG16-BN
        img_tensor = self.to_tensor(image)
        img_tensor = self.normalize(img_tensor)

        return img_tensor, norm_boxes, labels


class PascalVOCDetectionWrapper(Dataset):
    """
    Dataset wrapper per Pascal VOC con parsing da torchvision.datasets.VOCDetection
    o da directory locale pre-scaricata.
    """

    def __init__(
        self,
        root: str,
        year: str = "2007",
        image_set: str = "trainval",
        download: bool = False,
        target_size: int = 416,
        is_train: bool = True,
        hflip_prob: float = 0.5,
        color_jitter_params: Optional[Dict[str, float]] = None,
    ):
        self.root = root
        self.year = year
        self.image_set = image_set
        self.target_size = target_size
        self.is_train = is_train

        self.transform = DetectionAugmentation(
            target_size=target_size,
            is_train=is_train,
            hflip_prob=hflip_prob,
            color_jitter_params=color_jitter_params,
        )

        try:
            self.voc_dataset = VOCDetection(
                root=root,
                year=year,
                image_set=image_set,
                download=download,
            )
            self.is_torchvision = True
        except Exception:
            self.is_torchvision = False
            self._init_local_voc(root, year, image_set)

    def _init_local_voc(self, root: str, year: str, image_set: str):
        """Inizializzazione di fallback se i file VOC sono già estratti localmente."""
        base_path = Path(root)
        voc_dir = base_path / f"VOCdevkit/VOC{year}"
        if not voc_dir.exists():
            voc_dir = base_path / f"VOC{year}"

        splits_file = voc_dir / "ImageSets" / "Main" / f"{image_set}.txt"
        if splits_file.exists():
            with open(splits_file, "r") as f:
                self.ids = [line.strip() for line in f if line.strip()]
        else:
            self.ids = [f.stem for f in (voc_dir / "Annotations").glob("*.xml")]

        self.img_dir = voc_dir / "JPEGImages"
        self.ann_dir = voc_dir / "Annotations"

    def __len__(self) -> int:
        if self.is_torchvision:
            return len(self.voc_dataset)
        return len(self.ids)

    def _parse_xml_dict(self, target: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Estrae box e classi dal dizionario di torchvision VOCDetection."""
        boxes = []
        labels = []
        annotation = target.get("annotation", {})
        objects = annotation.get("object", [])
        if isinstance(objects, dict):
            objects = [objects]

        for obj in objects:
            difficult = int(obj.get("difficult", 0))
            if difficult == 1 and self.is_train:
                continue

            name = obj.get("name", "").strip().lower()
            if name in VOC_CLASS_TO_IDX:
                bndbox = obj["bndbox"]
                xmin = float(bndbox["xmin"])
                ymin = float(bndbox["ymin"])
                xmax = float(bndbox["xmax"])
                ymax = float(bndbox["ymax"])

                # Verifica validità coordinate
                if xmax > xmin and ymax > ymin:
                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(VOC_CLASS_TO_IDX[name])

        if boxes:
            return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)
        return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.long)

    def _parse_xml_file(self, xml_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        """Estrae box e classi da file XML locale."""
        tree = ET.parse(str(xml_path))
        root = tree.getroot()
        boxes = []
        labels = []

        for obj in root.findall("object"):
            diff = obj.find("difficult")
            if diff is not None and int(diff.text) == 1 and self.is_train:
                continue

            name = obj.find("name").text.strip().lower()
            if name in VOC_CLASS_TO_IDX:
                bnd = obj.find("bndbox")
                xmin = float(bnd.find("xmin").text)
                ymin = float(bnd.find("ymin").text)
                xmax = float(bnd.find("xmax").text)
                ymax = float(bnd.find("ymax").text)
                if xmax > xmin and ymax > ymin:
                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(VOC_CLASS_TO_IDX[name])

        if boxes:
            return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)
        return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.long)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.is_torchvision:
            image, target = self.voc_dataset[idx]
            boxes, labels = self._parse_xml_dict(target)
        else:
            img_id = self.ids[idx]
            img_path = self.img_dir / f"{img_id}.jpg"
            xml_path = self.ann_dir / f"{img_id}.xml"
            image = Image.open(img_path).convert("RGB")
            boxes, labels = self._parse_xml_file(xml_path)

        img_tensor, norm_boxes, labels = self.transform(image, boxes, labels)
        return img_tensor, norm_boxes, labels


class YOLODetectionDataset(Dataset):
    """Dataset loader per immagini e file di label in formato standard YOLO (class_id cx cy w h)."""

    def __init__(
        self,
        images_dir: str,
        labels_dir: Optional[str] = None,
        target_size: int = 416,
        is_train: bool = True,
        hflip_prob: float = 0.5,
        color_jitter_params: Optional[Dict[str, float]] = None,
    ):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir) if labels_dir else self.images_dir.parent / "labels" / self.images_dir.name
        self.target_size = target_size

        self.transform = DetectionAugmentation(
            target_size=target_size,
            is_train=is_train,
            hflip_prob=hflip_prob,
            color_jitter_params=color_jitter_params,
        )

        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        self.image_files = sorted(
            [f for f in self.images_dir.glob("*") if f.suffix.lower() in valid_exts]
        )

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")
        w_img, h_img = image.size

        label_path = self.labels_dir / f"{img_path.stem}.txt"
        boxes_pixel = []
        labels = []

        if label_path.exists():
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        cx, cy, bw, bh = map(float, parts[1:5])
                        xmin = (cx - bw / 2.0) * w_img
                        ymin = (cy - bh / 2.0) * h_img
                        xmax = (cx + bw / 2.0) * w_img
                        ymax = (cy + bh / 2.0) * h_img
                        boxes_pixel.append([xmin, ymin, xmax, ymax])
                        labels.append(cls_id)

        if boxes_pixel:
            boxes_tensor = torch.tensor(boxes_pixel, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.long)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.long)

        img_tensor, norm_boxes, labels_tensor = self.transform(image, boxes_tensor, labels_tensor)
        return img_tensor, norm_boxes, labels_tensor


def detection_collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate function per formattare i target in un unico tensore compatto (N_total, 6):
    [batch_idx, class_id, cx, cy, w, h] con coordinate normalizzate in [0, 1].
    """
    images = []
    targets_list = []

    for batch_idx, (img, boxes, labels) in enumerate(batch):
        images.append(img)
        num_boxes = boxes.shape[0]
        if num_boxes > 0:
            batch_col = torch.full((num_boxes, 1), batch_idx, dtype=torch.float32)
            cls_col = labels.unsqueeze(1).float()
            # [batch_idx, class_id, cx, cy, w, h]
            target_entry = torch.cat([batch_col, cls_col, boxes], dim=1)
            targets_list.append(target_entry)

    batched_images = torch.stack(images, dim=0)
    if targets_list:
        batched_targets = torch.cat(targets_list, dim=0)
    else:
        batched_targets = torch.zeros((0, 6), dtype=torch.float32)

    return batched_images, batched_targets


def build_dataloaders(config: Dict[str, Any]) -> Tuple[DataLoader, Optional[DataLoader], int, List[str]]:
    """
    Costruisce e restituisce i DataLoader di train e validation in base alla configurazione.
    Ritorna:
        train_loader, val_loader, num_classes, class_names
    """
    cfg_data = config.get("dataset", {})
    cfg_train = config.get("training", {})
    dataset_type = cfg_data.get("type", "voc").lower()
    target_size = config.get("model", {}).get("img_size", 416)
    batch_size = cfg_train.get("batch_size", 16)
    num_workers = cfg_train.get("num_workers", 4)
    hflip_prob = cfg_data.get("horizontal_flip_prob", 0.5)
    color_jitter = cfg_data.get("color_jitter", None)

    if dataset_type == "voc":
        voc_root = cfg_data.get("voc_root", "./data/voc")
        auto_download = cfg_data.get("auto_download", True)
        train_years = cfg_data.get("train_years", ["2007", "2012"])
        val_years = cfg_data.get("val_years", ["2007"])

        train_datasets = []
        for yr in train_years:
            ds = PascalVOCDetectionWrapper(
                root=voc_root,
                year=yr,
                image_set="trainval",
                download=auto_download,
                target_size=target_size,
                is_train=True,
                hflip_prob=hflip_prob,
                color_jitter_params=color_jitter,
            )
            train_datasets.append(ds)

        train_dataset = ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]

        val_datasets = []
        for yr in val_years:
            val_split = "test" if yr == "2007" else "val"
            ds = PascalVOCDetectionWrapper(
                root=voc_root,
                year=yr,
                image_set=val_split,
                download=auto_download,
                target_size=target_size,
                is_train=False,
                hflip_prob=0.0,
                color_jitter_params=None,
            )
            val_datasets.append(ds)

        val_dataset = ConcatDataset(val_datasets) if len(val_datasets) > 1 else val_datasets[0]
        num_classes = 20
        class_names = VOC_CLASSES

    elif dataset_type == "yolo":
        data_dir = Path(cfg_data.get("data_dir", "./data"))
        train_img = data_dir / "images" / "train"
        val_img = data_dir / "images" / "val"
        train_dataset = YOLODetectionDataset(
            str(train_img),
            target_size=target_size,
            is_train=True,
            hflip_prob=hflip_prob,
            color_jitter_params=color_jitter,
        )
        val_dataset = YOLODetectionDataset(
            str(val_img),
            target_size=target_size,
            is_train=False,
            hflip_prob=0.0,
            color_jitter_params=None,
        ) if val_img.exists() else None

        num_classes = config.get("model", {}).get("num_classes", 20)
        class_names = [f"class_{i}" for i in range(num_classes)]

    else:
        raise ValueError(f"Tipo dataset non supportato: {dataset_type}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=detection_collate_fn,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=detection_collate_fn,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    ) if val_dataset is not None else None

    return train_loader, val_loader, num_classes, class_names

