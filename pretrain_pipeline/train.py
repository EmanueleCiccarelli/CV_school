"""
train.py - Motore di Pre-Training a Due Stadi per VGG16-BN + Multi-Scale Detect Head
(100% Compatibile con Ultralytics Platform e CLI yolo train)

Pianificazione a due stadi:
1. Stage 1 - Head Warm-up (default: 5-10 epoche):
   - Backbone vgg16_bn.features [0:44] completamente congelato.
   - Protezione BatchNorm: tutti i 13 layer BatchNorm2d del backbone forzati in .eval().
   - Addestramento esclusivo della testa YOLO Detect con AdamW (lr=1e-3, wd=1e-4).
2. Stage 2 - Partial Backbone Fine-Tuning (default: 20-30 epoche):
   - Blocchi conv1-conv4 [0:34] congelati con BatchNorm in .eval().
   - Scongelamento dell'ultimo blocco conv5 [34:44] (conv5_1, conv5_2, conv5_3) e della testa Detect.
   - Learning rate ridotto e decadimento Cosine Annealing fino a 1e-6.
3. Compatibilità totale:
   - Il checkpoint generato (best.pt / last.pt) è nativo di Ultralytics DetectionModel:
     pronto per essere caricato sulla piattaforma Ultralytics o via CLI.
"""

import argparse
from datetime import datetime
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import yaml

# Assicura che la root del progetto sia sempre in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from pretrain_pipeline.model import get_vgg16_detector_yaml, freeze_batchnorm_layers


def load_yaml_config(config_path: Union[str, Path] = "pretrain_pipeline/config.yaml") -> Dict[str, Any]:
    """Carica il file di configurazione YAML."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configurazione non trovata: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_initial_checkpoint(
    pt_path: Path,
    nc: int = 80,
    imgsz: int = 640,
) -> Path:
    """Inizializza il DetectionModel con pesi ImageNet e salva il checkpoint .pt di base."""
    pt_path = Path(pt_path).resolve()
    pt_path.parent.mkdir(parents=True, exist_ok=True)

    yaml_cfg = get_vgg16_detector_yaml(nc=nc)
    det_model = DetectionModel(cfg=yaml_cfg, ch=3, nc=nc, verbose=False)

    checkpoint = {
        "epoch": -1,
        "best_fitness": None,
        "model": det_model.eval(),
        "ema": None,
        "updates": 0,
        "optimizer": None,
        "train_args": {
            "task": "detect",
            "model": pt_path.name,
            "imgsz": imgsz,
        },
        "train_metrics": {},
        "date": datetime.now().isoformat(),
        "version": "8.3.0",
    }
    torch.save(checkpoint, str(pt_path))
    return pt_path


def run_stage1_warmup(
    initial_pt: Path,
    data_yaml: str,
    epochs: int = 5,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 16,
    imgsz: int = 640,
    device: str = "cpu",
    workers: int = 4,
    save_dir: Path = Path("./checkpoints"),
) -> Path:
    """
    Esegue lo Stage 1: Head Warm-up
    - freeze=1: blocca l'intero backbone [0:44]
    - Callback: mantiene tutti i BatchNorm2d del backbone in .eval()
    """
    print("\n" + "=" * 70)
    print("  INIZIO STAGE 1: HEAD WARM-UP (ULTRALYTICS DETECT HEAD)")
    print(f"  • Epoche: {epochs} | LR0: {lr} | Weight Decay: {weight_decay}")
    print("  • Backbone vgg16_bn.features [0:44]: 100% CONGELATO (freeze=1)")
    print("  • Protezione BatchNorm: TUTTI i BatchNorm2d del backbone forzati in .eval()")
    print("=" * 70)

    yolo_model = YOLO(str(initial_pt))

    # Callback di sicurezza per forzare BatchNorm in .eval() ad ogni epoca
    def on_train_epoch_start(trainer):
        backbone_seq = trainer.model.model[0].m
        freeze_batchnorm_layers(backbone_seq, slice(0, 44))

    yolo_model.add_callback("on_train_epoch_start", on_train_epoch_start)

    runs_dir = save_dir / "runs_stage1"
    results = yolo_model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        freeze=1,  # Congela l'intero blocco 0 (vgg16_bn.features)
        lr0=lr,
        weight_decay=weight_decay,
        device=device,
        workers=workers,
        project=str(runs_dir),
        name="stage1",
        exist_ok=True,
        verbose=True,
    )

    stage1_weights = Path(results.save_dir) / "weights" / "best.pt"
    if not stage1_weights.exists():
        stage1_weights = Path(results.save_dir) / "weights" / "last.pt"

    dest_best = save_dir / "stage1_final.pt"
    shutil.copy2(stage1_weights, dest_best)
    print(f"  🏁 Stage 1 completato! Checkpoint salvato in: {dest_best}")
    return dest_best


def run_stage2_finetuning(
    stage1_pt: Path,
    data_yaml: str,
    epochs: int = 25,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    lrf: float = 0.01,
    batch_size: int = 16,
    imgsz: int = 640,
    device: str = "cpu",
    workers: int = 4,
    save_dir: Path = Path("./checkpoints"),
) -> Path:
    """
    Esegue lo Stage 2: Partial Backbone Fine-Tuning
    - conv1..conv4 [0:34] congelati con BatchNorm in .eval()
    - conv5 [34:44] e Detect head scongelati
    - Cosine Annealing fino a lrf * lr
    """
    print("\n" + "=" * 70)
    print("  INIZIO STAGE 2: PARTIAL BACKBONE FINE-TUNING")
    print(f"  • Epoche: {epochs} | LR0: {lr} | Weight Decay: {weight_decay} | LRF (Cosine): {lrf}")
    print("  • Blocchi conv1-conv4 [0:34]: 100% CONGELATI con BatchNorm in .eval()")
    print("  • Ultimo blocco conv5 [34:44] + Testa Detect: SCONGELATI per fine-tuning")
    print("=" * 70)

    yolo_model = YOLO(str(stage1_pt))

    def setup_stage2_freeze(trainer):
        backbone_seq = trainer.model.model[0].m
        detect_head = trainer.model.model[4]
        # Congela conv1-conv4 [0:34]
        for p in backbone_seq[0:34].parameters():
            p.requires_grad = False
        # Scongela conv5 [34:44]
        for p in backbone_seq[34:44].parameters():
            p.requires_grad = True
        # Scongela testa
        for p in detect_head.parameters():
            p.requires_grad = True

        freeze_batchnorm_layers(backbone_seq, slice(0, 34))

    def on_train_epoch_start(trainer):
        backbone_seq = trainer.model.model[0].m
        freeze_batchnorm_layers(backbone_seq, slice(0, 34))

    yolo_model.add_callback("on_pretrain_routine_end", setup_stage2_freeze)
    yolo_model.add_callback("on_train_epoch_start", on_train_epoch_start)

    runs_dir = save_dir / "runs_stage2"
    results = yolo_model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        lr0=lr,
        lrf=lrf,
        cos_lr=True,
        weight_decay=weight_decay,
        device=device,
        workers=workers,
        project=str(runs_dir),
        name="stage2",
        exist_ok=True,
        verbose=True,
    )

    stage2_best = Path(results.save_dir) / "weights" / "best.pt"
    stage2_last = Path(results.save_dir) / "weights" / "last.pt"

    dest_best = save_dir / "best.pt"
    dest_last = save_dir / "last.pt"
    if stage2_best.exists():
        shutil.copy2(stage2_best, dest_best)
    if stage2_last.exists():
        shutil.copy2(stage2_last, dest_last)

    print(f"\n  🎉 Addestramento a due stadi completato con successo!")
    print(f"  • Best Checkpoint: {dest_best}")
    print(f"  • Last Checkpoint: {dest_last}")
    return dest_best


def main():
    parser = argparse.ArgumentParser(
        description="Pre-Training Pipeline per VGG16-BN + Multi-Scale Detect Head (Ultralytics Compatible)"
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="pretrain_pipeline/config.yaml",
        help="Percorso al file di configurazione YAML.",
    )
    parser.add_argument(
        "--data",
        "-d",
        type=str,
        default=None,
        help="Dataset YAML di addestramento (es. VOC.yaml, coco8.yaml).",
    )
    parser.add_argument(
        "--stage",
        type=int,
        default=None,
        choices=[1, 2],
        help="Forza l'esecuzione di un solo stage (1 o 2).",
    )
    parser.add_argument(
        "--epochs-stage1",
        type=int,
        default=None,
        help="Numero di epoche per lo Stage 1.",
    )
    parser.add_argument(
        "--epochs-stage2",
        type=int,
        default=None,
        help="Numero di epoche per lo Stage 2.",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=None,
        help="Dimensione batch.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Dispositivo ('cuda' o 'cpu').",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory dove salvare i checkpoint.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verifica rapida (1 epoca Stage 1 + 1 epoca Stage 2 su dataset di test coco8).",
    )
    args = parser.parse_args()

    # 1. Carica configurazione
    config = load_yaml_config(args.config)
    cfg_model = config.get("model", {})
    cfg_train = config.get("training", {})

    nc = cfg_model.get("nc", 80)
    imgsz = cfg_model.get("imgsz", 640)

    # 2. Risoluzione dispositivo
    device = args.device or cfg_train.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA non disponibile: fallback automatico su CPU.")
        device = "cpu"

    save_dir = Path(args.checkpoint_dir or config.get("checkpoints", {}).get("save_dir", "./checkpoints"))
    save_dir.mkdir(parents=True, exist_ok=True)

    # 3. Risoluzione dataset e sincronizzazione classi (nc)
    from ultralytics.data.utils import check_det_dataset

    if args.dry_run:
        print("🔬 Modalità DRY-RUN attivata: utilizzo dataset di test 'coco8.yaml'.")
        data_yaml = "coco8.yaml"
        epochs_s1 = 1
        epochs_s2 = 1
        batch_size = 2
    else:
        data_yaml = args.data or cfg_train.get("data", "coco.yaml")
        epochs_s1 = args.epochs_stage1 or cfg_train.get("stage1", {}).get("epochs", 5)
        epochs_s2 = args.epochs_stage2 or cfg_train.get("stage2", {}).get("epochs", 25)
        batch_size = args.batch_size or cfg_train.get("batch_size", 16)

    # Risoluzione automatica di nc dal dataset per evitare mismatch
    try:
        data_info = check_det_dataset(data_yaml)
        nc = data_info.get("nc", nc)
        print(f"📊 Dataset: {data_yaml} | Classi rilevate (nc): {nc}")
    except Exception:
        print(f"📊 Dataset: {data_yaml} | Utilizzo nc di default: {nc}")

    # 4. Inizializzazione checkpoint base
    initial_pt = save_dir / f"initial_vgg16_detector_{nc}cls.pt"
    if not initial_pt.exists():
        print(f"📦 Creazione checkpoint iniziale VGG16-BN ({nc} classi)...")
        create_initial_checkpoint(initial_pt, nc=nc, imgsz=imgsz)

    stages_to_run = [args.stage] if args.stage is not None else [1, 2]
    current_pt = initial_pt

    # 5. Esecuzione Stage 1
    if 1 in stages_to_run:
        s1_lr = cfg_train.get("stage1", {}).get("lr0", 1e-3)
        s1_wd = cfg_train.get("stage1", {}).get("weight_decay", 1e-4)
        current_pt = run_stage1_warmup(
            initial_pt=current_pt,
            data_yaml=data_yaml,
            epochs=epochs_s1,
            lr=s1_lr,
            weight_decay=s1_wd,
            batch_size=batch_size,
            imgsz=imgsz,
            device=device,
            workers=cfg_train.get("workers", 4),
            save_dir=save_dir,
        )

    # 6. Esecuzione Stage 2
    if 2 in stages_to_run:
        s2_lr = cfg_train.get("stage2", {}).get("lr0_head", 1e-4)
        s2_wd = cfg_train.get("stage2", {}).get("weight_decay", 1e-4)
        s2_lrf = cfg_train.get("stage2", {}).get("lrf", 0.01)
        current_pt = run_stage2_finetuning(
            stage1_pt=current_pt,
            data_yaml=data_yaml,
            epochs=epochs_s2,
            lr=s2_lr,
            weight_decay=s2_wd,
            lrf=s2_lrf,
            batch_size=batch_size,
            imgsz=imgsz,
            device=device,
            workers=cfg_train.get("workers", 4),
            save_dir=save_dir,
        )


if __name__ == "__main__":
    main()
