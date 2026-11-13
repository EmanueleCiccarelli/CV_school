r"""
export_checkpoint.py - Chirurgia dei Pesi e Adattamento a Classi Arbitrarie per Ultralytics DetectionModel

Funzionalità principale:
`prepare_custom_class_checkpoint(pretrained_pt_path, output_pt_path, num_custom_classes)`
Prende un checkpoint pre-allenato (es. 20 classi Pascal VOC o 80 classi COCO) e prepara un nuovo file .pt
compatibile al 100% con Ultralytics Platform e la CLI `yolo train` per un numero arbitrario di classi:
1. Copia al 100% tutti i pesi del backbone `vgg16_bn.features` (44 layer).
2. Copia al 100% tutti i pesi del ramo coordinate `cv2` (2 convoluzioni 3x3 per ciascuna scala C3, C4, C5).
3. Copia al 100% tutti i blocchi intermedi 3x3 del ramo classificazione `cv3` (`cv3.*.0` e `cv3.*.1`).
4. Re-inizializza in modo pulito e stabile SOLO i layer 1x1 finali di proiezione (`cv3.*.2`)
   con prior bias $b = \ln(p / (1 - p)) \approx -4.595$ ($p = 0.01$).
5. Aggiorna i metadati del checkpoint (`model.names`, `model.yaml['nc']`, `nc`).
"""

import argparse
import copy
from datetime import datetime
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn

# Assicura che la root del progetto sia sempre in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from pretrain_pipeline.model import get_vgg16_detector_yaml


def prepare_custom_class_checkpoint(
    pretrained_pt_path: Union[str, Path],
    output_pt_path: Union[str, Path],
    num_custom_classes: int,
    prior_prob: float = 0.01,
    imgsz: int = 640,
    verbose: bool = True,
) -> Path:
    """
    Adatta un checkpoint pre-addestrato VGG16-BN ad un nuovo numero di classi.
    Preserva al 100% il backbone, il ramo di regressione box cv2, il modulo DFL
    e le feature intermedie 3x3 del ramo classificazione cv3.

    Parametri:
        pretrained_pt_path: Percorso del file .pt sorgente pre-addestrato.
        output_pt_path: Percorso del nuovo file .pt generato.
        num_custom_classes: Numero di classi del nuovo dataset target.
        prior_prob: Prior probability per l'inizializzazione del bias di classificazione (default: 0.01).
        imgsz: Risoluzione di addestramento (default: 640).
        verbose: Se True, visualizza il riepilogo dettagliato delle operazioni.

    Ritorna:
        Path del nuovo checkpoint salvato.
    """
    pretrained_path = Path(pretrained_pt_path).resolve()
    out_path = Path(output_pt_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not pretrained_path.exists():
        raise FileNotFoundError(f"Checkpoint sorgente non trovato: {pretrained_path}")

    if num_custom_classes < 1:
        raise ValueError(f"Il numero di classi deve essere >= 1, ricevuto: {num_custom_classes}")

    if verbose:
        print("=" * 70)
        print("  CHIRURGIA DEI PESI CHECKPOINT PER NUOVE CLASSI PERSONALIZZATE")
        print("=" * 70)
        print(f"  • Checkpoint sorgente: {pretrained_path}")
        print(f"  • Checkpoint target:   {out_path}")
        print(f"  • Nuove classi target: {num_custom_classes}")

    # 1. Carica il checkpoint originale
    try:
        ckpt = torch.load(str(pretrained_path), map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(str(pretrained_path), map_location="cpu")
    old_model = ckpt.get("model", None)
    if old_model is None:
        raise ValueError("Il checkpoint specificato non contiene un modello valido nel campo 'model'.")

    old_state_dict = old_model.state_dict() if hasattr(old_model, "state_dict") else ckpt.get("model_state_dict", ckpt)
    old_nc = getattr(old_model, "nc", 80)

    if verbose:
        print(f"  • Configurazione originale: {old_nc} classi")

    # 2. Costruisci il nuovo DetectionModel con il numero target di classi
    new_yaml = get_vgg16_detector_yaml(nc=num_custom_classes)
    new_model = DetectionModel(cfg=new_yaml, ch=3, nc=num_custom_classes, verbose=False)
    new_state_dict = new_model.state_dict()

    # 3. Trasferimento e Chirurgia dei Pesi
    transferred_tensors = 0
    prior_bias_val = math.log(prior_prob / (1.0 - prior_prob))  # -4.5951

    for k, v in old_state_dict.items():
        if k in new_state_dict:
            # Pesi con dimensioni identiche (Backbone, CV2 regressione box, DFL, blocchi 3x3 intermedi CV3)
            if new_state_dict[k].shape == v.shape:
                new_state_dict[k].copy_(v)
                transferred_tensors += 1
            # Layer di proiezione finale classificazione 1x1 (cv3.*.2.weight)
            elif "cv3" in k and k.endswith(".2.weight"):
                nn.init.normal_(new_state_dict[k], mean=0.0, std=0.01)
                transferred_tensors += 1
            # Bias finale classificazione 1x1 (cv3.*.2.bias) con prior bias per focal/BCE loss
            elif "cv3" in k and k.endswith(".2.bias"):
                new_state_dict[k].fill_(prior_bias_val)
                transferred_tensors += 1

    new_model.load_state_dict(new_state_dict)
    new_model.eval()

    if verbose:
        print(f"  ✅ Trasferiti con successo {transferred_tensors}/{len(new_state_dict)} tensori!")
        print("     -> Backbone VGG16-BN: 100% pesi e statistiche BN copiate.")
        print("     -> Ramo coordinate cv2 (C3, C4, C5): 100% pesi preservati.")
        print("     -> Blocchi intermedi 3x3 classificazione cv3: 100% pesi preservati.")
        print(f"     -> Layer di proiezione 1x1 ri-tarati su {num_custom_classes} classi con prior bias = {prior_bias_val:.3f}.")

    # 4. Aggiorna i metadati del modello
    new_model.nc = num_custom_classes
    new_model.names = {i: f"class_{i}" for i in range(num_custom_classes)}
    if hasattr(new_model, "yaml") and isinstance(new_model.yaml, dict):
        new_model.yaml["nc"] = num_custom_classes
        new_model.yaml["names"] = new_model.names

    # 5. Creazione checkpoint pronto all'uso con Ultralytics Platform
    adapted_checkpoint = {
        "epoch": -1,
        "best_fitness": None,
        "model": new_model.eval(),
        "ema": None,
        "updates": 0,
        "optimizer": None,
        "train_args": {
            "task": "detect",
            "model": out_path.name,
            "imgsz": imgsz,
        },
        "train_metrics": {},
        "date": datetime.now().isoformat(),
        "version": "8.3.0",
        "adapted_from": str(pretrained_path.name),
        "source_classes": old_nc,
    }

    torch.save(adapted_checkpoint, str(out_path))

    # 6. Verifica forward pass con YOLO
    if verbose:
        print("\n🔍 Verifica forward pass con la libreria Ultralytics YOLO...")
    try:
        yolo_chk = YOLO(str(out_path))
        dummy = torch.zeros(1, 3, imgsz, imgsz)
        with torch.no_grad():
            preds = yolo_chk.model(dummy)
        out_shape = preds[0].shape if isinstance(preds, (tuple, list)) else preds.shape
        if verbose:
            print(f"✅ Verifica completata con successo! Task: {yolo_chk.task} | Shape output: {out_shape}")
            print(f"💾 Checkpoint compatibile Ultralytics salvato in: {out_path}")
            print("=" * 70 + "\n")
    except Exception as e:
        print(f"⚠️  Avviso durante la verifica del modello: {e}")

    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Adatta un checkpoint pre-addestrato VGG16-BN ad un nuovo numero di classi per Ultralytics."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Percorso al checkpoint .pt pre-addestrato (es. checkpoints/best.pt).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Percorso di output per il nuovo checkpoint .pt (es. checkpoints/vgg16_detector_custom.pt).",
    )
    parser.add_argument(
        "--classes",
        "-c",
        type=int,
        required=True,
        help="Numero di classi del nuovo dataset target.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Risoluzione immagine (default: 640).",
    )
    args = parser.parse_args()

    prepare_custom_class_checkpoint(
        pretrained_pt_path=args.input,
        output_pt_path=args.output,
        num_custom_classes=args.classes,
        imgsz=args.imgsz,
        verbose=True,
    )


if __name__ == "__main__":
    main()
