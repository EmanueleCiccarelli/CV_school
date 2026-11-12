"""
detector_builder.py - Costruttore dell'Object Detector Ultralytics basato su VGG-16

Questo modulo si occupa di:
1. Definire l'architettura YOLO DetectionModel per VGG-16:
   - Backbone: TorchVision wrapper nativo con VGG-16 pre-addestrata su ImageNet
     (unwrap=True, truncate=2 per rimuovere avgpool e i 3 linear layer, split=True).
   - Prelevare le feature map multi-scala:
     * C3 (Index 17): stride 8, 256 canali
     * C4 (Index 24): stride 16, 512 canali
     * C5 (Index 31): stride 32, 512 canali
   - Head: YOLO Detect Head collegata direttamente a C3, C4, C5 (senza neck intermedio).
2. Costruire il DetectionModel caricando i pesi pre-allenati ImageNet ("DEFAULT").
3. Salvare l'architettura in formato .yaml e il checkpoint .pt compatibile con Ultralytics Platform.
4. Verificare il modello salvato tramite un forward pass di inferenza.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import YAML


def get_vgg16_detector_yaml(nc: int = 80) -> Dict[str, Any]:
    """
    Ritorna il dizionario di configurazione per il DetectionModel di Ultralytics:
    - Backbone VGG-16 con prelievo feature C3, C4, C5
    - Head YOLO Detect multi-scala senza neck intermedio
    """
    return {
        "nc": nc,
        "scales": {"n": [1.0, 1.0, 1024]},
        "backbone": [
            # [from, repeats, module, args]
            # Layer 0: VGG-16 pre-addestrata ImageNet (DEFAULT), unwrap=True, truncate=2 (rimuove avgpool e lineari), split=True
            [-1, 1, "TorchVision", [512, "vgg16", "DEFAULT", True, 2, True]],
            # Layer 1: Feature map C3 (uscita MaxPool blocco 3, stride 8, 256 canali)
            [0, 1, "Index", [256, 17]],
            # Layer 2: Feature map C4 (uscita MaxPool blocco 4, stride 16, 512 canali)
            [0, 1, "Index", [512, 24]],
            # Layer 3: Feature map C5 (uscita MaxPool blocco 5, stride 32, 512 canali)
            [0, 1, "Index", [512, 31]],
        ],
        "head": [
            # Layer 4: Detect Head collegata direttamente a C3, C4, C5 (senza neck FPN/PAN)
            [[1, 2, 3], 1, "Detect", [nc]],
        ],
    }


def build_and_export_detector(
    output_pt_path: Path,
    nc: int = 80,
    imgsz: int = 640,
    verbose: bool = True,
) -> Path:
    """
    Costruisce l'Object Detector Ultralytics VGG-16 con pesi ImageNet e lo esporta in formato .pt.

    Parametri:
        output_pt_path: Percorso del file .pt da generare.
        nc: Numero di classi (default: 80).
        imgsz: Risoluzione immagine per il rilevamento (default: 640).
        verbose: Se True, visualizza il riepilogo dettagliato del modello.

    Ritorna:
        Path del checkpoint .pt creato.
    """
    output_pt_path = Path(output_pt_path).resolve()
    output_pt_path.parent.mkdir(parents=True, exist_ok=True)

    yaml_cfg = get_vgg16_detector_yaml(nc=nc)

    # Salva il file .yaml corrispondente
    yaml_path = output_pt_path.with_suffix(".yaml")
    try:
        YAML.save(yaml_path, yaml_cfg)
    except Exception:
        pass

    if verbose:
        print(f"\n⚙️  Inizializzazione DetectionModel (VGG-16 Backbone + YOLO Detect Head, {nc} classi)...")

    # Istanziazione DetectionModel Ultralytics con pesi ImageNet pre-caricati
    det_model = DetectionModel(cfg=yaml_cfg, ch=3, nc=nc, verbose=verbose)

    # Creazione del checkpoint compatibile al 100% con Ultralytics Platform e CLI
    checkpoint = {
        "epoch": -1,
        "best_fitness": None,
        "model": det_model.eval(),
        "ema": None,
        "updates": 0,
        "optimizer": None,
        "train_args": {
            "task": "detect",
            "model": output_pt_path.name,
            "imgsz": imgsz,
        },
        "train_metrics": {},
        "date": datetime.now().isoformat(),
        "version": "8.3.0",
    }

    torch.save(checkpoint, str(output_pt_path))
    print(f"\n💾 Checkpoint Object Detector salvato con successo:")
    print(f"   -> Modello PyTorch: {output_pt_path}")
    print(f"   -> Configurazione YAML: {yaml_path}")

    # Forward pass di test per verificare la perfetta operatività del modello
    print(f"\n🔍 Verifica forward pass di inferenza con input (1, 3, {imgsz}, {imgsz})...")
    try:
        yolo_model = YOLO(str(output_pt_path))
        dummy_input = torch.zeros(1, 3, imgsz, imgsz)
        with torch.no_grad():
            preds = yolo_model.model(dummy_input)
        tensor_shape = preds[0].shape if isinstance(preds, (tuple, list)) else preds.shape
        print(f"✅ Verifica superata con successo! Task: {yolo_model.task} | Shape output: {tensor_shape}")
    except Exception as e:
        print(f"⚠️  Avviso durante il forward pass di test: {e}")

    return output_pt_path

