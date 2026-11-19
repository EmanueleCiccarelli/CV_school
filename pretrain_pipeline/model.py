"""
model.py - Architettura Object Detector VGG16-BN + Multi-Scale Detect Head (Ultralytics Platform Compatible)

Definisce e gestisce il modello compatibile al 100% con Ultralytics Platform e CLI:
- Backbone: VGG16-BN pre-addestrata su ImageNet (solo i 44 layer di features, stride 32, 512 canali).
- Connessioni multi-scala senza colli di bottiglia:
  * C3 (Index 24): uscita MaxPool blocco 3, stride 8, 256 canali (oggetti piccoli)
  * C4 (Index 34): uscita MaxPool blocco 4, stride 16, 512 canali (oggetti medi)
  * C5 (Index 44): uscita MaxPool blocco 5, stride 32, 512 canali (oggetti grandi)
- Testa Detect di Ultralytics:
  * Ramo coordinate cv2: 2 blocchi convoluzionali 3x3 Conv-BN separati per ciascuna scala + conv 1x1.
  * Ramo classi cv3: 2 blocchi convoluzionali 3x3 separati per ciascuna scala + conv 1x1 finale.
  * Modulo DFL (Distribution Focal Loss) per la massima precisione dei bounding box.
"""

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from ultralytics.nn.tasks import DetectionModel


def get_vgg16_detector_yaml(nc: int = 80) -> Dict[str, Any]:
    """
    Ritorna il dizionario architetturale per il DetectionModel di Ultralytics:
    - Backbone VGG16-BN (44 layer)
    - Taps multi-scala C3 (layer 24), C4 (layer 34), C5 (layer 44)
    - Testa Detect multi-scala collegata direttamente (default: 80 classi MS COCO)
    """
    return {
        "nc": nc,
        "scale": "n",
        "scales": {"n": [1.0, 1.0, 1024]},
        "backbone": [
            # [from, repeats, module, args]
            [-1, 1, "TorchVision", [512, "vgg16_bn", "DEFAULT", True, 2, True]],
            [0, 1, "Index", [256, 24]],  # C3 (stride 8, 256 ch)
            [0, 1, "Index", [512, 34]],  # C4 (stride 16, 512 ch)
            [0, 1, "Index", [512, 44]],  # C5 (stride 32, 512 ch)
        ],
        "head": [
            # Utilizziamo la stringa dinamica "nc" invece del numero fisso [nc] per auto-adattamento
            [[1, 2, 3], 1, "Detect", ["nc"]],
        ],
    }


def build_detector_model(nc: int = 80, verbose: bool = False) -> DetectionModel:
    """Costruisce un'istanza di DetectionModel Ultralytics con backbone VGG16-BN pre-addestrata ImageNet (default: 80 classi)."""
    yaml_cfg = get_vgg16_detector_yaml(nc=nc)
    model = DetectionModel(cfg=yaml_cfg, ch=3, nc=nc, verbose=verbose)
    return model


def freeze_batchnorm_layers(module: nn.Module, feature_slice: Optional[slice] = None) -> None:
    """
    Forza tutti i layer BatchNorm2d (e BatchNorm1d) a rimanere in modalità .eval()
    per preservare le statistiche running_mean e running_var di ImageNet.
    """
    if feature_slice is None:
        target_modules = module.modules()
    else:
        if hasattr(module, "__getitem__"):
            layers = module[feature_slice]
            target_modules = []
            for lyr in layers:
                target_modules.extend(lyr.modules())
        else:
            target_modules = module.modules()

    for m in target_modules:
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.eval()


def configure_detector_stage(model: DetectionModel, stage: int) -> None:
    """
    Configura il modello DetectionModel per gli stadi di addestramento:
    - Stage 1: Backbone [0:44] completamente congelato. BatchNorm in .eval(). Testa YOLO allenabile.
    - Stage 2: Conv1..Conv4 [0:34] congelati con BatchNorm in .eval(). Conv5 [34:44] e testa YOLO allenabili.
    """
    backbone_seq = model.model[0].m  # Sequenza di 44 layer vgg16_bn.features
    detect_head = model.model[4]     # Modulo Detect di Ultralytics

    if stage == 1:
        # Congela l'intero backbone (tutti i 44 layer)
        for p in backbone_seq.parameters():
            p.requires_grad = False
        # Scongela tutta la testa
        for p in detect_head.parameters():
            p.requires_grad = True

        freeze_batchnorm_layers(backbone_seq, slice(0, 44))

    elif stage == 2:
        # Congela conv1..conv4 [0:34]
        for p in backbone_seq[0:34].parameters():
            p.requires_grad = False
        # Scongela conv5 [34:44]
        for p in backbone_seq[34:44].parameters():
            p.requires_grad = True
        # Scongela la testa
        for p in detect_head.parameters():
            p.requires_grad = True

        freeze_batchnorm_layers(backbone_seq, slice(0, 34))

    else:
        raise ValueError(f"Stadio sconosciuto: {stage}")
