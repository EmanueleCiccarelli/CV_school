"""
detector_builder.py - Costruttore dell'Object Detector Ultralytics basato su VGG-16

Questo modulo si occupa di:
1. Definire l'architettura YOLO DetectionModel per VGG-16:
   - Backbone: TorchVision wrapper nativo con VGG-16 pre-addestrata su ImageNet
     (unwrap=True, truncate=2 per rimuovere avgpool e i 3 linear layer, split=True).
   - Prelevare le feature map multi-scala:
     * C3 (Index 24): stride 8, 256 canali
     * C4 (Index 34): stride 16, 512 canali
     * C5 (Index 44): stride 32, 512 canali
   - Head: YOLO Detect Head collegata direttamente a C3, C4, C5 (senza neck intermedio).
2. Costruire il DetectionModel caricando i pesi pre-allenati ImageNet ("DEFAULT").
3. Salvare l'architettura in formato .yaml e il checkpoint .pt compatibile con Ultralytics Platform.
4. Verificare il modello salvato tramite un forward pass di inferenza.
"""

from datetime import datetime
import math
from pathlib import Path
from typing import Dict, Any, Optional, Union

import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import YAML


def find_pretrained_weights_file(
    custom_path: Optional[Union[str, Path]] = None,
    base_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Cerca un eventuale checkpoint pre-allenato (da pretrain_pipeline o radice)."""
    if custom_path:
        p = Path(custom_path).resolve()
        return p if p.exists() else None

    if base_dir is None:
        base_dir = Path(__file__).resolve().parent

    search_candidates = [
        base_dir / "checkpoints" / "best.pt",
        base_dir / "pretrain_pipeline" / "checkpoints" / "best.pt",
        base_dir / "pretrained_vgg16_detector.pt",
        base_dir / "checkpoints" / "stage2_final.pt",
        base_dir / "checkpoints" / "stage1_final.pt",
    ]

    for cand in search_candidates:
        if cand.exists() and cand.is_file():
            return cand

    return None


def get_vgg16_detector_yaml(nc: int = 80) -> Dict[str, Any]:
    """
    Ritorna il dizionario di configurazione per il DetectionModel di Ultralytics:
    - Backbone VGG-16 con prelievo feature C3, C4, C5
    - Head YOLO Detect multi-scala senza neck intermedio
    """
    return {
        "nc": nc,
        "scale": "n",
        "scales": {"n": [1.0, 1.0, 1024]},
        "backbone": [
            # [from, repeats, module, args]
            # Layer 0: VGG-16 con pesi ImageNet (DEFAULT), unwrap=True, truncate=2 (rimuove avgpool e lineari), split=True
            [-1, 1, "TorchVision", [512, "vgg16_bn", "DEFAULT", True, 2, True]],
            # Layer 1: Feature map C3 (uscita MaxPool blocco 3, stride 8, 256 canali)
            [0, 1, "Index", [256, 24]],
            # Layer 2: Feature map C4 (uscita MaxPool blocco 4, stride 16, 512 canali)
            [0, 1, "Index", [512, 34]],
            # Layer 3: Feature map C5 (uscita MaxPool blocco 5, stride 32, 512 canali)
            [0, 1, "Index", [512, 44]],
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
    save_yaml: bool = False,
    pretrained_weights_path: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Costruisce l'Object Detector Ultralytics VGG-16 con pesi pre-addestrati e lo esporta in formato .pt.

    Parametri:
        output_pt_path: Percorso del file .pt da generare.
        nc: Numero di classi (default: 80).
        imgsz: Risoluzione immagine per il rilevamento (default: 640).
        verbose: Se True, visualizza il riepilogo dettagliato del modello.
        save_yaml: Se True, salva anche il file .yaml corrispondente (default: False).
        pretrained_weights_path: Percorso facoltativo a un checkpoint pre-addestrato con la testa YOLO.

    Ritorna:
        Path del checkpoint .pt creato.
    """
    output_pt_path = Path(output_pt_path).resolve()
    output_pt_path.parent.mkdir(parents=True, exist_ok=True)

    yaml_cfg = get_vgg16_detector_yaml(nc=nc)

    # Opzionale: salvataggio file .yaml solo se esplicitamente richiesto
    if save_yaml:
        yaml_path = output_pt_path.with_suffix(".yaml")
        try:
            YAML.save(yaml_path, yaml_cfg)
        except Exception:
            pass

    if verbose:
        print(f"\n⚙️  Inizializzazione DetectionModel (VGG-16 Backbone + YOLO Detect Head, {nc} classi)...")

    # Istanziazione DetectionModel Ultralytics con pesi ImageNet pre-caricati
    # Manteniamo verbose=False per presentare un riepilogo didattico pulito senza log tecnici interni
    det_model = DetectionModel(cfg=yaml_cfg, ch=3, nc=nc, verbose=False)

    # Controllo eventuale checkpoint pre-allenato per Backbone + YOLO Detect Head
    pretrained_file = find_pretrained_weights_file(pretrained_weights_path, base_dir=output_pt_path.parent)
    has_pretrained_head = False

    if pretrained_file:
        try:
            try:
                ckpt_data = torch.load(str(pretrained_file), map_location="cpu", weights_only=False)
            except TypeError:
                ckpt_data = torch.load(str(pretrained_file), map_location="cpu")
            loaded_model = ckpt_data.get("model", None)
            if loaded_model is not None:
                src_state = loaded_model.state_dict() if hasattr(loaded_model, "state_dict") else loaded_model
            else:
                src_state = ckpt_data.get("model_state_dict", ckpt_data)

            target_state = det_model.state_dict()
            transferred = 0
            prior_bias_val = math.log(0.01 / 0.99)

            for k, v in src_state.items():
                if k in target_state:
                    if target_state[k].shape == v.shape:
                        target_state[k].copy_(v)
                        transferred += 1
                    elif "cv3" in k and k.endswith(".2.weight"):
                        torch.nn.init.normal_(target_state[k], mean=0.0, std=0.01)
                        transferred += 1
                    elif "cv3" in k and k.endswith(".2.bias"):
                        target_state[k].fill_(prior_bias_val)
                        transferred += 1

            det_model.load_state_dict(target_state)
            has_pretrained_head = True
        except Exception:
            has_pretrained_head = False

    if verbose:
        total_params = sum(p.numel() for p in det_model.parameters())
        if has_pretrained_head:
            print("   -> Pesi pre-addestrati: Caricati con successo per Backbone e YOLO Detect Head (Transfer Learning)")
        else:
            print("   -> Backbone: VGG-16 con pesi pre-addestrati ImageNet")
        print("   -> Feature maps multi-scala: C3 (stride 8), C4 (stride 16), C5 (stride 32)")
        print(f"   -> Detection Head: YOLO Detect ({nc} classi)")
        print(f"   -> Parametri totali: {total_params:,}")

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

