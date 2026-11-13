"""
weights_vault.py - Gestore Sicuro e Offuscato dei Pesi Pre-Addestrati (Anti-Cheat Vault)

Questo modulo protegge i pesi del modello pre-allenato (Backbone VGG16 + YOLO Detect Head)
evitando che gli studenti possano aggirare l'esperienza didattica di BlockDL
prendendo o trascinando (drag-and-drop) direttamente file .pt preesistenti.

Caratteristiche:
1. Container binario offuscato (.vault/model_weights.bin):
   - Header di firma proprietario (MAGIC header).
   - Compressione ad alta velocità zlib.
   - Scrambling bitwise XOR vettoriale (non leggibile da PyTorch né da YOLO CLI né da Ultralytics Platform).
   - Se rinominato in .pt o trascinato nella piattaforma, PyTorch genera errore di unpickling / file non valido.
2. Posizione protetta e nascosta:
   - Resiede all'interno della sottocartella nascosta 'checkpoints/.vault/'
3. Decodifica runtime sicura:
   - Solo quando 'vgg_validator.py' ha certificato la corretta architettura VGG-16 creata
     dallo studente su BlockDL, 'detector_builder.py' decodifica in memoria i pesi
     e genera il modello finale 'vgg16_detector.pt' per lo studente.
"""

import argparse
import io
import os
from pathlib import Path
import time
from typing import Any, Dict, Optional, Union
import zlib

import numpy as np
import torch


# Parametri di cifratura/offuscamento del Vault
VAULT_MAGIC = b"VGG16_SECURE_VAULT_V1\x00\xff"
VAULT_KEY = np.frombuffer(
    b"\x9a\x3f\x7b\x12\xc4\x8e\x51\xd9\x03\xfa\x66\x42\xbe\x11\x59\x7c"
    b"\xd1\x85\x32\xa7\x4f\x90\xbc\x63\x17\xe8\x2d\x5b\x08\x9c\x74\xe2",
    dtype=np.uint8,
)


def _scramble_buffer(data_bytes: bytes) -> bytes:
    """Applica uno scrambling bitwise XOR veloce tramite NumPy."""
    arr = np.frombuffer(data_bytes, dtype=np.uint8).copy()
    key_len = len(VAULT_KEY)
    full_key = np.resize(VAULT_KEY, len(arr))
    scrambled = np.bitwise_xor(arr, full_key)
    return scrambled.tobytes()


def pack_checkpoint_to_vault(
    src_pt_path: Union[str, Path],
    vault_path: Union[str, Path],
    verbose: bool = True,
) -> Path:
    """
    Impacchetta un checkpoint .pt standard in un container binario offuscato .bin.
    Rende il file completamente inutilizzabile con un semplice drag-and-drop o rinomina.
    """
    src_p = Path(src_pt_path).resolve()
    vault_p = Path(vault_path).resolve()
    vault_p.parent.mkdir(parents=True, exist_ok=True)

    if not src_p.exists():
        raise FileNotFoundError(f"Checkpoint sorgente non trovato: {src_p}")

    if verbose:
        print(f"🔒 Impacchettamento vault di sicurezza: {src_p.name} -> {vault_p.name}...")

    # 1. Lettura byte grezzi del checkpoint
    with open(src_p, "rb") as f:
        raw_bytes = f.read()

    # 2. Compressione zlib rapida (livello 1)
    compressed = zlib.compress(raw_bytes, level=1)

    # 3. Scrambling XOR bitwise
    scrambled_payload = _scramble_buffer(compressed)

    # 4. Scrittura con Magic Header
    final_payload = VAULT_MAGIC + scrambled_payload
    with open(vault_p, "wb") as f:
        f.write(final_payload)

    if verbose:
        orig_mb = len(raw_bytes) / (1024 * 1024)
        vault_mb = len(final_payload) / (1024 * 1024)
        print(f"   ✅ Vault generato con successo: {vault_mb:.1f} MB (da {orig_mb:.1f} MB)")
        print(f"   🛡️  Il file è ora offuscato e protetto contro drag-and-drop o accessi diretti.")

    return vault_p


def load_checkpoint_from_vault(vault_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Decodifica in memoria il container binario offuscato e restituisce il dizionario PyTorch.
    Nessun file .pt temporaneo viene scritto su disco durante questa operazione.
    """
    vault_p = Path(vault_path).resolve()
    if not vault_p.exists():
        raise FileNotFoundError(f"File vault non trovato: {vault_p}")

    with open(vault_p, "rb") as f:
        vault_bytes = f.read()

    if not vault_bytes.startswith(VAULT_MAGIC):
        raise ValueError(f"Firma vault non valida o file corrotto: {vault_p}")

    # 1. Rimuovi header magic
    scrambled_payload = vault_bytes[len(VAULT_MAGIC):]

    # 2. Descramble XOR bitwise
    compressed = _scramble_buffer(scrambled_payload)

    # 3. Decompressione zlib
    raw_bytes = zlib.decompress(compressed)

    # 4. Deserializzazione diretta in memoria tramite PyTorch
    bio = io.BytesIO(raw_bytes)
    try:
        data = torch.load(bio, map_location="cpu", weights_only=False)
    except TypeError:
        bio.seek(0)
        data = torch.load(bio, map_location="cpu")

    return data


def unpack_vault_to_checkpoint(
    vault_path: Union[str, Path],
    dest_pt_path: Union[str, Path],
    verbose: bool = True,
) -> Path:
    """Utility per il docente: ripristina un file .pt standard dal vault offuscato."""
    vault_p = Path(vault_path).resolve()
    dest_p = Path(dest_pt_path).resolve()
    dest_p.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"🔓 Estrazione checkpoint da vault: {vault_p.name} -> {dest_p.name}...")

    ckpt_data = load_checkpoint_from_vault(vault_p)
    torch.save(ckpt_data, str(dest_p))

    if verbose:
        print(f"   ✅ Checkpoint .pt estratto con successo in: {dest_p}")

    return dest_p


def find_vault_file(base_dir: Optional[Path] = None) -> Optional[Path]:
    """Individua il file vault offuscato all'interno del progetto."""
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent

    candidates = [
        base_dir / "checkpoints" / ".vault" / "model_weights.bin",
        base_dir / "checkpoints" / ".vault" / "pretrained_vgg16_detector.bin",
        base_dir / "checkpoints" / ".model_weights.bin",
        base_dir / "pretrain_pipeline" / "checkpoints" / ".vault" / "model_weights.bin",
    ]

    for c in candidates:
        if c.exists() and c.is_file():
            return c

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Gestione del Vault di Sicurezza dei Pesi Pre-Addestrati (Anti-Cheat)"
    )
    parser.add_argument("--pack", type=str, help="Percorso del checkpoint .pt da offuscare nel vault.")
    parser.add_argument("--unpack", type=str, help="Percorso del file .bin del vault da estrarre in .pt.")
    parser.add_argument("--output", "-o", type=str, help="Percorso di output desiderato.")
    parser.add_argument("--status", action="store_true", help="Verifica la presenza e integrità del vault.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent

    if args.status:
        v = find_vault_file(project_root)
        if v:
            print(f"✅ Vault presente: {v} ({v.stat().st_size / (1024 * 1024):.1f} MB)")
            try:
                data = load_checkpoint_from_vault(v)
                print("   -> Integrità verificata: checkpoint decodificabile con successo in memoria.")
            except Exception as e:
                print(f"   ❌ Errore integrità vault: {e}")
        else:
            print("❌ Nessun vault trovato.")
        return

    if args.pack:
        out = Path(args.output) if args.output else project_root / "checkpoints" / ".vault" / "model_weights.bin"
        pack_checkpoint_to_vault(args.pack, out)
        return

    if args.unpack:
        out = Path(args.output) if args.output else project_root / "checkpoints" / "restored_best.pt"
        unpack_vault_to_checkpoint(args.unpack, out)
        return

    parser.print_help()


if __name__ == "__main__":
    main()

