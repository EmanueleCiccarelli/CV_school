"""
main.py - Entry point per il laboratorio di Computer Vision (BlockDL -> Ultralytics)

Uso semplice per gli studenti (zero configurazione richiesta):
    python main.py

Cosa fa lo script:
1. Cerca automaticamente il primo file Python (.py) nella cartella 'network to convert/'.
2. Converte la rete definita graficamente in BlockDL in un modello .pt compatibile al 100%
   con Ultralytics Platform e la libreria YOLO (task=classify).
3. Calibra automaticamente i layer lineari (in_features) e i canali (1 -> 3 canali RGB).
4. Genera sia il file .pt sia l'architettura .yaml.
5. Mostra a terminale i comandi pronti da copiare e incollare per avviare l'addestramento
   sia su Ubuntu (Linux) sia su Windows (PowerShell / CMD).
"""

import argparse
import platform
import sys
from pathlib import Path

from convert_to_ultralytics import export_to_ultralytics_pt


def get_first_network_file(folder: Path) -> Path | None:
    """Cerca e restituisce il primo file .py valido all'interno della cartella specificata."""
    if not folder.exists() or not folder.is_dir():
        return None

    py_files = sorted(
        [f for f in folder.glob("*.py") if f.is_file() and not f.name.startswith((".", "_"))],
        key=lambda p: p.name.lower(),
    )
    return py_files[0] if py_files else None


def print_banner():
    print("=" * 70)
    print("  LABORATORIO COMPUTER VISION - BLOCKDL -> ULTRALYTICS CONVERTER")
    print(f"  Sistema operativo rilevato: {platform.system()} ({platform.release()})")
    print("=" * 70)


def print_instructions(pt_path: Path, imgsz: int):
    pt_name = pt_path.name
    print("\n" + "=" * 70)
    print("  COME ADDESTRARE IL TUO MODELLO SU ULTRALYTICS")
    print("=" * 70)

    print("\n[Opzione A] Con sincronizzazione su Ultralytics Platform (Cloud):")
    print("  1. Su Ubuntu / Linux (Bash):")
    print(f'     export ULTRALYTICS_API_KEY="INSERISCI_LA_TUA_API_KEY"')
    print(f'     yolo train model={pt_name} data=ul://tuo-dataset task=classify imgsz={imgsz} project=tuo-progetto')

    print("\n  2. Su Windows (PowerShell):")
    print(f'     $env:ULTRALYTICS_API_KEY="INSERISCI_LA_TUA_API_KEY"')
    print(f'     yolo train model={pt_name} data=ul://tuo-dataset task=classify imgsz={imgsz} project=tuo-progetto')

    print("\n  3. Su Windows (Prompt dei comandi - CMD):")
    print(f'     set ULTRALYTICS_API_KEY=INSERISCI_LA_TUA_API_KEY')
    print(f'     yolo train model={pt_name} data=ul://tuo-dataset task=classify imgsz={imgsz} project=tuo-progetto')

    print("\n[Opzione B] Addestramento completamente in locale (senza API key):")
    print(f"     yolo train model={pt_name} data=percorso/al/tuo/dataset task=classify imgsz={imgsz} epochs=10")

    print("\n[Opzione C] Tramite script Python:")
    print("     from ultralytics import YOLO")
    print(f'     model = YOLO("{pt_name}")')
    print(f'     model.train(data="percorso/al/dataset", epochs=10, imgsz={imgsz})')
    print("=" * 70 + "\n")


def main():
    print_banner()

    base_dir = Path(__file__).resolve().parent
    default_network_dir = base_dir / "network to convert"

    parser = argparse.ArgumentParser(
        description="Converte la rete esportata da BlockDL in un file .pt compatibile con Ultralytics."
    )
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        default=None,
        help="Percorso di un file .py specifico da convertire (opzionale).",
    )
    parser.add_argument(
        "--imgsz",
        "-s",
        type=int,
        default=28,
        help="Risoluzione delle immagini (default: 28 per MNIST; usa 224 per VGG16).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="model.pt",
        help="Nome o percorso del file .pt in uscita (default: model.pt).",
    )
    parser.add_argument(
        "--classes",
        "-c",
        nargs="+",
        default=None,
        help="Nomi delle classi (opzionale, es. --classes 1 7 oppure --classes uno sette).",
    )
    parser.add_argument(
        "--no-rgb",
        action="store_true",
        help="Mantiene il canale grayscale originale a 1 canale anziché convertire a 3 canali RGB.",
    )

    args = parser.parse_args()

    # 1. Trova il file da convertire
    if args.file:
        target_file = Path(args.file).resolve()
        if not target_file.exists():
            print(f"[ERRORE] Il file specificato non esiste: {target_file}")
            sys.exit(1)
    else:
        target_file = get_first_network_file(default_network_dir)
        if target_file is None:
            print(f"[ERRORE] Nessun file .py trovato nella cartella:")
            print(f"         '{default_network_dir}'")
            print("\nCosa fare:")
            print("1. Vai su BlockDL (https://blockdl.com/editor).")
            print("2. Disegna la tua rete e clicca su Export -> PyTorch.")
            print(f"3. Salva il file .py dentro la cartella 'network to convert'.")
            print("4. Esegui nuovamente questo comando: python main.py")
            sys.exit(1)

    print(f"\n[INFO] File selezionato per la conversione: {target_file.name}")
    print(f"[INFO] Percorso: {target_file}")

    # Costruisci mappa classi se fornita
    class_names_map = None
    if args.classes:
        class_names_map = {i: name for i, name in enumerate(args.classes)}

    output_path = base_dir / args.output if not Path(args.output).is_absolute() else Path(args.output)
    force_rgb = not args.no_rgb

    # 2. Esegui la conversione tramite l'engine
    try:
        result = export_to_ultralytics_pt(
            py_file_path=target_file,
            output_pt_path=output_path,
            imgsz=args.imgsz,
            force_rgb=force_rgb,
            class_names=class_names_map,
        )
    except Exception as e:
        print(f"\n[ERRORE durante la conversione]: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 3. Verifica immediata del modello generato tramite forward pass di test
    try:
        from ultralytics import YOLO
        import torch

        print("\n[Verifica] Test di integrità del modello con Ultralytics YOLO...")
        test_model = YOLO(str(output_path))
        dummy_in = torch.zeros(1, result["in_channels"], args.imgsz, args.imgsz)
        with torch.no_grad():
            dummy_out = test_model.model(dummy_in)
        print(f"[Verifica OK] Modello pronto per il training! Task: {test_model.task} | Logits output: {tuple(dummy_out.shape)}")
    except Exception as e:
        print(f"[Avviso verifica]: Impossibile eseguire il forward pass di test: {e}")

    # 4. Mostra istruzioni per il training
    print_instructions(output_path, args.imgsz)


if __name__ == "__main__":
    main()

