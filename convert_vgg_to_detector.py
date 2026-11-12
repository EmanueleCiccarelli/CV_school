"""
convert_vgg_to_detector.py - Coordinatore Conversione VGG-16 BlockDL in Object Detector YOLO

Questo script fa da punto di ingresso (entry point) principale per il laboratorio scolastico:
1. Ricerca automatica del file esportato da BlockDL nella cartella 'network to convert/' (o 'network_to_convert/').
2. Validazione rigorosa tramite 'vgg_validator.py' (13 Conv2d, 5 MaxPool2d, dimensioni canali e kernel).
   - Se ci sono errori, fornisce indicazioni chiare su come correggere i blocchi grafici su BlockDL.
3. Se l'architettura è valida, illustra agli studenti il processo di adattamento ad Object Detection:
   - Rimozione dei 3 layer densi finali di classificazione.
   - Connessione delle feature map multi-scala C3 (stride 8), C4 (stride 16) e C5 (stride 32).
   - Innesto della YOLO Detect Head.
   - Applicazione del Transfer Learning con pesi pre-addestrati ImageNet per il successivo fine-tuning.
4. Generazione del checkpoint .pt e file .yaml compatibili con Ultralytics Platform tramite 'detector_builder.py'.
5. Stampa comandi pronti all'uso per Ubuntu (Bash) e Windows (PowerShell e CMD).

Uso:
    python convert_vgg_to_detector.py
    python convert_vgg_to_detector.py --nc 2 --imgsz 640
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from vgg_validator import validate_student_vgg
from detector_builder import build_and_export_detector


def find_target_file(explicit_file: Optional[str] = None, base_dir: Optional[Path] = None) -> Optional[Path]:
    """Individua il file .py da convertire, supportando sia 'network to convert' che 'network_to_convert'."""
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent

    if explicit_file:
        target = Path(explicit_file).resolve()
        return target if target.exists() else None

    search_dirs = [
        base_dir / "network to convert",
        base_dir / "network_to_convert",
        base_dir,
    ]

    for directory in search_dirs:
        if directory.exists() and directory.is_dir():
            candidates = sorted(
                [f for f in directory.glob("*.py") if f.is_file() and not f.name.startswith((".", "_"))],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            # Ignoriamo gli script di sistema se cerchiamo nella root
            if directory == base_dir:
                system_names = {
                    "convert_vgg_to_detector.py",
                    "vgg_validator.py",
                    "detector_builder.py",
                    "convert_to_ultralytics.py",
                    "main.py",
                    "prepare_mnist_1_7.py",
                    "prepare_mnist_color_1_7.py",
                    "predict.py",
                }
                candidates = [c for c in candidates if c.name not in system_names]

            if candidates:
                return candidates[0]

    return None


def print_platform_instructions(output_pt: Path, nc: int, imgsz: int):
    """Stampa le istruzioni chiare per l'addestramento su Ultralytics Platform sia per Ubuntu che per Windows."""
    pt_name = output_pt.name
    print("\n" + "=" * 75)
    print("  COME AVVIARE L'ADDESTRAMENTO SU ULTRALYTICS PLATFORM")
    print("=" * 75)
    print("Il tuo modello VGG-16 con testa YOLO è pronto e compatibile al 100%!")
    print("Caricalo su Ultralytics Platform oppure addestralo in locale con i comandi sotto:\n")

    print("[1] Con sincronizzazione su Ultralytics Platform (Cloud):")
    print("    • Ubuntu / Linux (Bash):")
    print(f'      export ULTRALYTICS_API_KEY="INSERISCI_LA_TUA_API_KEY"')
    print(f'      yolo train model={pt_name} data=ul://tuo-dataset task=detect imgsz={imgsz} project=tuo-progetto\n')

    print("    • Windows (PowerShell):")
    print(f'      $env:ULTRALYTICS_API_KEY="INSERISCI_LA_TUA_API_KEY"')
    print(f'      yolo train model={pt_name} data=ul://tuo-dataset task=detect imgsz={imgsz} project=tuo-progetto\n')

    print("    • Windows (Prompt dei Comandi - CMD):")
    print(f'      set ULTRALYTICS_API_KEY=INSERISCI_LA_TUA_API_KEY')
    print(f'      yolo train model={pt_name} data=ul://tuo-dataset task=detect imgsz={imgsz} project=tuo-progetto\n')

    print("[2] Addestramento rapido in locale (es. dataset di prova coco8):")
    print(f"      yolo train model={pt_name} data=coco8.yaml task=detect imgsz={imgsz} epochs=10\n")

    print("[3] Addestramento tramite script Python:")
    print("      from ultralytics import YOLO")
    print(f'      model = YOLO("{pt_name}")')
    print(f'      model.train(data="coco8.yaml", task="detect", epochs=10, imgsz={imgsz})')
    print("=" * 75 + "\n")


def main():
    base_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Converte la rete VGG-16 creata in BlockDL in un Object Detector YOLO compatibile con Ultralytics."
    )
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        default=None,
        help="Percorso del file .py BlockDL (default: primo file in 'network to convert/').",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="vgg16_detector.pt",
        help="Nome o percorso del file .pt in uscita (default: vgg16_detector.pt).",
    )
    parser.add_argument(
        "--nc",
        type=int,
        default=80,
        help="Numero di classi per l'Object Detector (default: 80 per COCO).",
    )
    parser.add_argument(
        "--imgsz",
        "-s",
        type=int,
        default=640,
        help="Risoluzione immagine per il rilevamento oggetti (default: 640).",
    )
    args = parser.parse_args()

    print("=" * 75)
    print("  LABORATORIO COMPUTER VISION: VGG-16 BLOCKDL -> YOLO OBJECT DETECTOR")
    print("=" * 75)

    # 1. Ricerca del file esportato da BlockDL
    input_file = find_target_file(args.file, base_dir=base_dir)

    if not input_file:
        print("\n❌ Nessun file .py trovato nella cartella 'network to convert/' o 'network_to_convert/'.")
        print("\nCosa fare:")
        print("1. Vai su BlockDL (https://blockdl.com/editor) e realizza l'architettura VGG-16 a blocchi.")
        print("2. Clicca su 'Export' -> 'PyTorch' e salva il file all'interno della cartella 'network to convert/'.")
        print("3. Esegui nuovamente: python convert_vgg_to_detector.py\n")
        sys.exit(1)

    print(f"\n📂 File BlockDL individuato: {input_file.name}")
    print(f"📍 Percorso: {input_file}")

    # 2. Validazione rigorosa dell'architettura dello studente
    print("\n[FASE 1/3] Validazione architetturale della rete creata su BlockDL...")
    is_valid = validate_student_vgg(input_file)

    if not is_valid:
        print("🛑 Conversione interrotta: correggi i blocchi su BlockDL seguendo le istruzioni sopra riportate.")
        sys.exit(1)

    # 3. Presentazione pedagogica dell'adattamento ad Object Detection
    print("[FASE 2/3] Adattamento dell'architettura VGG-16 per Object Detection:")
    print("  ✂️  Rimozione del classificatore denso finale (i 3 layer Linear di classificazione)")
    print("  🔗 Connessione delle feature maps multi-scala dal backbone VGG-16:")
    print("      • C3: Scala P3 (stride 8,  256 canali) - rileva oggetti piccoli")
    print("      • C4: Scala P4 (stride 16, 512 canali) - rileva oggetti medi")
    print("      • C5: Scala P5 (stride 32, 512 canali) - rileva oggetti grandi")
    print("  🎯 Innesto della YOLO Detect Head direttamente sulle feature map C3, C4, C5 (senza neck)")
    print("  🧠 Applicazione Transfer Learning: caricamento pesi pre-allenati ImageNet per consentire")
    print("      il fine-tuning immediato con il dataset personalizzato degli studenti.")

    # 4. Costruzione del modello ed esportazione formato Ultralytics .pt
    print("\n[FASE 3/3] Costruzione del modello ed esportazione per Ultralytics...")
    output_path = base_dir / args.output if not Path(args.output).is_absolute() else Path(args.output)
    build_and_export_detector(output_path, nc=args.nc, imgsz=args.imgsz)

    # 5. Istruzioni d'uso per Ultralytics Platform
    print_platform_instructions(output_path, nc=args.nc, imgsz=args.imgsz)


if __name__ == "__main__":
    main()
