"""
reset_environment.py - Reset Completo dell'Ambiente di Laboratorio per la Consegna

Questo script permette al docente di ripristinare completamente l'ambiente di lavoro
allo stato iniziale pulito dopo aver effettuato prove o simulazioni:

1. Rimuove tutti i modelli generati (.pt) e file di configurazione (.yaml) nella root.
2. Svuota completamente la cartella 'network to convert/' (e 'network_to_convert/').
3. Rimuove cartelle temporanee create da Ultralytics o Python ('runs/', 'datasets/', '__pycache__').
4. Resetta l'ambiente interattivo BlockDL (blockdl-main):
   - Genera una nuova versione di reset in sandbox-version.json
   - Al prossimo avvio/refresh nel browser, tutti i livelli 2..6 tornano bloccati
   - Tutte le modifiche salvate nella cache del browser vengono cancellate
   - Il canvas si riapre pulito sul Livello 1 per gli studenti.

Sicurezza:
- Richiede conferma interattiva da tastiera per evitare esecuzioni accidentali.
- Può essere lanciato con il flag '--force' o '-f' per confermare automaticamente.

Uso:
    python reset_environment.py
    python reset_environment.py --force
"""

import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path


def confirm_action() -> bool:
    """Richiede conferma esplicita all'utente da terminale."""
    print("\n" + "!" * 75)
    print("  ATTENZIONE: RIPRISTINO TOTALE DELL'AMBIENTE DI LABORATORIO")
    print("!" * 75)
    print("Questa operazione riporterà l'intera cartella allo stato iniziale per la classe:")
    print("  • Eliminazione di tutti i modelli 'vgg16_detector.pt' e file '.yaml' generati")
    print("  • Svuotamento di tutti i file esportati dentro 'network to convert/'")
    print("  • Eliminazione di eventuali cartelle temporanee ('runs/', 'datasets/', '__pycache__')")
    print("  • Reset totale della sandbox BlockDL (riblocco livelli 2-6, reset al Livello 1)")
    print("!" * 75)

    try:
        response = input("\n👉 Vuoi procedere con il reset completo? Digita 'RESET' per confermare: ").strip()
        return response.upper() in {"RESET", "SI", "S", "YES", "Y"}
    except (KeyboardInterrupt, EOFError):
        print("\n")
        return False


def reset_environment(base_dir: Path) -> None:
    """Esegue la pulizia di tutti i file generati e il reset della sandbox."""
    print("\n" + "=" * 75)
    print("  AVVIO RESET AMBIENTE LABORATORIO...")
    print("=" * 75)

    deleted_count = 0

    # 1. Rimuovi file .pt e .yaml nella cartella radice
    for ext in ("*.pt", "*.yaml"):
        for file_path in base_dir.glob(ext):
            if file_path.is_file():
                try:
                    file_path.unlink()
                    print(f"🗑️  Rimosso file: {file_path.name}")
                    deleted_count += 1
                except Exception as e:
                    print(f"⚠️  Impossibile rimuovere {file_path.name}: {e}")

    # 2. Svuota 'network to convert' e 'network_to_convert'
    search_dirs = [
        base_dir / "network to convert",
        base_dir / "network_to_convert",
    ]

    has_net_dir = False
    for net_dir in search_dirs:
        if net_dir.exists() and net_dir.is_dir():
            has_net_dir = True
            for item in net_dir.iterdir():
                try:
                    if item.is_file():
                        item.unlink()
                        print(f"🧹 Svuotato da '{net_dir.name}': {item.name}")
                        deleted_count += 1
                    elif item.is_dir():
                        shutil.rmtree(item)
                        print(f"🧹 Rimossa sottocartella da '{net_dir.name}': {item.name}")
                        deleted_count += 1
                except Exception as e:
                    print(f"⚠️  Impossibile eliminare {item.name}: {e}")

    # Se nessuna cartella esisteva, creiamo solo 'network to convert'
    if not has_net_dir:
        (base_dir / "network to convert").mkdir(parents=True, exist_ok=True)

    # 3. Rimuovi cartelle temporanee runs/, datasets/, __pycache__/
    temp_folders = ["runs", "datasets"]
    for folder_name in temp_folders:
        folder_path = base_dir / folder_name
        if folder_path.exists() and folder_path.is_dir():
            try:
                shutil.rmtree(folder_path)
                print(f"🗑️  Rimossa cartella temporanea: {folder_name}/")
                deleted_count += 1
            except Exception as e:
                print(f"⚠️  Impossibile rimuovere {folder_name}/: {e}")

    # Rimuovi tutte le cartelle __pycache__
    for pycache in base_dir.rglob("__pycache__"):
        if pycache.is_dir() and "node_modules" not in str(pycache):
            try:
                shutil.rmtree(pycache)
                print(f"🧹 Rimossa cache Python: {pycache.relative_to(base_dir)}")
            except Exception:
                pass

    # 4. Reset della sandbox BlockDL (sandbox-version.json)
    sandbox_version_path = base_dir / "blockdl-main" / "src" / "lib" / "sandbox-version.json"
    if sandbox_version_path.exists():
        try:
            new_version = int(time.time() * 1000)
            data = {
                "resetVersion": new_version,
                "resetAt": datetime.now().isoformat(),
            }
            sandbox_version_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
            print(f"🔄 Versione sandbox BlockDL aggiornata: {new_version}")
            print("   -> Livelli 2..6 ribloccati, cache browser azzerata, riapertura sul Livello 1.")
        except Exception as e:
            print(f"⚠️  Impossibile aggiornare sandbox-version.json: {e}")

    print("=" * 75)
    print("✅ RESET COMPLETATO CON SUCCESSO!")
    print(f"   Totale elementi ripuliti: {deleted_count}")
    print("   L'ambiente è ora esattamente come all'inizio, pronto per la classe.")
    print("=" * 75 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Ripristina l'ambiente di laboratorio allo stato iniziale (cancella modelli .pt, .yaml, file in 'network to convert/' e resetta BlockDL)."
    )
    parser.add_argument(
        "--force",
        "-f",
        "-y",
        action="store_true",
        help="Esegue il reset senza chiedere conferma interattiva.",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent

    if not args.force:
        if not confirm_action():
            print("\n🛑 Operazione annullata. Nessuna modifica apportata all'ambiente.\n")
            sys.exit(0)

    reset_environment(base_dir)


if __name__ == "__main__":
    main()
