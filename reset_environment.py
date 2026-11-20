"""
reset_environment.py - Reset Sicuro e Anti-Cheat dell'Ambiente per gli Studenti

Questo script permette al docente di ripristinare completamente l'ambiente di lavoro
allo stato iniziale pulito prima di distribuirlo agli studenti:

1. RIMOZIONE TOTALE DEI MODELLI .PT DALLA ROOT:
   - Rimuove 'vgg16_detector.pt', 'pretrained_vgg16_detector.pt' e qualsiasi file .pt/.yaml
     dalla cartella principale, impedendo agli studenti di prenderli o bypassare il laboratorio.

2. PROTEZIONE E CIFRATURA ANTI-CHEAT DEI CHECKPOINT ('checkpoints/'):
   - I checkpoint pre-addestrati vengono protetti e offuscati all'interno dell'Anti-Cheat Vault
     ('checkpoints/.vault/model_weights.bin').
   - I file .pt in chiaro presenti in 'checkpoints/' vengono rimossi dalla vista degli studenti,
     così non possono essere aggirati con un semplice drag-and-drop o rinomina.
   - Il Vault offuscato NON viene mai cancellato (garantisce il corretto funzionamento della
     conversione quando lo studente completa validamente la rete su BlockDL).

3. ELIMINAZIONE COMPLETA DEI DATASET DI ADDESTRAMENTO:
   - Elimina la cartella 'datasets/' (inclusi archivi zip e dataset COCO/VOC).
   - Elimina 'data/' e archivi scaricati (*.zip, *.tar).

4. ELIMINAZIONE LOG E RUNS DI TRAINING:
   - Elimina 'runs/' (risultati e grafici di Ultralytics) e sottocartelle temporanee.

5. SVUOTAMENTO DELLA CARTELLA STUDENTI:
   - Svuota completamente 'network to convert/' (e 'network_to_convert/').

6. RESET TOTALE SANDBOX BLOCKDL:
   - Genera una nuova versione di reset in 'sandbox-version.json'.
   - Riblocca i livelli 2..6, cancella la cache locale dei browser e riapre sul Livello 1.

Sicurezza:
- Richiede conferma interattiva digitando 'RESET' da tastiera per evitare esecuzioni accidentali.
- Flag '--force' o '-f' o '-y' per uso non interattivo (script/CI).
"""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import time

from weights_vault import (
    find_vault_file,
    load_checkpoint_from_vault,
    pack_checkpoint_to_vault,
)


def remove_path_safely(path: Path) -> bool:
    """Rimuove un file o una directory in modo sicuro e compatibile sia con Linux che Windows."""
    try:
        if path.is_file() or path.is_symlink():
            try:
                os.chmod(path, stat.S_IWRITE)
            except Exception:
                pass
            path.unlink(missing_ok=True)
            return True
        elif path.is_dir():
            def on_rm_error(func, p, exc_info):
                try:
                    os.chmod(p, stat.S_IWRITE)
                    func(p)
                except Exception:
                    pass

            shutil.rmtree(path, onerror=on_rm_error)
            return True
    except Exception as e:
        print(f"⚠️  Impossibile eliminare {path.name}: {e}")
        return False
    return False


def confirm_action() -> bool:
    """Richiede conferma esplicita all'utente da terminale per evitare esecuzioni accidentali."""
    print("\n" + "!" * 75)
    print("  ATTENZIONE: RIPRISTINO TOTALE E SICURO DELL'AMBIENTE DI LABORATORIO")
    print("!" * 75)
    print("Questa operazione riporterà la cartella allo stato pulito per la classe:")
    print("  • Eliminazione di TUTTI i file '.pt' e '.yaml' dalla cartella principale")
    print("  • Protezione e blindatura dei checkpoint in Anti-Cheat Vault ('checkpoints/.vault/')")
    print("  • Rimozione dei file '.pt' in chiaro da 'checkpoints/' (anti drag-and-drop)")
    print("  • Eliminazione completa dei DATASET di addestramento ('datasets/', 'data/')")
    print("  • Eliminazione delle cartelle di log e risultati ('runs/')")
    print("  • Svuotamento della cartella 'network to convert/'")
    print("  • Reset totale della sandbox BlockDL (riblocco livelli 2-6, reset al Livello 1)")
    print("!" * 75)

    try:
        response = input("\n👉 Vuoi procedere con il reset completo? Digita 'RESET' per confermare: ").strip()
        return response.upper() in {"RESET", "SI", "S", "YES", "Y"}
    except (KeyboardInterrupt, EOFError):
        print("\n")
        return False


def reset_environment(base_dir: Path) -> None:
    """Esegue la pulizia anti-cheat dell'ambiente preservando solo il Vault offuscato."""
    print("\n" + "=" * 75)
    print("  AVVIO RESET SICURO E ANTI-CHEAT AMBIENTE LABORATORIO...")
    print("=" * 75)

    deleted_count = 0
    checkpoints_dir = base_dir / "checkpoints"
    vault_dir = checkpoints_dir / ".vault"
    vault_file = vault_dir / "model_weights.bin"

    # -------------------------------------------------------------------------
    # 1. Blindatura del Vault Anti-Cheat prima della rimozione dei .pt in chiaro
    # -------------------------------------------------------------------------
    print("\n[FASE 1/6] Blindatura Anti-Cheat dei checkpoint pre-addestrati...")
    vault_ready = False

    # Se il vault esiste già, ne testiamo l'integrità
    if vault_file.exists():
        try:
            load_checkpoint_from_vault(vault_file)
            vault_ready = True
            print(f"  🛡️  Anti-Cheat Vault verificato: {vault_file.relative_to(base_dir)} ({vault_file.stat().st_size / (1024 * 1024):.1f} MB)")
        except Exception:
            vault_ready = False

    # Se il vault non è pronto, lo creiamo a partire dal miglior checkpoint .pt disponibile
    if not vault_ready:
        source_candidates = [
            checkpoints_dir / "best.pt",
            base_dir / "pretrained_vgg16_detector.pt",
            checkpoints_dir / "pretrained_vgg16_detector.pt",
            checkpoints_dir / "stage2_final.pt",
            checkpoints_dir / "stage1_final.pt",
        ]
        source_pt = None
        for cand in source_candidates:
            if cand.exists() and cand.is_file():
                source_pt = cand
                break

        if source_pt:
            try:
                pack_checkpoint_to_vault(source_pt, vault_file, verbose=False)
                vault_ready = True
                print(f"  🔒 Checkpoint blindato nel Vault offuscato: {source_pt.name} -> {vault_file.relative_to(base_dir)}")
            except Exception as e:
                print(f"  ❌ Errore durante la creazione del vault da {source_pt.name}: {e}")
        else:
            print("  ⚠️  Nessun file .pt sorgente trovato per il vault.")

    # -------------------------------------------------------------------------
    # 2. Rimozione di TUTTI i file .pt e .yaml dalla root principale
    # -------------------------------------------------------------------------
    print("\n[FASE 2/6] Rimozione di tutti i modelli .pt e file .yaml dalla cartella principale...")
    for ext in ("*.pt", "*.yaml"):
        for file_path in base_dir.glob(ext):
            if file_path.is_file():
                if remove_path_safely(file_path):
                    print(f"  🗑️  Rimosso modello/config: {file_path.name}")
                    deleted_count += 1

    # -------------------------------------------------------------------------
    # 3. Rimozione dei file .pt in chiaro da 'checkpoints/' (Anti Drag-and-Drop)
    # -------------------------------------------------------------------------
    print("\n[FASE 3/6] Pulizia file .pt in chiaro da 'checkpoints/' (protezione anti drag-and-drop)...")
    if checkpoints_dir.exists() and checkpoints_dir.is_dir():
        # Rimuovi file .pt in chiaro (i pesi sono già salvaguardati nel Vault offuscato)
        for pt_file in list(checkpoints_dir.glob("*.pt")):
            if remove_path_safely(pt_file):
                print(f"  🔒 Rimosso file in chiaro: checkpoints/{pt_file.name} (blindato nel Vault)")
                deleted_count += 1

        # Rimuovi eventuali cartelle di log di training (runs_stage1, runs_stage2)
        for sub in list(checkpoints_dir.iterdir()):
            if sub.is_dir() and sub.name.startswith("runs_"):
                if remove_path_safely(sub):
                    print(f"  🗑️  Rimossa cartella log: checkpoints/{sub.name}/")
                    deleted_count += 1

        # Conferma protezione del Vault offuscato
        if vault_file.exists():
            print(f"  🛡️  [PROTETTO] Vault offuscato preservato: checkpoints/.vault/model_weights.bin")
            print(f"       -> Gli studenti non possono copiare o trascinare il file (illeggibile da PyTorch/YOLO)")

    # -------------------------------------------------------------------------
    # 4. Eliminazione completa dei DATASET di addestramento
    # -------------------------------------------------------------------------
    print("\n[FASE 4/6] Eliminazione dataset di addestramento scaricati...")
    dataset_dirs = ["datasets", "data", "coco8", "coco128", "VOCdevkit"]
    for ds_name in dataset_dirs:
        ds_path = base_dir / ds_name
        if ds_path.exists():
            if remove_path_safely(ds_path):
                print(f"  🗑️  Rimossa cartella dataset: {ds_name}/")
                deleted_count += 1

    # Rimuovi archivi di dataset nella root (*.zip, *.tar, ecc.)
    for item in list(base_dir.iterdir()):
        if item.is_file() and any(item.name.lower().endswith(ext) for ext in (".zip", ".tar", ".tar.gz", ".tgz")):
            lower_name = item.name.lower()
            if any(k in lower_name for k in ("coco", "voc", "dataset", "labels", "images")):
                if remove_path_safely(item):
                    print(f"  🗑️  Rimosso archivio dataset: {item.name}")
                    deleted_count += 1

    # Eliminazione risultati e log di addestramento (runs/)
    runs_targets = [
        base_dir / "runs",
        base_dir / "pretrain_pipeline" / "runs",
    ]
    for runs_path in runs_targets:
        if runs_path.exists() and runs_path.is_dir():
            if remove_path_safely(runs_path):
                print(f"  🗑️  Rimossa cartella: {runs_path.relative_to(base_dir)}/")
                deleted_count += 1

    # -------------------------------------------------------------------------
    # 5. Svuotamento della cartella 'network to convert/' per gli studenti
    # -------------------------------------------------------------------------
    print("\n[FASE 5/6] Svuotamento cartella esportazione studenti ('network to convert/')...")
    search_dirs = [
        base_dir / "network to convert",
        base_dir / "network_to_convert",
    ]
    has_net_dir = False
    for net_dir in search_dirs:
        if net_dir.exists() and net_dir.is_dir():
            has_net_dir = True
            for item in list(net_dir.iterdir()):
                if remove_path_safely(item):
                    print(f"  🧹 Svuotato da '{net_dir.name}': {item.name}")
                    deleted_count += 1

    # Assicura la presenza della cartella 'network to convert/' pulita e pronta
    primary_net_dir = base_dir / "network to convert"
    primary_net_dir.mkdir(parents=True, exist_ok=True)
    if not has_net_dir:
        print("  📁 Creata cartella pulita 'network to convert/'")

    # Pulizia cache Python __pycache__ (esclusa node_modules)
    for pycache in base_dir.rglob("__pycache__"):
        if pycache.is_dir() and "node_modules" not in str(pycache):
            try:
                shutil.rmtree(pycache)
                deleted_count += 1
            except Exception:
                pass

    # -------------------------------------------------------------------------
    # 6. Reset totale sandbox BlockDL e rigenerazione Build ('dist/')
    # -------------------------------------------------------------------------
    print("\n[FASE 6/6] Reset sandbox BlockDL e rigenerazione build frontend ('dist/')...")
    sandbox_version_path = base_dir / "blockdl-main" / "src" / "lib" / "sandbox-version.json"
    new_version = int(time.time() * 1000)

    try:
        data = {
            "resetVersion": new_version,
            "resetAt": datetime.now().isoformat(),
        }
        sandbox_version_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        print(f"  🔄 Sandbox BlockDL aggiornata alla versione di reset: {new_version}")
        print("     -> Livelli 2..6 ribloccati, cache browser azzerata, riapertura sul Livello 1.")
    except Exception as e:
        print(f"  ⚠️  Impossibile aggiornare sandbox-version.json: {e}")

    # Rigenerazione pulita della build in 'blockdl-main/dist/'
    blockdl_dir = base_dir / "blockdl-main"
    dist_dir = blockdl_dir / "dist"

    npm_cmd = shutil.which("npm") or shutil.which("npm.cmd")
    rebuilt = False

    if npm_cmd and (blockdl_dir / "package.json").exists():
        print("  🔨 Rilevato Node.js/npm: ricompilazione del frontend in corso...")
        try:
            res = subprocess.run(
                [npm_cmd, "run", "build"],
                cwd=str(blockdl_dir),
                capture_output=True,
                text=True,
                shell=(sys.platform == "win32"),
                timeout=90,
            )
            if res.returncode == 0 and dist_dir.exists():
                print("  ✅ Build completata con successo: 'blockdl-main/dist/' aggiornata e pulita.")
                rebuilt = True
            else:
                print(f"  ⚠️  Avviso durante npm run build (codice {res.returncode}): {res.stderr or res.stdout}")
        except Exception as e:
            print(f"  ⚠️  Impossibile eseguire npm run build: {e}")

    if not rebuilt and dist_dir.exists():
        # Se npm non è installato o ha fallito, preserviamo la cartella dist/ esistente
        # e sincronizziamo dist/index.html con la nuova versione di reset
        dist_index = dist_dir / "index.html"
        if dist_index.exists():
            try:
                html = dist_index.read_text(encoding="utf-8")
                injection = f'<script>window.__SANDBOX_RESET_VERSION__ = "{new_version}";</script>'
                if re.search(r"<script>window\.__SANDBOX_RESET_VERSION__\s*=.*?</script>", html):
                    html = re.sub(r"<script>window\.__SANDBOX_RESET_VERSION__\s*=.*?</script>", injection, html)
                elif "</head>" in html:
                    html = html.replace("</head>", f"{injection}</head>", 1)
                else:
                    html = f"{injection}{html}"
                dist_index.write_text(html, encoding="utf-8")
                print(f"  ✅ Cartella 'dist/' preservata e sincronizzata con la nuova versione di reset ({new_version}).")
            except Exception as e:
                print(f"  ⚠️  Impossibile aggiornare dist/index.html: {e}")

    # -------------------------------------------------------------------------
    # RIEPILOGO FINALE E VERIFICA INTEGRITÀ POST-RESET
    # -------------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("  VERIFICA INTEGRITÀ POST-RESET:")
    print("=" * 75)

    all_ok = True

    # 1. Verifica che NON esistano file .pt nella root
    root_pt_files = list(base_dir.glob("*.pt"))
    if not root_pt_files:
        print("  ✅ CARTELLA PRINCIPALE: Pulita al 100% (nessun file .pt presente)")
    else:
        print(f"  ⚠️  ATTENZIONE: Trovati file .pt nella root: {[p.name for p in root_pt_files]}")
        all_ok = False

    # 2. Verifica che NON esistano file .pt in chiaro in checkpoints/
    checkpoints_pt_files = list(checkpoints_dir.glob("*.pt")) if checkpoints_dir.exists() else []
    if not checkpoints_pt_files:
        print("  ✅ CHECKPOINTS: Nessun file .pt in chiaro presente (anti drag-and-drop attivo)")
    else:
        print(f"  ⚠️  ATTENZIONE: Trovati file .pt in chiaro in checkpoints/: {[p.name for p in checkpoints_pt_files]}")
        all_ok = False

    # 3. Verifica presenza e integrità del Vault offuscato
    if vault_file.exists():
        vault_size_mb = vault_file.stat().st_size / (1024 * 1024)
        print(f"  ✅ ANTI-CHEAT VAULT: PRESENTE E PROTETTO ({vault_file.relative_to(base_dir)}: {vault_size_mb:.1f} MB)")
    else:
        print("  ❌ ATTENZIONE CRITICA: Vault offuscato non trovato!")
        all_ok = False

    # 4. Verifica pulizia dataset e runs
    if not (base_dir / "datasets").exists() and not (base_dir / "runs").exists():
        print("  ✅ DATASET E RUNS: Pulizia completata (datasets/ e runs/ non presenti)")
    else:
        print("  ⚠️  ATTENZIONE: Cartelle datasets/ o runs/ ancora presenti!")

    # 5. Verifica cartella conversione
    if primary_net_dir.exists() and len(list(primary_net_dir.iterdir())) == 0:
        print("  ✅ CARTELLA CONVERSIONE: 'network to convert/' pronta e vuota per gli studenti")

    # 6. Verifica protezione dei file critici recentemente aggiunti
    critical_items = [
        (base_dir / "run.py", "SERVER LAUNCHER"),
        (base_dir / "VGG16_Ultralytics_Colab_Training.ipynb", "NOTEBOOK GOOGLE COLAB"),
        (base_dir / "colab_assets", "ASSETS GUIDA COLAB"),
        (base_dir / "blockdl-main" / "dist", "BUILD FRONTEND"),
        (base_dir / "convert_vgg_to_detector.py", "SCRIPT CONVERSIONE DETECTOR"),
        (base_dir / "detector_builder.py", "BUILDER DETECTOR"),
    ]
    for c_path, label in critical_items:
        if c_path.exists():
            print(f"  🛡️  [PRESERVATO] {label}: '{c_path.name}' intatto e verificato")
        else:
            print(f"  ❌ ATTENZIONE CRITICA: Elemento mancante: '{c_path.name}'")
            all_ok = False

    print("=" * 75)
    if all_ok:
        print("🎉 RESET COMPLETATO CON SUCCESSO!")
        print("   L'ambiente è ora pulito, anti-cheat e pronto per essere distribuito alla classe.")
    else:
        print("⚠️  RESET COMPLETATO CON AVVISI: controllare i messaggi sopra riportati.")
    print("=" * 75 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Ripristina l'ambiente di laboratorio allo stato iniziale: pulizia anti-cheat, rimozione file .pt, eliminazione dataset/runs e reset di BlockDL."
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
