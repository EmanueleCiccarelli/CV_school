#!/usr/bin/env python3
"""
=============================================================================
BlockDL -> VGG-16 Object Detector: Local Python Server & App Launcher
=============================================================================
Avvia l'applicazione web didattica BlockDL (pre-compilata) e fornisce le API
locali per la validazione progressiva e la conversione in YOLO Object Detector.

Zero dipendenze Node.js/npm richieste per gli studenti!
Usa solo la libreria standard di Python.

Uso:
    python run.py [--port 5173] [--no-browser]
=============================================================================
"""

import argparse
import json
import mimetypes
import os
import re
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DIST_DIR = ROOT_DIR / "blockdl-main" / "dist"
NETWORK_DIR = ROOT_DIR / "network to convert"
CONVERT_SCRIPT = ROOT_DIR / "convert_vgg_to_detector.py"
RESET_SCRIPT = ROOT_DIR / "reset_environment.py"
OUTPUT_PT_FILE = ROOT_DIR / "vgg16_detector.pt"

# Assicura tipi MIME corretti
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("text/css", ".css")
mimetypes.add_type("image/svg+xml", ".svg")
mimetypes.add_type("image/webp", ".webp")
mimetypes.add_type("video/mp4", ".mp4")


class BlockDLRequestHandler(SimpleHTTPRequestHandler):
    """Handler HTTP che serve il frontend pre-compilato ed espone le API di conversione."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DIST_DIR), **kwargs)

    def log_message(self, format, *args):
        # Log compatto solo per le chiamate API o errori per non inquinare la console
        path = self.path
        if path.startswith("/api/") or (args and str(args[1])[0] in ("4", "5")):
            sys.stdout.write(f"[{time.strftime('%H:%M:%S')}] {args[0]} -> {args[1]}\n")
            sys.stdout.flush()

    def send_json(self, status: int, data: dict):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        clean_path = self.path.split("?")[0]

        # 1. API: Download vgg16_detector.pt
        if clean_path == "/api/download-detector":
            if OUTPUT_PT_FILE.exists():
                file_size = OUTPUT_PT_FILE.stat().st_size
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Disposition", 'attachment; filename="vgg16_detector.pt"')
                self.send_header("Content-Length", str(file_size))
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()

                with open(OUTPUT_PT_FILE, "rb") as f:
                    # Stream a chunk da 64KB
                    while chunk := f.read(65536):
                        self.wfile.write(chunk)
                return
            else:
                self.send_json(
                    HTTPStatus.NOT_FOUND,
                    {"success": False, "error": "File vgg16_detector.pt non ancora generato."},
                )
                return

        # 2. API: Stato vgg16_detector.pt
        if clean_path == "/api/detector-status":
            if OUTPUT_PT_FILE.exists():
                stat = OUTPUT_PT_FILE.stat()
                self.send_json(
                    HTTPStatus.OK,
                    {
                        "exists": True,
                        "fileName": "vgg16_detector.pt",
                        "size": stat.st_size,
                        "mtime": stat.st_mtime,
                    },
                )
            else:
                self.send_json(HTTPStatus.OK, {"exists": False})
            return

        # 3. Serving Frontend Statico (SPA)
        if not DIST_DIR.exists():
            self.send_response(HTTPStatus.INTERNAL_SERVER_ERROR)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(
                b"<h1>Errore: Cartella 'dist' non trovata</h1><p>Eseguire 'cd blockdl-main && npm run build' per generarla.</p>"
            )
            return

        # Se il percorso richiesto punta a un file statico esistente, servilo normalmente
        requested_file = DIST_DIR / clean_path.lstrip("/")
        if requested_file.is_file():
            super().do_GET()
            return

        # Fallback SPA: per qualsiasi altra rotta (es. /), servi index.html
        index_file = DIST_DIR / "index.html"
        if index_file.exists():
            html = index_file.read_text(encoding="utf-8")
            # Inietta la versione di reset dinamica per supportare reset_environment.py senza ricompilazione
            sandbox_ver_path = ROOT_DIR / "blockdl-main" / "src" / "lib" / "sandbox-version.json"
            reset_ver = ""
            if sandbox_ver_path.exists():
                try:
                    data = json.loads(sandbox_ver_path.read_text(encoding="utf-8"))
                    reset_ver = str(data.get("resetVersion", ""))
                except Exception:
                    pass
            if reset_ver:
                injection = f'<script>window.__SANDBOX_RESET_VERSION__ = "{reset_ver}";</script>'
                if "</head>" in html:
                    html = html.replace("</head>", f"{injection}</head>", 1)
                else:
                    html = f"{injection}{html}"

            encoded = html.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.end_headers()
            self.wfile.write(encoded)
            return

        super().do_GET()

    def do_POST(self):
        clean_path = self.path.split("?")[0]

        # 1. API: Build Detector
        if clean_path == "/api/build-detector":
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length).decode("utf-8")
                data = json.loads(body) if body else {}
                code = data.get("code", "")

                NETWORK_DIR.mkdir(parents=True, exist_ok=True)
                target_py = NETWORK_DIR / "vgg16_blockdl.py"

                if code and isinstance(code, str) and len(code.strip()) > 0:
                    with open(target_py, "w", encoding="utf-8") as f:
                        f.write(code)

                # Esegue la conversione usando l'eseguibile Python corrente (sys.executable)
                proc = subprocess.run(
                    [
                        sys.executable,
                        str(CONVERT_SCRIPT),
                        "--file",
                        str(target_py),
                        "--output",
                        "vgg16_detector.pt",
                    ],
                    cwd=str(ROOT_DIR),
                    capture_output=True,
                    text=True,
                )

                stdout = proc.stdout or ""
                stderr = proc.stderr or ""
                pt_exists = OUTPUT_PT_FILE.exists()

                if proc.returncode == 0 and pt_exists:
                    self.send_json(
                        HTTPStatus.OK,
                        {
                            "success": True,
                            "ptFileName": "vgg16_detector.pt",
                            "ptFilePath": str(OUTPUT_PT_FILE),
                            "downloadUrl": "/api/download-detector",
                            "log": stdout,
                        },
                    )
                else:
                    self.send_json(
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                        {
                            "success": False,
                            "error": stderr or stdout or "Errore durante la conversione del modello in Object Detector.",
                            "log": stdout,
                        },
                    )
            except Exception as e:
                self.send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"success": False, "error": str(e)})
            return

        # 2. API: Reset Environment
        if clean_path == "/api/reset-env":
            try:
                proc = subprocess.run(
                    [sys.executable, str(RESET_SCRIPT), "--force"],
                    cwd=str(ROOT_DIR),
                    capture_output=True,
                    text=True,
                )
                self.send_json(
                    HTTPStatus.OK,
                    {"success": proc.returncode == 0, "log": proc.stdout or proc.stderr},
                )
            except Exception as e:
                self.send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"success": False, "error": str(e)})
            return

        self.send_json(HTTPStatus.NOT_FOUND, {"error": "Not Found"})


def find_free_port(start_port: int = 5173, max_tries: int = 10) -> int:
    """Cerca una porta libera partendo da start_port."""
    for p in range(start_port, start_port + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", p))
                return p
            except OSError:
                continue
    return start_port


def main():
    parser = argparse.ArgumentParser(
        description="Avvia il server locale BlockDL per il laboratorio VGG-16 YOLO"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5173,
        help="Porta su cui ascoltare (predefinita: 5173)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Non aprire automaticamente il browser all'avvio",
    )
    args = parser.parse_args()

    if not DIST_DIR.exists():
        print(f"[!] ERRORE: La cartella pre-compilata non esiste: {DIST_DIR}")
        print("    Esegui prima: cd blockdl-main && npm run build")
        sys.exit(1)

    port = find_free_port(args.port)
    url = f"http://localhost:{port}"

    server = ThreadingHTTPServer(("0.0.0.0", port), BlockDLRequestHandler)

    banner = f"""
======================================================================
 🎓 LABORATORIO COMPUTER VISION: BLOCKDL -> ULTRALYTICS YOLO
======================================================================
  Interfaccia Web:  {url}
  File Detector:    vgg16_detector.pt (generato al Livello 6)
  Interprete Py:    {sys.executable}
======================================================================
  Premi Ctrl+C per arrestare il server locale.
======================================================================
"""
    print(banner)

    if not args.no_browser:
        def open_browser():
            time.sleep(0.6)
            try:
                webbrowser.open(url)
            except Exception:
                pass

        threading.Thread(target=open_browser, daemon=True).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[i] Chiusura server in corso...")
    finally:
        server.server_close()
        print("[✓] Server arrestato correttamente. Arrivederci!")


if __name__ == "__main__":
    main()
