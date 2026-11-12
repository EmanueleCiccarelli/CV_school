import type { Plugin } from "vite";
import path from "path";
import fs from "fs";
import { spawn, spawnSync } from "child_process";

/**
 * Cerca un eseguibile Python configurato con torch e ultralytics.
 */
function findPythonExecutable(projectRoot?: string): string {
  // 1. Variabile d'ambiente esplicita
  if (process.env.PYTHON_EXEC && fs.existsSync(process.env.PYTHON_EXEC)) {
    return process.env.PYTHON_EXEC;
  }

  const isWin = process.platform === "win32";
  const candidates: (string | null | undefined)[] = [
    // Venv locale all'interno del progetto se presente
    projectRoot
      ? isWin
        ? path.join(projectRoot, ".venv", "Scripts", "python.exe")
        : path.join(projectRoot, ".venv", "bin", "python")
      : null,
    // Conda / Venv attivi nel terminale dell'utente
    process.env.CONDA_PREFIX
      ? isWin
        ? path.join(process.env.CONDA_PREFIX, "python.exe")
        : path.join(process.env.CONDA_PREFIX, "bin", "python")
      : null,
    process.env.VIRTUAL_ENV
      ? isWin
        ? path.join(process.env.VIRTUAL_ENV, "Scripts", "python.exe")
        : path.join(process.env.VIRTUAL_ENV, "bin", "python")
      : null,
    // Percorso predefinito conda per l'ambiente YOLO su questa macchina
    "/home/yoshi/anaconda3/envs/yolo/bin/python",
    // Binari generici di sistema
    isWin ? "python.exe" : "python3",
    "python",
    "py",
  ];

  for (const cand of candidates) {
    if (!cand) continue;
    try {
      const res = spawnSync(cand, ["-c", "import torch, ultralytics; print(1)"], {
        encoding: "utf8",
        timeout: 4000,
        shell: isWin,
      });
      if (res.status === 0) {
        return cand;
      }
    } catch {
      // Continua la ricerca
    }
  }

  return isWin ? "python" : "python3";
}

/**
 * Plugin Vite che integra le API locali per la generazione e l'esportazione
 * dell'Object Detector VGG-16 + YOLO Detect Head per Ultralytics Platform.
 */
export function detectorApiPlugin(): Plugin {
  return {
    name: "vite-plugin-blockdl-detector-api",
    configureServer(server) {
      server.middlewares.use(async (req, res, next) => {
        const url = req.url ? req.url.split("?")[0] : "";
        const rootDir = path.resolve(__dirname, "..");
        const isWin = process.platform === "win32";

        // 1. Endpoint: Creazione dell'Object Detector
        if (url === "/api/build-detector" && req.method === "POST") {
          let body = "";
          req.on("data", (chunk) => {
            body += chunk;
          });

          req.on("end", () => {
            try {
              const parsed = body ? JSON.parse(body) : {};
              const code = parsed.code || "";

              const networkDir = path.join(rootDir, "network to convert");
              if (!fs.existsSync(networkDir)) {
                fs.mkdirSync(networkDir, { recursive: true });
              }

              const targetPyFile = path.join(networkDir, "vgg16_blockdl.py");
              if (code && typeof code === "string" && code.trim().length > 0) {
                fs.writeFileSync(targetPyFile, code, "utf8");
              }

              const pythonExec = findPythonExecutable(rootDir);
              const scriptPath = path.join(rootDir, "convert_vgg_to_detector.py");
              const outputPtPath = path.join(rootDir, "vgg16_detector.pt");

              const child = spawn(
                pythonExec,
                [scriptPath, "--file", targetPyFile, "--output", "vgg16_detector.pt"],
                {
                  cwd: rootDir,
                  env: { ...process.env },
                  shell: isWin,
                }
              );

              let stdout = "";
              let stderr = "";

              child.stdout.on("data", (data) => {
                stdout += data.toString();
              });

              child.stderr.on("data", (data) => {
                stderr += data.toString();
              });

              child.on("close", (exitCode) => {
                const ptExists = fs.existsSync(outputPtPath);

                res.setHeader("Content-Type", "application/json");
                if (exitCode === 0 && ptExists) {
                  res.statusCode = 200;
                  res.end(
                    JSON.stringify({
                      success: true,
                      ptFileName: "vgg16_detector.pt",
                      ptFilePath: outputPtPath,
                      downloadUrl: "/api/download-detector",
                      log: stdout,
                    })
                  );
                } else {
                  res.statusCode = 500;
                  res.end(
                    JSON.stringify({
                      success: false,
                      error: stderr || stdout || "Errore durante la conversione del modello in Object Detector.",
                      log: stdout,
                    })
                  );
                }
              });
            } catch (err: unknown) {
              res.statusCode = 500;
              res.setHeader("Content-Type", "application/json");
              res.end(
                JSON.stringify({
                  success: false,
                  error: err instanceof Error ? err.message : String(err),
                })
              );
            }
          });
          return;
        }

        // 2. Endpoint: Download diretto del file .pt generato
        if (url === "/api/download-detector" && req.method === "GET") {
          const ptPath = path.join(rootDir, "vgg16_detector.pt");
          if (fs.existsSync(ptPath)) {
            const stat = fs.statSync(ptPath);
            res.writeHead(200, {
              "Content-Type": "application/octet-stream",
              "Content-Length": stat.size,
              "Content-Disposition": 'attachment; filename="vgg16_detector.pt"',
            });
            const stream = fs.createReadStream(ptPath);
            stream.pipe(res);
            return;
          } else {
            res.statusCode = 404;
            res.setHeader("Content-Type", "application/json");
            res.end(JSON.stringify({ success: false, error: "File vgg16_detector.pt non ancora generato." }));
            return;
          }
        }

        // 3. Endpoint: Controllo stato modello
        if (url === "/api/detector-status" && req.method === "GET") {
          const ptPath = path.join(rootDir, "vgg16_detector.pt");
          res.setHeader("Content-Type", "application/json");
          if (fs.existsSync(ptPath)) {
            const stat = fs.statSync(ptPath);
            res.end(
              JSON.stringify({
                exists: true,
                fileName: "vgg16_detector.pt",
                size: stat.size,
                mtime: stat.mtime,
              })
            );
          } else {
            res.end(JSON.stringify({ exists: false }));
          }
          return;
        }

        next();
      });
    },
  };
}
