import { useState, useEffect, useCallback } from "react";
import {
  CheckCircle2,
  AlertTriangle,
  ArrowRight,
  Wrench,
  Download,
  Copy,
  Check,
  Terminal,
  Cpu,
  Layers,
  Sparkles,
  Rocket,
  RefreshCw,
  ChevronDown,
  ChevronUp,
  Sliders,
} from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "./ui/dialog";
import { Button } from "./ui/button";
import type { ValidationResult } from "../lib/vgg16Validator";

interface ValidationDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  result: ValidationResult | null;
  onProceed: () => void;
  isLastLevel: boolean;
  generatedCode?: string;
}

export function ValidationDialog({
  open,
  onOpenChange,
  result,
  onProceed,
  isLastLevel,
  generatedCode = "",
}: ValidationDialogProps) {
  // Conversion state for Level 6 (Final Object Detector)
  const [conversionStatus, setConversionStatus] = useState<"idle" | "building" | "success" | "error">("idle");
  const [conversionLog, setConversionLog] = useState<string>("");
  const [errorMessage, setErrorMessage] = useState<string>("");
  const [showTerminalLog, setShowTerminalLog] = useState<boolean>(false);
  const [activePlatformTab, setActivePlatformTab] = useState<"cloud" | "local">("cloud");
  const [activeOs, setActiveOs] = useState<"bash" | "powershell" | "cmd">("bash");
  const [copiedKey, setCopiedKey] = useState<string | null>(null);

  const isSuccess = result?.isValid ?? false;
  const isFinalStep = isLastLevel && isSuccess;

  // Trigger automated detector creation when Level 6 validation passes
  const runDetectorCreation = useCallback(async () => {
    setConversionStatus("building");
    setErrorMessage("");

    try {
      const response = await fetch("/api/build-detector", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ code: generatedCode }),
      });

      const data = await response.json();

      if (response.ok && data.success) {
        setConversionStatus("success");
        setConversionLog(data.log || "");
      } else {
        setConversionStatus("error");
        setErrorMessage(data.error || "Errore durante la conversione in Object Detector.");
        if (data.log) setConversionLog(data.log);
      }
    } catch (err: unknown) {
      setConversionStatus("error");
      setErrorMessage(
        err instanceof Error
          ? err.message
          : "Impossibile contattare il server locale per la conversione. Assicurati che 'npm run dev' sia attivo."
      );
    }
  }, [generatedCode]);

  useEffect(() => {
    if (open && isFinalStep) {
      // Automatic trigger if not already converted
      if (conversionStatus === "idle" || conversionStatus === "error") {
        runDetectorCreation();
      }
    } else if (!open) {
      // Keep state if successful, or reset if closed
      if (conversionStatus === "error") {
        setConversionStatus("idle");
      }
    }
  }, [open, isFinalStep, conversionStatus, runDetectorCreation]);

  const handleCopy = (text: string, key: string) => {
    navigator.clipboard.writeText(text);
    setCopiedKey(key);
    setTimeout(() => setCopiedKey(null), 2500);
  };

  if (!result) return null;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        className={`p-0 overflow-hidden border-0 shadow-2xl rounded-2xl transition-all duration-300 ${
          isFinalStep ? "max-w-3xl" : "max-w-lg"
        }`}
      >
        {/* Banner Header */}
        <div
          className={`p-6 text-white ${
            isFinalStep
              ? "bg-gradient-to-r from-indigo-700 via-purple-700 to-blue-700"
              : isSuccess
              ? "bg-gradient-to-r from-emerald-600 to-teal-600"
              : "bg-gradient-to-r from-amber-600 to-rose-600"
          }`}
        >
          <div className="flex items-center gap-3.5">
            <div
              className={`p-3 rounded-2xl ${
                isFinalStep
                  ? "bg-white/20 backdrop-blur-xs shadow-inner"
                  : isSuccess
                  ? "bg-emerald-500/30"
                  : "bg-white/20"
              }`}
            >
              {isFinalStep ? (
                <Rocket className="h-8 w-8 text-amber-300 animate-bounce" />
              ) : isSuccess ? (
                <CheckCircle2 className="h-7 w-7 text-white" />
              ) : (
                <AlertTriangle className="h-7 w-7 text-white" />
              )}
            </div>
            <div>
              <span className="text-xs uppercase tracking-wider font-semibold opacity-90 flex items-center gap-1.5">
                {isFinalStep ? (
                  <>
                    <Sparkles className="h-3.5 w-3.5 text-amber-300" />
                    <span>Architettura VGG16 Completata • Ultralytics Export</span>
                  </>
                ) : isSuccess ? (
                  "Validazione Superata"
                ) : (
                  "Attenzione: Riscontrate Incongruenze"
                )}
              </span>
              <DialogTitle className="text-xl sm:text-2xl font-bold text-white mt-0.5">
                {isFinalStep ? "🎉 VGG16 Pronta per l'Object Detection!" : result.title}
              </DialogTitle>
            </div>
          </div>
        </div>

        {/* Content Body */}
        <div className="p-6 space-y-5 bg-white text-slate-700 max-h-[75vh] overflow-y-auto">
          {/* STANDARD LEVELS 1 - 5 OR ERROR VIEW */}
          {!isFinalStep ? (
            <>
              <DialogDescription className="text-sm leading-relaxed text-slate-700 font-normal">
                {result.message}
              </DialogDescription>

              {/* Diagnostic Details */}
              {result.details && result.details.length > 0 && (
                <div
                  className={`p-4 rounded-xl border text-xs space-y-2 ${
                    isSuccess
                      ? "bg-emerald-50/70 border-emerald-200 text-emerald-800"
                      : "bg-amber-50/80 border-amber-200 text-amber-900"
                  }`}
                >
                  <div className="font-semibold flex items-center gap-1.5 mb-1 text-sm">
                    <Wrench className="h-4 w-4" />
                    <span>Indicazioni di correzione:</span>
                  </div>
                  <ul className="list-disc list-inside space-y-1.5 pl-1">
                    {result.details.map((detail, idx) => (
                      <li key={idx} className="leading-relaxed">
                        {detail}
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {isSuccess && !isLastLevel && (
                <div className="bg-indigo-50/70 border border-indigo-200 rounded-xl p-3.5 text-xs text-indigo-900 flex items-start gap-2.5">
                  <span className="text-base">💡</span>
                  <div>
                    <strong>Prossimo step:</strong> Nel livello successivo troverai questo blocco già integrato nel canvas per continuare la costruzione della VGG16!
                  </div>
                </div>
              )}
            </>
          ) : (
            /* =========================================================================
               LEVEL 6: DEDICATED OBJECT DETECTOR ADAPTATION & EXPORT EXPERIENCE
               ========================================================================= */
            <div className="space-y-5">
              <p className="text-sm text-slate-600 leading-relaxed">
                Congratulazioni! Hai completato l'intera architettura <strong>VGG16</strong> con tutti i 13 layer convoluzionali e i 5 MaxPool. Stiamo ora adattando la tua rete per trasformarla in un <strong>Object Detector Ultralytics</strong> pronto per il training.
              </p>

              {/* Progress & Adaptation Breakdown Card */}
              <div className="bg-slate-50 border border-slate-200 rounded-2xl p-4.5 space-y-3.5 shadow-2xs">
                <div className="flex items-center justify-between border-b border-slate-200/80 pb-2.5">
                  <span className="text-xs font-bold uppercase tracking-wider text-slate-500 flex items-center gap-1.5">
                    <Layers className="h-4 w-4 text-indigo-600" />
                    Adattamento della tua rete in Object Detector
                  </span>
                  {conversionStatus === "building" && (
                    <span className="inline-flex items-center gap-1.5 text-xs font-semibold px-2.5 py-0.5 rounded-full bg-indigo-100 text-indigo-700 animate-pulse">
                      <RefreshCw className="h-3 w-3 animate-spin" />
                      Elaborazione in corso...
                    </span>
                  )}
                  {conversionStatus === "success" && (
                    <span className="inline-flex items-center gap-1.5 text-xs font-semibold px-2.5 py-0.5 rounded-full bg-emerald-100 text-emerald-800">
                      <CheckCircle2 className="h-3 w-3 text-emerald-600" />
                      Modello Pronto
                    </span>
                  )}
                  {conversionStatus === "error" && (
                    <span className="inline-flex items-center gap-1.5 text-xs font-semibold px-2.5 py-0.5 rounded-full bg-rose-100 text-rose-800">
                      <AlertTriangle className="h-3 w-3 text-rose-600" />
                      Verifica Richiesta
                    </span>
                  )}
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5 text-xs">
                  <div className="p-3 bg-white rounded-xl border border-slate-200/70 space-y-1">
                    <div className="font-semibold text-slate-800 flex items-center gap-1.5">
                      <span className="text-emerald-600">✂️</span> 1. Rimozione Classifier
                    </div>
                    <p className="text-slate-500 text-[11px] leading-snug">
                      Disconnessione dei 3 layer lineari finali (1000 classi) per fare spazio al rilevamento oggetti multi-scala.
                    </p>
                  </div>

                  <div className="p-3 bg-white rounded-xl border border-slate-200/70 space-y-1">
                    <div className="font-semibold text-slate-800 flex items-center gap-1.5">
                      <span className="text-indigo-600">🔗</span> 2. Taps Multi-Scala C3-C5
                    </div>
                    <p className="text-slate-500 text-[11px] leading-snug">
                      Connessione diretta delle feature map a diverse risoluzioni: C3 (piccoli), C4 (medi), C5 (grandi).
                    </p>
                  </div>

                  <div className="p-3 bg-white rounded-xl border border-slate-200/70 space-y-1">
                    <div className="font-semibold text-slate-800 flex items-center gap-1.5">
                      <span className="text-blue-600">🎯</span> 3. Innesto YOLO Detect Head
                    </div>
                    <p className="text-slate-500 text-[11px] leading-snug">
                      Integrazione della testa di predizione Ultralytics (bounding box + confidenze) direttamente sul backbone.
                    </p>
                  </div>

                  <div className="p-3 bg-white rounded-xl border border-slate-200/70 space-y-1">
                    <div className="font-semibold text-slate-800 flex items-center gap-1.5">
                      <span className="text-amber-500">🧠</span> 4. Pesi ImageNet (Transfer Learning)
                    </div>
                    <p className="text-slate-500 text-[11px] leading-snug">
                      Inizializzazione con pesi pre-allenati per permettere il fine-tuning immediato con il tuo dataset!
                    </p>
                  </div>
                </div>
              </div>

              {/* Conversion Feedback / Actions */}
              {conversionStatus === "building" && (
                <div className="p-6 text-center space-y-3 bg-indigo-50/50 rounded-2xl border border-indigo-100">
                  <RefreshCw className="h-8 w-8 text-indigo-600 animate-spin mx-auto" />
                  <div className="text-sm font-semibold text-indigo-900">
                    Generazione del modello PyTorch <code className="bg-indigo-100/70 px-1.5 py-0.5 rounded text-indigo-800 font-mono">vgg16_detector.pt</code>...
                  </div>
                  <p className="text-xs text-indigo-700/80 max-w-md mx-auto">
                    Stiamo assemblando il DetectionModel Ultralytics e validando l'inferenza con un forward pass di test.
                  </p>
                </div>
              )}

              {conversionStatus === "error" && (
                <div className="p-4 bg-rose-50 border border-rose-200 rounded-xl space-y-3 text-xs text-rose-800">
                  <div className="flex items-center gap-2 font-bold text-sm text-rose-900">
                    <AlertTriangle className="h-4 w-4 text-rose-600" />
                    <span>Errore durante la generazione del modello</span>
                  </div>
                  <p className="leading-relaxed font-mono bg-rose-100/60 p-2.5 rounded-lg overflow-x-auto text-[11px]">
                    {errorMessage}
                  </p>
                  <div className="flex items-center justify-between pt-1">
                    <span className="text-slate-600">
                      Puoi anche generarlo manualmente da terminale con: <code className="bg-slate-200 px-1 py-0.5 rounded font-mono text-slate-800">python convert_vgg_to_detector.py</code>
                    </span>
                    <Button
                      size="sm"
                      onClick={runDetectorCreation}
                      className="bg-rose-600 hover:bg-rose-700 text-white font-medium cursor-pointer"
                    >
                      <RefreshCw className="h-3.5 w-3.5 mr-1" />
                      Riprova
                    </Button>
                  </div>
                </div>
              )}

              {conversionStatus === "success" && (
                <div className="space-y-4">
                  {/* Success Banner & Download Action */}
                  <div className="p-4 bg-emerald-50 border border-emerald-200 rounded-2xl flex flex-col sm:flex-row items-center justify-between gap-4">
                    <div className="flex items-center gap-3">
                      <div className="p-2.5 bg-emerald-600 text-white rounded-xl">
                        <Cpu className="h-6 w-6" />
                      </div>
                      <div>
                        <div className="text-sm font-bold text-emerald-950 flex items-center gap-2">
                          <span>vgg16_detector.pt</span>
                          <span className="text-[10px] bg-emerald-200/80 text-emerald-800 font-bold px-2 py-0.5 rounded-full uppercase tracking-wider">
                            Ultralytics Ready
                          </span>
                        </div>
                        <p className="text-xs text-emerald-800 mt-0.5">
                          Salvato automaticamente nella cartella di progetto: <code className="font-mono bg-emerald-100/80 px-1 rounded text-emerald-900">vgg16_detector.pt</code>
                        </p>
                      </div>
                    </div>

                    <a
                      href="/api/download-detector"
                      download="vgg16_detector.pt"
                      className="w-full sm:w-auto inline-flex items-center justify-center gap-2 px-4 py-2.5 bg-emerald-600 hover:bg-emerald-700 text-white text-xs font-bold rounded-xl shadow-sm transition-colors cursor-pointer"
                    >
                      <Download className="h-4 w-4" />
                      Scarica .pt
                    </a>
                  </div>

                  {/* Fine-Tuning Guidance: Parameters to Freeze */}
                  <div className="p-4 bg-indigo-50/80 border border-indigo-200 rounded-2xl space-y-3 shadow-2xs">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-bold uppercase tracking-wider text-indigo-950 flex items-center gap-1.5">
                        <Sliders className="h-4 w-4 text-indigo-600" />
                        Guida al Fine-Tuning: Quanti parametri bloccare?
                      </span>
                      <span className="text-[10px] bg-indigo-200/90 text-indigo-900 font-bold px-2 py-0.5 rounded-full uppercase tracking-wider font-mono">
                        freeze: 1
                      </span>
                    </div>

                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5 text-xs">
                      <div className="p-3 bg-white rounded-xl border border-indigo-100 space-y-1">
                        <div className="text-[11px] font-semibold text-slate-500 uppercase tracking-wide">
                          Parametri da Congelare (Backbone)
                        </div>
                        <div className="text-base font-bold text-indigo-700">
                          14,723,136 <span className="text-xs font-normal text-slate-500">(72.3%)</span>
                        </div>
                        <p className="text-[11px] text-slate-600 leading-snug">
                          Le feature convoluzionali VGG pre-addestrate su ImageNet. Imposta <code className="bg-indigo-100/80 px-1 py-0.5 rounded font-mono font-bold text-indigo-900">freeze: 1</code> in <em>Advanced Settings</em> per proteggerle!
                        </p>
                      </div>

                      <div className="p-3 bg-white rounded-xl border border-indigo-100 space-y-1">
                        <div className="text-[11px] font-semibold text-slate-500 uppercase tracking-wide">
                          Parametri da Addestrare (Head)
                        </div>
                        <div className="text-base font-bold text-emerald-700">
                          5,644,480 <span className="text-xs font-normal text-slate-500">(27.7%)</span>
                        </div>
                        <p className="text-[11px] text-slate-600 leading-snug">
                          La YOLO Detect Head: si addestrerà per riconoscere le classi del tuo dataset e calibrare i bounding box.
                        </p>
                      </div>
                    </div>

                    <div className="bg-indigo-100/60 p-2.5 rounded-xl text-[11px] text-indigo-950 flex items-start gap-2">
                      <span className="text-sm">🎮</span>
                      <div className="leading-relaxed">
                        <strong>Ora tocca a te sperimentare:</strong> Imposta <code className="font-mono bg-white/90 px-1 py-0.5 rounded text-indigo-900 font-bold">freeze: 1</code> e gioca liberamente con gli altri iperparametri (numero di epoche, learning rate iniziale <code className="font-mono">lr0</code>, batch size, ottimizzatore) su Ultralytics Platform per trovare la configurazione migliore!
                      </div>
                    </div>
                  </div>

                  {/* Ultralytics Training Instructions */}
                  <div className="border border-slate-200 rounded-2xl overflow-hidden bg-slate-900 text-slate-100">
                    <div className="px-4 py-2.5 bg-slate-800 border-b border-slate-700 flex items-center justify-between flex-wrap gap-2">
                      <div className="flex items-center gap-2">
                        <Terminal className="h-4 w-4 text-emerald-400" />
                        <span className="text-xs font-bold text-slate-200">
                          Come avviare l'addestramento su Ultralytics
                        </span>
                      </div>

                      {/* Tab Selector */}
                      <div className="flex items-center gap-1 bg-slate-900 p-0.5 rounded-lg border border-slate-700 text-xs">
                        <button
                          onClick={() => setActivePlatformTab("cloud")}
                          className={`px-2.5 py-1 rounded-md text-[11px] font-medium transition-colors cursor-pointer ${
                            activePlatformTab === "cloud"
                              ? "bg-indigo-600 text-white"
                              : "text-slate-400 hover:text-white"
                          }`}
                        >
                          Ultralytics Platform (Cloud)
                        </button>
                        <button
                          onClick={() => setActivePlatformTab("local")}
                          className={`px-2.5 py-1 rounded-md text-[11px] font-medium transition-colors cursor-pointer ${
                            activePlatformTab === "local"
                              ? "bg-indigo-600 text-white"
                              : "text-slate-400 hover:text-white"
                          }`}
                        >
                          Test Locale (coco8)
                        </button>
                      </div>
                    </div>

                    <div className="p-4 space-y-3">
                      {activePlatformTab === "cloud" ? (
                        <>
                          <div className="flex items-center justify-between text-xs text-slate-400 pb-1 border-b border-slate-800">
                            <span>Scegli il tuo sistema operativo:</span>
                            <div className="flex items-center gap-1">
                              {(["bash", "powershell", "cmd"] as const).map((os) => (
                                <button
                                  key={os}
                                  onClick={() => setActiveOs(os)}
                                  className={`px-2 py-0.5 rounded text-[10px] font-mono uppercase cursor-pointer ${
                                    activeOs === os
                                      ? "bg-indigo-500/30 text-indigo-300 border border-indigo-400/30"
                                      : "text-slate-500 hover:text-slate-300"
                                  }`}
                                >
                                  {os === "bash" ? "Linux / Mac" : os === "powershell" ? "Win PowerShell" : "Win CMD"}
                                </button>
                              ))}
                            </div>
                          </div>

                          <div className="relative group bg-slate-950 p-3 rounded-xl border border-slate-800 font-mono text-xs leading-relaxed text-emerald-400">
                            {activeOs === "bash" && (
                              <pre className="overflow-x-auto whitespace-pre-wrap">
                                {`export ULTRALYTICS_API_KEY="INSERISCI_LA_TUA_API_KEY"\nyolo train model=vgg16_detector.pt data=ul://tuo-dataset task=detect imgsz=640 freeze=1 project=tuo-progetto`}
                              </pre>
                            )}
                            {activeOs === "powershell" && (
                              <pre className="overflow-x-auto whitespace-pre-wrap">
                                {`$env:ULTRALYTICS_API_KEY="INSERISCI_LA_TUA_API_KEY"\nyolo train model=vgg16_detector.pt data=ul://tuo-dataset task=detect imgsz=640 freeze=1 project=tuo-progetto`}
                              </pre>
                            )}
                            {activeOs === "cmd" && (
                              <pre className="overflow-x-auto whitespace-pre-wrap">
                                {`set ULTRALYTICS_API_KEY=INSERISCI_LA_TUA_API_KEY\nyolo train model=vgg16_detector.pt data=ul://tuo-dataset task=detect imgsz=640 freeze=1 project=tuo-progetto`}
                              </pre>
                            )}

                            <button
                              onClick={() => {
                                const cmd =
                                  activeOs === "bash"
                                    ? `export ULTRALYTICS_API_KEY="INSERISCI_LA_TUA_API_KEY"\nyolo train model=vgg16_detector.pt data=ul://tuo-dataset task=detect imgsz=640 freeze=1 project=tuo-progetto`
                                    : activeOs === "powershell"
                                    ? `$env:ULTRALYTICS_API_KEY="INSERISCI_LA_TUA_API_KEY"\nyolo train model=vgg16_detector.pt data=ul://tuo-dataset task=detect imgsz=640 freeze=1 project=tuo-progetto`
                                    : `set ULTRALYTICS_API_KEY=INSERISCI_LA_TUA_API_KEY\nyolo train model=vgg16_detector.pt data=ul://tuo-dataset task=detect imgsz=640 freeze=1 project=tuo-progetto`;
                                handleCopy(cmd, "cloud");
                              }}
                              className="absolute top-2 right-2 p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 transition-colors cursor-pointer"
                              title="Copia comandi"
                            >
                              {copiedKey === "cloud" ? (
                                <Check className="h-3.5 w-3.5 text-emerald-400" />
                              ) : (
                                <Copy className="h-3.5 w-3.5" />
                              )}
                            </button>
                          </div>
                        </>
                      ) : (
                        <div className="relative group bg-slate-950 p-3 rounded-xl border border-slate-800 font-mono text-xs leading-relaxed text-emerald-400">
                          <pre className="overflow-x-auto whitespace-pre-wrap">
                            {`yolo train model=vgg16_detector.pt data=coco8.yaml task=detect imgsz=640 freeze=1 epochs=15`}
                          </pre>
                          <button
                            onClick={() =>
                              handleCopy(
                                `yolo train model=vgg16_detector.pt data=coco8.yaml task=detect imgsz=640 freeze=1 epochs=15`,
                                "local"
                              )
                            }
                            className="absolute top-2 right-2 p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 transition-colors cursor-pointer"
                            title="Copia comando"
                          >
                            {copiedKey === "local" ? (
                              <Check className="h-3.5 w-3.5 text-emerald-400" />
                            ) : (
                              <Copy className="h-3.5 w-3.5" />
                            )}
                          </button>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Collapsible Log Viewer */}
                  {conversionLog && (
                    <div className="border border-slate-200 rounded-xl overflow-hidden text-xs">
                      <button
                        onClick={() => setShowTerminalLog(!showTerminalLog)}
                        className="w-full px-3.5 py-2 bg-slate-50 hover:bg-slate-100 flex items-center justify-between text-slate-600 font-medium cursor-pointer"
                      >
                        <span className="flex items-center gap-1.5">
                          <Terminal className="h-3.5 w-3.5" />
                          <span>Dettagli tecnici di conversione (Log terminale)</span>
                        </span>
                        {showTerminalLog ? (
                          <ChevronUp className="h-4 w-4" />
                        ) : (
                          <ChevronDown className="h-4 w-4" />
                        )}
                      </button>
                      {showTerminalLog && (
                        <div className="p-3 bg-slate-950 text-slate-300 font-mono text-[11px] max-h-48 overflow-y-auto whitespace-pre-wrap leading-relaxed">
                          {conversionLog}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Footer Actions */}
        <DialogFooter className="p-4 bg-slate-50/90 border-t border-slate-100 flex items-center justify-between sm:justify-end gap-2">
          {!isFinalStep ? (
            isSuccess ? (
              <Button
                onClick={onProceed}
                className="bg-emerald-600 hover:bg-emerald-700 text-white font-medium flex items-center gap-2 cursor-pointer shadow-sm ml-auto"
              >
                Passa al Livello Successivo
                <ArrowRight className="h-4 w-4" />
              </Button>
            ) : (
              <Button
                onClick={() => onOpenChange(false)}
                className="bg-slate-800 hover:bg-slate-900 text-white font-medium cursor-pointer ml-auto"
              >
                Ho capito, correggo!
              </Button>
            )
          ) : (
            <div className="w-full flex items-center justify-between">
              <span className="text-xs text-slate-500 hidden sm:inline">
                Piattaforma didattica VGG-16 • Ultralytics YOLO
              </span>
              <Button
                onClick={() => onOpenChange(false)}
                className="bg-slate-800 hover:bg-slate-900 text-white font-medium cursor-pointer ml-auto"
              >
                Chiudi Finestra
              </Button>
            </div>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
