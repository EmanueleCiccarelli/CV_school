import { useCallback, useEffect, useState } from "react";
import CodeMirror from "@uiw/react-codemirror";
import { python } from "@codemirror/lang-python";
import { Check, Copy, Download } from "lucide-react";

import { Button } from "./ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { parseGraphToDAG } from "../lib/dag-parser";
import { generatePyTorchCode } from "../lib/code-generation";
import { useFlowStore } from "../lib/flow-store";
import { cn } from "../lib/utils";

const UI_CONFIG = {
  COPY_TIMEOUT: 2000,
  BUTTON_HEIGHT: "h-9",
  BORDER_RADIUS: "rounded-lg",
  SPACING: {
    CARD: "px-4 sm:px-6",
    PADDING: "p-4 sm:p-6",
  },
} as const;

function fallbackCopyToClipboard(text: string): void {
  const textArea = document.createElement("textarea");
  textArea.value = text;
  document.body.appendChild(textArea);
  textArea.select();
  document.execCommand("copy");
  document.body.removeChild(textArea);
}

function PyTorchBadge() {
  return (
    <span className="inline-flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-full font-semibold border bg-orange-50 text-orange-700 border-orange-200 shadow-xs">
      <span className="inline-block w-2 h-2 rounded-full bg-orange-500 animate-pulse"></span>
      PyTorch nn.Module
    </span>
  );
}

interface ActionButtonsProps {
  onDownload: () => void;
  onCopy: () => void;
  isDisabled: boolean;
  isCopied: boolean;
}

function ActionButtons({
  onDownload,
  onCopy,
  isDisabled,
  isCopied,
}: ActionButtonsProps) {
  const baseButtonClass = cn(
    UI_CONFIG.BUTTON_HEIGHT,
    "px-4",
    UI_CONFIG.BORDER_RADIUS,
    "transition-all duration-200 shadow-sm"
  );

  return (
    <div className="flex items-center justify-between mt-3 pt-3 border-t border-slate-100">
      <div className="flex items-center gap-2">
        <Button
          variant="outline"
          size="sm"
          onClick={onDownload}
          disabled={isDisabled}
          className={cn(
            baseButtonClass,
            "border-slate-200 hover:bg-slate-50 hover:border-slate-300 hover:shadow-md cursor-pointer"
          )}
        >
          <Download className="h-4 w-4 mr-2" />
          Scarica .py
        </Button>
        <Button
          variant="outline"
          size="sm"
          onClick={onCopy}
          disabled={isDisabled}
          className={cn(
            baseButtonClass,
            isCopied
              ? "border-green-300 bg-green-50 text-green-700 hover:bg-green-100 shadow-md"
              : "border-slate-200 hover:bg-slate-50 hover:border-slate-300 hover:shadow-md cursor-pointer"
          )}
        >
          {isCopied ? (
            <>
              <Check className="h-4 w-4 mr-2" />
              Copiato!
            </>
          ) : (
            <>
              <Copy className="h-4 w-4 mr-2" />
              Copia Codice
            </>
          )}
        </Button>
      </div>
    </div>
  );
}

interface CodeEditorProps {
  code: string;
}

function CodeEditor({ code }: CodeEditorProps) {
  return (
    <div className="rounded-xl border border-slate-200 shadow-inner bg-slate-50/30 flex-1 min-h-0">
      <div className="w-full h-full overflow-auto">
        <CodeMirror
          value={code}
          height="100%"
          extensions={[python()]}
          editable={false}
          basicSetup={{
            lineNumbers: true,
            foldGutter: true,
            dropCursor: false,
            allowMultipleSelections: false,
            indentOnInput: false,
            bracketMatching: true,
            closeBrackets: false,
            autocompletion: false,
            highlightSelectionMatches: false,
            searchKeymap: false,
          }}
        />
      </div>
    </div>
  );
}

interface CodeViewerProps {
  className?: string;
}

// Generates PyTorch code from visual neural network graph
export function CodeViewer({ className = "" }: CodeViewerProps) {
  const { nodes, edges } = useFlowStore();
  const [generatedCode, setGeneratedCode] = useState("");
  const [isCopied, setIsCopied] = useState(false);

  const resetCopyState = useCallback(() => {
    setTimeout(() => setIsCopied(false), UI_CONFIG.COPY_TIMEOUT);
  }, []);

  // Auto-generate PyTorch code when graph changes
  useEffect(() => {
    if (nodes.length === 0) {
      setGeneratedCode(
        "# Canvas vuoto\n# Trascina e collega i blocchi sul canvas per generare la rete PyTorch"
      );
      return;
    }

    const dagResult = parseGraphToDAG(nodes, edges);

    if (!dagResult.isValid) {
      setGeneratedCode(
        `# Struttura rete non ancora completa o non valida:\n# ${dagResult.errors.join(
          "\n# "
        )}`
      );
      return;
    }

    try {
      const code = generatePyTorchCode(dagResult.orderedNodes);
      setGeneratedCode(code);
    } catch (err) {
      setGeneratedCode(
        `# Errore durante la generazione PyTorch: ${
          err instanceof Error ? err.message : String(err)
        }`
      );
    }
  }, [nodes, edges]);

  const handleCopyCode = useCallback(async () => {
    if (!generatedCode.trim()) return;

    try {
      await navigator.clipboard.writeText(generatedCode);
      setIsCopied(true);
      resetCopyState();
    } catch {
      // Fallback for older browsers
      fallbackCopyToClipboard(generatedCode);
      setIsCopied(true);
      resetCopyState();
    }
  }, [generatedCode, resetCopyState]);

  const handleDownloadCode = useCallback(() => {
    if (!generatedCode.trim()) return;

    const blob = new Blob([generatedCode], { type: "text/python" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");

    link.href = url;
    link.download = `vgg16_pytorch_model.py`;
    document.body.appendChild(link);
    link.click();

    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }, [generatedCode]);

  return (
    <div className={cn("h-full flex flex-col bg-slate-50/80", className)}>
      <div
        className={cn(
          "flex-1 flex flex-col min-h-0",
          UI_CONFIG.SPACING.PADDING
        )}
      >
        <Card className="border-slate-200 bg-white shadow-sm rounded-xl flex-1 flex flex-col min-h-0">
          <CardHeader className={cn("flex-shrink-0", UI_CONFIG.SPACING.CARD)}>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="flex items-center gap-2">
                  <span className="text-xl">🔥</span>
                  <CardTitle className="text-lg text-slate-800 font-semibold">
                    Codice PyTorch
                  </CardTitle>
                </div>
                <PyTorchBadge />
              </div>
            </div>
            <ActionButtons
              onDownload={handleDownloadCode}
              onCopy={handleCopyCode}
              isDisabled={!generatedCode.trim()}
              isCopied={isCopied}
            />
          </CardHeader>
          <CardContent
            className={cn(
              "pt-0 flex-1 flex flex-col min-h-0",
              UI_CONFIG.SPACING.CARD
            )}
          >
            <CodeEditor code={generatedCode} />
          </CardContent>
          <div className="px-2 pb-2 text-center">
            <div className="text-xs text-slate-500 font-medium">
              VGG16 Progressive Sandbox • PyTorch Engine
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
}
