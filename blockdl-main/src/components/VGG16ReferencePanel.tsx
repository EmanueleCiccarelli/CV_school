import { useState } from "react";
import {
  BookOpen,
  Sparkles,
  Eye,
  EyeOff,
  Maximize2,
  Box,
  Maximize,
} from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "./ui/dialog";
import { Button } from "./ui/button";
import type { VGG16Level } from "../lib/vgg16LevelsConfig";

// Import real VGG image assets
import blocksRefinedImg from "../vgg image assets/vgg16_blocks_refined.jpeg";
import featuresMapsImg from "../vgg image assets/vgg16_features_maps.webp";

interface VGG16ReferencePanelProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  level: VGG16Level;
  unlockedLevel?: number;
  onOpenFullscreen?: (view?: "blocks" | "features") => void;
}

export function VGG16ReferencePanel({
  open,
  onOpenChange,
  level,
  onOpenFullscreen,
}: VGG16ReferencePanelProps) {
  // Toggle between block diagram and 3D feature maps
  const [activeView, setActiveView] = useState<"blocks" | "features">("blocks");
  const [showBoundingBox, setShowBoundingBox] = useState<boolean>(true);

  const currentBox =
    activeView === "blocks"
      ? level.boxes.blocksImage
      : level.boxes.featuresImage;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-4xl max-h-[92vh] overflow-y-auto p-6 rounded-2xl">
        <DialogHeader className="space-y-1">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-indigo-600 font-semibold text-xs uppercase tracking-wider">
              <BookOpen className="h-4 w-4" />
              <span>Blueprint Didattico Architettura VGG16</span>
            </div>
            <span className="text-xs px-2.5 py-1 rounded-full font-semibold bg-indigo-50 text-indigo-700 border border-indigo-200">
              {level.badge}
            </span>
          </div>
          <DialogTitle className="text-xl font-bold text-slate-800">
            {level.title}
          </DialogTitle>
          <DialogDescription className="text-xs text-slate-500">
            Consulta la reference visiva qui sotto: la porzione della VGG16 da costruire in questo livello è evidenziata con una bounding box colorata. Clicca sull'immagine o sul pulsante <strong>"Ingrandisci a Schermo Intero"</strong> per esaminarla nel dettaglio con zoom e panoramica.
          </DialogDescription>
        </DialogHeader>

        {/* View Switcher & Fullscreen Action Bar */}
        <div className="mt-3 flex items-center justify-between bg-slate-100 p-1.5 rounded-xl border border-slate-200 flex-wrap gap-2">
          <div className="flex items-center gap-1">
            <button
              onClick={() => setActiveView("blocks")}
              className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium transition-all cursor-pointer ${
                activeView === "blocks"
                  ? "bg-white text-slate-900 shadow-sm font-semibold"
                  : "text-slate-600 hover:text-slate-900"
              }`}
            >
              <Box className="h-3.5 w-3.5 text-indigo-600" />
              <span>1. Schema a Blocchi</span>
            </button>
            <button
              onClick={() => setActiveView("features")}
              className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium transition-all cursor-pointer ${
                activeView === "features"
                  ? "bg-white text-slate-900 shadow-sm font-semibold"
                  : "text-slate-600 hover:text-slate-900"
              }`}
            >
              <Maximize2 className="h-3.5 w-3.5 text-blue-600" />
              <span>2. Mappa Tensori 3D</span>
            </button>
          </div>

          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => setShowBoundingBox((prev) => !prev)}
              className={`flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-semibold border transition-all cursor-pointer ${
                showBoundingBox
                  ? "bg-amber-50 border-amber-300 text-amber-800 hover:bg-amber-100"
                  : "bg-slate-100 border-slate-300 text-slate-500 hover:bg-slate-200 hover:text-slate-700"
              }`}
              title={
                showBoundingBox
                  ? "Nascondi l'evidenziazione della Bounding Box"
                  : "Mostra l'evidenziazione della Bounding Box"
              }
            >
              {showBoundingBox ? (
                <>
                  <Eye className="h-3.5 w-3.5 text-amber-600" />
                  <span>Bounding Box: ON</span>
                </>
              ) : (
                <>
                  <EyeOff className="h-3.5 w-3.5 text-slate-400" />
                  <span>Bounding Box: OFF</span>
                </>
              )}
            </button>
            <Button
              size="sm"
              onClick={() => onOpenFullscreen?.(activeView)}
              className="h-7 text-xs flex items-center gap-1.5 bg-indigo-600 hover:bg-indigo-700 text-white font-semibold cursor-pointer shadow-xs transition-colors"
              title="Apri il diagramma a tutto schermo con controlli di zoom e panoramica"
            >
              <Maximize className="h-3.5 w-3.5" />
              <span>Ingrandisci a Schermo Intero</span>
            </Button>
          </div>
        </div>

        {/* Visual Reference with Dynamic Bounding Box */}
        <div
          onClick={() => onOpenFullscreen?.(activeView)}
          className="mt-3 relative rounded-xl border-2 border-slate-200 bg-white shadow-sm overflow-hidden p-2 group cursor-pointer hover:border-indigo-400 transition-colors"
          title="Clicca per visualizzare a tutto schermo"
        >
          <div className="relative inline-block w-full">
            <img
              src={activeView === "blocks" ? blocksRefinedImg : featuresMapsImg}
              alt={`Reference VGG16 ${activeView}`}
              className="w-full h-auto block select-none rounded-lg"
            />

            {/* Bounding Box Overlay */}
            {showBoundingBox && (
              <div
                className="absolute border-3 border-amber-500 bg-amber-400/25 rounded-md shadow-lg pointer-events-none transition-all duration-300"
                style={{
                  left: `${currentBox.left}%`,
                  top: `${currentBox.top}%`,
                  width: `${currentBox.width}%`,
                  height: `${currentBox.height}%`,
                }}
              >
                {/* Badge Label on Bounding Box: bottom for features to avoid covering dimensions, top for blocks */}
                <div
                  className={`absolute ${
                    activeView === "features" ? "-bottom-7" : "-top-7"
                  } left-1/2 -translate-x-1/2 bg-amber-600 text-white text-[10px] font-bold px-2 py-0.5 rounded shadow-md whitespace-nowrap flex items-center gap-1`}
                >
                  <span className="w-1.5 h-1.5 rounded-full bg-white animate-ping"></span>
                  <span>Obiettivo {level.blockName}</span>
                </div>
              </div>
            )}

            {/* Hover Fullscreen Prompt Overlay */}
            <div className="absolute bottom-3 right-3 bg-slate-900/80 hover:bg-slate-900 text-white text-[11px] font-medium px-2.5 py-1 rounded-md shadow-md backdrop-blur-xs flex items-center gap-1.5 opacity-90 group-hover:opacity-100 transition-opacity">
              <Maximize className="h-3.5 w-3.5 text-indigo-400" />
              <span>Clicca per ingrandire a tutto schermo</span>
            </div>
          </div>

          <div className="mt-2 text-center text-[11px] text-slate-500">
            {activeView === "blocks"
              ? "🟢 Convoluzioni (filtri e kernel) • 🔴 Pooling (dimezzamento risoluzione) • 🔵 Classificatore Dense"
              : "Dimensioni spaziali e canali tensoriali: analizza la profondità e l'evoluzione dei tensori lungo la rete"}
          </div>
        </div>

        {/* Didactic Context */}
        <div className="mt-4 p-4 rounded-xl bg-blue-50/70 border border-blue-200/80 text-slate-800 space-y-2">
          <div className="flex items-center gap-2 font-semibold text-blue-900 text-sm">
            <Sparkles className="h-4 w-4 text-blue-600" />
            <span>Obiettivo Didattico</span>
          </div>
          <p className="text-xs leading-relaxed text-slate-700">
            {level.didacticExplanation}
          </p>
        </div>

        {/* Instructions */}
        <div className="mt-3 p-3 rounded-xl bg-slate-50 border border-slate-200 text-xs text-slate-600 space-y-1">
          <div className="font-semibold text-slate-700">💡 Come procedere:</div>
          <ul className="list-disc list-inside space-y-0.5 text-[11px]">
            <li>Trascina dalla palette a sinistra i blocchi corrispondenti alla sezione evidenziata.</li>
            <li>Fai doppio clic su ciascun blocco per configurarne i parametri in accordo con il blueprint.</li>
            <li>Collega i blocchi con gli archi nell'ordine corretto e premi <strong>"Verifica Livello"</strong> quando hai terminato.</li>
          </ul>
        </div>
      </DialogContent>
    </Dialog>
  );
}
