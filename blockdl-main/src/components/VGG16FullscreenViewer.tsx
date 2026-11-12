import { useState, useEffect, useRef } from "react";
import {
  X,
  ZoomIn,
  ZoomOut,
  Box,
  Maximize2,
  Eye,
  EyeOff,
  RotateCcw,
} from "lucide-react";
import { Button } from "./ui/button";
import type { VGG16Level } from "../lib/vgg16LevelsConfig";
import blocksRefinedImg from "../vgg image assets/vgg16_blocks_refined.jpeg";
import featuresMapsImg from "../vgg image assets/vgg16_features_maps.webp";

interface VGG16FullscreenViewerProps {
  open: boolean;
  onClose: () => void;
  level: VGG16Level;
  initialView?: "blocks" | "features";
}

export function VGG16FullscreenViewer({
  open,
  onClose,
  level,
  initialView = "blocks",
}: VGG16FullscreenViewerProps) {
  const [activeView, setActiveView] = useState<"blocks" | "features">(initialView);
  const [zoomLevel, setZoomLevel] = useState<number>(1);
  const [showBoundingBox, setShowBoundingBox] = useState<boolean>(true);
  const containerRef = useRef<HTMLDivElement>(null);

  // Sync initial view when opening
  useEffect(() => {
    if (open) {
      setActiveView(initialView);
      setZoomLevel(1);
    }
  }, [open, initialView]);

  // Handle ESC key to close viewer
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape" && open) {
        onClose();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [open, onClose]);

  if (!open) return null;

  const currentBox =
    activeView === "blocks"
      ? level.boxes.blocksImage
      : level.boxes.featuresImage;

  const handleZoomIn = () => {
    setZoomLevel((prev) => Math.min(prev + 0.25, 3));
  };

  const handleZoomOut = () => {
    setZoomLevel((prev) => Math.max(prev - 0.25, 0.5));
  };

  const handleResetZoom = () => {
    setZoomLevel(1);
  };

  return (
    <div className="fixed inset-0 z-[100] bg-slate-950/95 backdrop-blur-md flex flex-col text-white animate-in fade-in duration-200">
      {/* Top Header & Controls Bar */}
      <div className="flex-shrink-0 px-6 py-3 bg-slate-900/90 border-b border-slate-800 flex items-center justify-between gap-4 shadow-md">
        {/* Title and Level Badge */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-amber-400 animate-pulse"></span>
            <span className="font-bold text-sm tracking-wide text-white">
              VGG16 Blueprint Schermo Intero:
            </span>
          </div>
          <span className="text-xs px-3 py-1 rounded-full font-semibold bg-indigo-600/30 text-indigo-300 border border-indigo-500/40">
            {level.title}
          </span>
        </div>

        {/* View Switcher & Zoom Controls */}
        <div className="flex items-center gap-3">
          {/* Switcher */}
          <div className="flex bg-slate-800 rounded-lg p-0.5 border border-slate-700 text-xs">
            <button
              onClick={() => setActiveView("blocks")}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md font-medium transition-all cursor-pointer ${
                activeView === "blocks"
                  ? "bg-indigo-600 text-white shadow-xs font-semibold"
                  : "text-slate-300 hover:text-white"
              }`}
            >
              <Box className="h-3.5 w-3.5" />
              <span>1. Schema Blocchi</span>
            </button>
            <button
              onClick={() => setActiveView("features")}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md font-medium transition-all cursor-pointer ${
                activeView === "features"
                  ? "bg-indigo-600 text-white shadow-xs font-semibold"
                  : "text-slate-300 hover:text-white"
              }`}
            >
              <Maximize2 className="h-3.5 w-3.5" />
              <span>2. Mappa Tensori 3D</span>
            </button>
          </div>

          {/* Bounding Box Toggle Button */}
          <button
            onClick={() => setShowBoundingBox((prev) => !prev)}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold border transition-all cursor-pointer ${
              showBoundingBox
                ? "bg-amber-500/20 text-amber-300 border-amber-500/50 hover:bg-amber-500/30"
                : "bg-slate-800 text-slate-400 border-slate-700 hover:text-slate-200 hover:bg-slate-700"
            }`}
            title={
              showBoundingBox
                ? "Nascondi l'evidenziazione della Bounding Box"
                : "Mostra l'evidenziazione della Bounding Box"
            }
          >
            {showBoundingBox ? (
              <>
                <Eye className="h-3.5 w-3.5 text-amber-400" />
                <span>Bounding Box: ON</span>
              </>
            ) : (
              <>
                <EyeOff className="h-3.5 w-3.5 text-slate-400" />
                <span>Bounding Box: OFF</span>
              </>
            )}
          </button>

          {/* Zoom Buttons */}
          <div className="flex items-center gap-1 bg-slate-800 rounded-lg p-1 border border-slate-700">
            <Button
              variant="ghost"
              size="sm"
              onClick={handleZoomOut}
              disabled={zoomLevel <= 0.5}
              className="h-7 w-7 p-0 text-slate-300 hover:text-white hover:bg-slate-700 cursor-pointer"
              title="Riduci zoom"
            >
              <ZoomOut className="h-4 w-4" />
            </Button>
            <span className="text-xs font-mono font-bold px-2 text-slate-200 select-none min-w-[50px] text-center">
              {Math.round(zoomLevel * 100)}%
            </span>
            <Button
              variant="ghost"
              size="sm"
              onClick={handleZoomIn}
              disabled={zoomLevel >= 3}
              className="h-7 w-7 p-0 text-slate-300 hover:text-white hover:bg-slate-700 cursor-pointer"
              title="Aumenta zoom"
            >
              <ZoomIn className="h-4 w-4" />
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={handleResetZoom}
              className="h-7 px-2 text-[11px] text-slate-300 hover:text-white hover:bg-slate-700 cursor-pointer"
              title="Adatta allo schermo"
            >
              <RotateCcw className="h-3 w-3 mr-1" />
              Adatta
            </Button>
          </div>

          {/* Close Fullscreen Button */}
          <Button
            variant="ghost"
            size="sm"
            onClick={onClose}
            className="h-8 px-3 rounded-lg bg-rose-600/20 text-rose-300 hover:bg-rose-600 hover:text-white border border-rose-500/30 flex items-center gap-1.5 text-xs font-semibold cursor-pointer"
            title="Chiudi schermo intero (ESC)"
          >
            <X className="h-4 w-4" />
            <span>Chiudi (ESC)</span>
          </Button>
        </div>
      </div>

      {/* Main Image Viewport Area (with smooth scroll & pan) */}
      <div
        ref={containerRef}
        className="flex-1 overflow-auto p-6 flex items-center justify-center select-none"
      >
        <div
          className="relative inline-block transition-transform duration-150 ease-out shadow-2xl rounded-xl border-4 border-slate-700 bg-white"
          style={{
            transform: `scale(${zoomLevel})`,
            transformOrigin: "center center",
            maxWidth: zoomLevel === 1 ? "95vw" : undefined,
            maxHeight: zoomLevel === 1 ? "82vh" : undefined,
          }}
        >
          <img
            src={activeView === "blocks" ? blocksRefinedImg : featuresMapsImg}
            alt="Reference VGG16 Fullscreen"
            className="block h-auto rounded-lg"
            style={{
              maxHeight: zoomLevel === 1 ? "82vh" : undefined,
              maxWidth: zoomLevel === 1 ? "95vw" : undefined,
            }}
          />

          {/* Dynamic Bounding Box Overlay with Glowing Border */}
          {showBoundingBox && (
            <div
              className="absolute border-4 border-amber-500 bg-amber-400/25 rounded-lg shadow-2xl pointer-events-none transition-all duration-200 ring-4 ring-amber-400/30"
              style={{
                left: `${currentBox.left}%`,
                top: `${currentBox.top}%`,
                width: `${currentBox.width}%`,
                height: `${currentBox.height}%`,
              }}
            >
              {/* Label Badge: placed at bottom for features so it doesn't cover top dimensions, top for blocks */}
              <div
                className={`absolute ${
                  activeView === "features" ? "-bottom-9" : "-top-9"
                } left-1/2 -translate-x-1/2 bg-amber-600 text-white text-xs font-bold px-3 py-1 rounded-md shadow-lg whitespace-nowrap flex items-center gap-1.5 border border-amber-300/40`}
              >
                <Eye className="h-3.5 w-3.5 animate-pulse" />
                <span>PORZIONE DA COSTRUIRE: {level.blockName}</span>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Bottom Informative Bar */}
      <div className="flex-shrink-0 px-6 py-2.5 bg-slate-900/80 border-t border-slate-800/80 text-xs text-slate-400 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-amber-400 font-semibold">💡 Suggerimento didattico:</span>
          <span>
            {activeView === "blocks"
              ? "Nel diagramma a blocchi: le caselle verdi indicano le convoluzioni (nota la sigla Conv), quelle rosse il Pooling, quelle blu i layer Fully Connected (Dense)."
              : "Nella mappa tensoriale: osserva le dimensioni numeriche dei blocchi per ricavare la risoluzione spaziale e il numero di canali."}
          </span>
        </div>
        <div className="flex items-center gap-3 text-slate-400 text-[11px]">
          <span>Usa i pulsanti di zoom per ingrandire i dettagli</span>
          <button
            onClick={() => setZoomLevel(zoomLevel === 1 ? 1.75 : 1)}
            className="text-indigo-400 hover:text-indigo-300 underline cursor-pointer"
          >
            {zoomLevel === 1 ? "Visualizza a 175%" : "Ripristina 100%"}
          </button>
        </div>
      </div>
    </div>
  );
}
