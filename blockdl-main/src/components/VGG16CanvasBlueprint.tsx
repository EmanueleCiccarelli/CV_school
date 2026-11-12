import { useState } from "react";
import { Eye, EyeOff, BookOpen, Box, Maximize2 } from "lucide-react";
import type { VGG16Level } from "../lib/vgg16LevelsConfig";
import blocksRefinedImg from "../vgg image assets/vgg16_blocks_refined.jpeg";
import featuresMapsImg from "../vgg image assets/vgg16_features_maps.webp";

interface VGG16CanvasBlueprintProps {
  level: VGG16Level;
  onOpenFullModal: () => void;
}

export function VGG16CanvasBlueprint({
  level,
  onOpenFullModal,
}: VGG16CanvasBlueprintProps) {
  const [isExpanded, setIsExpanded] = useState(true);
  const [activeView, setActiveView] = useState<"blocks" | "features">("blocks");

  const currentBox =
    activeView === "blocks"
      ? level.boxes.blocksImage
      : level.boxes.featuresImage;

  if (!isExpanded) {
    return (
      <button
        onClick={() => setIsExpanded(true)}
        className="flex items-center gap-2 bg-white/95 hover:bg-white text-slate-800 px-3 py-1.5 rounded-xl border border-slate-300 shadow-md text-xs font-semibold backdrop-blur-xs transition-all cursor-pointer"
        title="Mostra Blueprint VGG16"
      >
        <Eye className="h-3.5 w-3.5 text-indigo-600" />
        <span>Blueprint: {level.blockName}</span>
      </button>
    );
  }

  return (
    <div className="bg-white/95 backdrop-blur-xs border border-slate-300 rounded-2xl shadow-xl w-96 p-3 space-y-2 text-slate-800 transition-all">
      {/* Widget Header */}
      <div className="flex items-center justify-between border-b border-slate-100 pb-2">
        <div className="flex items-center gap-1.5 text-xs font-bold text-slate-800">
          <span className="w-2 h-2 rounded-full bg-amber-500 animate-pulse"></span>
          <span>Blueprint: {level.blockName}</span>
        </div>
        <div className="flex items-center gap-1">
          <button
            onClick={onOpenFullModal}
            className="p-1 rounded-md text-slate-600 hover:text-indigo-600 hover:bg-slate-100 cursor-pointer flex items-center gap-1 text-[11px] font-medium"
            title="Apri Guida e Blueprint dettagliato"
          >
            <Maximize2 className="h-3.5 w-3.5 text-indigo-600" />
          </button>
          <button
            onClick={() => setIsExpanded(false)}
            className="p-1 rounded-md text-slate-500 hover:text-slate-800 hover:bg-slate-100 cursor-pointer"
            title="Riduci a icona"
          >
            <EyeOff className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>

      {/* View Switcher Pill */}
      <div className="flex bg-slate-100 rounded-lg p-0.5 text-[10px] font-medium text-slate-600">
        <button
          onClick={() => setActiveView("blocks")}
          className={`flex-1 py-1 rounded-md text-center transition-all cursor-pointer ${
            activeView === "blocks"
              ? "bg-white text-indigo-700 shadow-2xs font-semibold"
              : "hover:text-slate-900"
          }`}
        >
          <span className="inline-flex items-center gap-1">
            <Box className="h-3 w-3" />
            Schema Blocchi
          </span>
        </button>
        <button
          onClick={() => setActiveView("features")}
          className={`flex-1 py-1 rounded-md text-center transition-all cursor-pointer ${
            activeView === "features"
              ? "bg-white text-indigo-700 shadow-2xs font-semibold"
              : "hover:text-slate-900"
          }`}
        >
          <span className="inline-flex items-center gap-1">
            <Maximize2 className="h-3 w-3" />
            Mappa 3D
          </span>
        </button>
      </div>

      {/* Image with Bounding Box Overlay */}
      <div
        onClick={onOpenFullModal}
        className="relative rounded-lg overflow-hidden border border-slate-200 bg-white group cursor-pointer hover:border-indigo-400 transition-colors"
        title="Clicca per aprire la Guida & Blueprint"
      >
        <img
          src={activeView === "blocks" ? blocksRefinedImg : featuresMapsImg}
          alt={`Blueprint VGG16 ${activeView}`}
          className="w-full h-auto block select-none"
        />

        {/* Bounding Box */}
        <div
          className="absolute border-2 border-amber-500 bg-amber-400/25 rounded-xs pointer-events-none transition-all duration-200"
          style={{
            left: `${currentBox.left}%`,
            top: `${currentBox.top}%`,
            width: `${currentBox.width}%`,
            height: `${currentBox.height}%`,
          }}
        >
          <div
            className={`absolute ${
              activeView === "features" ? "-bottom-5" : "-top-5"
            } left-1/2 -translate-x-1/2 bg-amber-600 text-white text-[9px] font-bold px-1.5 py-0.2 rounded shadow whitespace-nowrap`}
          >
            {level.blockName}
          </div>
        </div>

        {/* Floating tooltip on hover */}
        <div className="absolute inset-0 bg-slate-950/20 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center pointer-events-none">
          <div className="bg-slate-900/90 text-white text-[10px] font-semibold px-2 py-1 rounded shadow-md flex items-center gap-1.5">
            <BookOpen className="h-3 w-3 text-indigo-400" />
            <span>Apri Guida & Blueprint</span>
          </div>
        </div>
      </div>
    </div>
  );
}
