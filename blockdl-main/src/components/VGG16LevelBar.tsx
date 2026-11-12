import {
  CheckCircle2,
  Lock,
  Play,
  RotateCcw,
  BookOpen,
  Sparkles,
  Rocket,
} from "lucide-react";
import { Button } from "./ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "./ui/dialog";
import type { VGG16Level } from "../lib/vgg16LevelsConfig";

interface VGG16LevelBarProps {
  levels: VGG16Level[];
  currentLevelId: number;
  unlockedLevel: number;
  onSelectLevel: (levelId: number) => void;
  onValidate: () => void;
  onReset: () => void;
  onOpenGuide: () => void;
  showResetConfirm: boolean;
  setShowResetConfirm: (open: boolean) => void;
}

export function VGG16LevelBar({
  levels,
  currentLevelId,
  unlockedLevel,
  onSelectLevel,
  onValidate,
  onReset,
  onOpenGuide,
  showResetConfirm,
  setShowResetConfirm,
}: VGG16LevelBarProps) {
  const currentLevel = levels.find((l) => l.id === currentLevelId) || levels[0];

  return (
    <>
      <div className="bg-white border-b border-slate-200 px-6 py-2.5 flex items-center justify-between gap-4 shadow-2xs">
        {/* Left: Level Stepper */}
        <div className="flex items-center gap-1.5 flex-wrap">
          <div className="flex items-center gap-2 mr-2">
            <span className="w-2 h-2 rounded-full bg-indigo-600 animate-pulse"></span>
            <span className="text-xs font-bold uppercase tracking-wider text-slate-500">
              VGG16 Sandbox:
            </span>
          </div>

          <div className="flex items-center gap-1 bg-slate-100 p-1 rounded-xl border border-slate-200/80">
            {levels.map((lvl) => {
              const isActive = lvl.id === currentLevelId;
              const isUnlocked = lvl.id <= unlockedLevel;
              const isCompleted = lvl.id < unlockedLevel;

              return (
                <button
                  key={lvl.id}
                  onClick={() => isUnlocked && onSelectLevel(lvl.id)}
                  disabled={!isUnlocked}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all cursor-pointer ${
                    isActive
                      ? "bg-white text-indigo-700 shadow-sm font-semibold border border-indigo-200"
                      : isCompleted
                      ? "text-emerald-700 hover:bg-white/60"
                      : isUnlocked
                      ? "text-slate-700 hover:bg-white/60"
                      : "text-slate-400 opacity-60 cursor-not-allowed"
                  }`}
                  title={
                    !isUnlocked
                      ? `Livello ${lvl.id} bloccato. Completa prima il livello ${lvl.id - 1}!`
                      : lvl.title
                  }
                >
                  {isCompleted && !isActive ? (
                    <CheckCircle2 className="h-3.5 w-3.5 text-emerald-600" />
                  ) : !isUnlocked ? (
                    <Lock className="h-3.5 w-3.5 text-slate-400" />
                  ) : (
                    <span
                      className={`w-4 h-4 rounded-full flex items-center justify-center text-[10px] font-bold ${
                        isActive
                          ? "bg-indigo-600 text-white"
                          : "bg-slate-200 text-slate-700"
                      }`}
                    >
                      {lvl.id}
                    </span>
                  )}
                  <span>{lvl.blockName}</span>
                </button>
              );
            })}
          </div>
        </div>

        {/* Center / Right: Active Objective & Action Buttons */}
        <div className="flex items-center gap-2">
          {/* Guide & Blueprint */}
          <Button
            variant="outline"
            size="sm"
            onClick={onOpenGuide}
            className="text-xs flex items-center gap-1.5 h-8 border-slate-200 hover:bg-indigo-50/70 hover:text-indigo-700 hover:border-indigo-200 cursor-pointer"
          >
            <BookOpen className="h-3.5 w-3.5 text-indigo-600" />
            <span>Guida & Blueprint</span>
          </Button>

          {/* Reset Current Level Step (Single Block) */}
          <Button
            variant="outline"
            size="sm"
            onClick={() => setShowResetConfirm(true)}
            className="text-xs flex items-center gap-1.5 h-8 border-slate-200 text-slate-600 hover:bg-slate-100 cursor-pointer"
            title="Ripristina il canvas allo stato iniziale del livello corrente"
          >
            <RotateCcw className="h-3.5 w-3.5" />
            <span>Reset Step</span>
          </Button>

          {/* Primary Action: Verify Level */}
          <Button
            size="sm"
            onClick={onValidate}
            className="text-xs font-semibold flex items-center gap-2 h-8 px-4 bg-gradient-to-r from-indigo-600 to-blue-600 hover:from-indigo-700 hover:to-blue-700 text-white shadow-xs cursor-pointer ml-1"
          >
            <Play className="h-3.5 w-3.5 fill-current" />
            <span>Verifica Livello</span>
            <Sparkles className="h-3 w-3 text-amber-300" />
          </Button>

          {/* Quick Export Detector button if level 6 is reached */}
          {unlockedLevel >= 6 && (
            <Button
              size="sm"
              onClick={onValidate}
              className="text-xs font-semibold flex items-center gap-1.5 h-8 px-3 bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-700 hover:to-teal-700 text-white shadow-xs cursor-pointer ml-1"
              title="Apri pannello esportazione Object Detector per Ultralytics"
            >
              <Rocket className="h-3.5 w-3.5 text-amber-300" />
              <span>Esporta (.pt)</span>
            </Button>
          )}
        </div>
      </div>

      {/* Reset Current Step Dialog */}
      <Dialog open={showResetConfirm} onOpenChange={setShowResetConfirm}>
        <DialogContent className="max-w-md rounded-2xl">
          <DialogHeader>
            <DialogTitle className="text-lg font-bold text-slate-800 flex items-center gap-2">
              <RotateCcw className="h-5 w-5 text-rose-500" />
              <span>Ripristinare il {currentLevel.blockName}?</span>
            </DialogTitle>
            <DialogDescription className="text-sm text-slate-600 pt-1 leading-relaxed">
              Questa operazione cancellerà le modifiche correnti sul canvas per questo livello e ricaricherà la configurazione iniziale corretta del <strong>{currentLevel.title}</strong>.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter className="gap-2 sm:gap-0 pt-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setShowResetConfirm(false)}
            >
              Annulla
            </Button>
            <Button
              variant="destructive"
              size="sm"
              onClick={onReset}
              className="bg-rose-600 hover:bg-rose-700 text-white"
            >
              Sì, ripristina blocco
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
