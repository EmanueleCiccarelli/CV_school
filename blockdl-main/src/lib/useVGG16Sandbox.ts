import { useState, useEffect, useCallback } from "react";
import { VGG16_LEVELS } from "./vgg16LevelsConfig";
import { validateVGG16Level, type ValidationResult } from "./vgg16Validator";
import { useFlowStore } from "./flow-store";
import sandboxVersion from "./sandbox-version.json";

const STORAGE_KEYS = {
  UNLOCKED_LEVEL: "blockdl_vgg16_unlocked_level",
  CURRENT_LEVEL: "blockdl_vgg16_current_level",
  LEVEL_PREFIX: "blockdl_vgg16_level_data_",
  RESET_VERSION: "blockdl_vgg16_reset_version",
} as const;

// Checks if a CLI reset occurred (via `npm run reset-sandbox`)
function checkAndApplyCliReset(): boolean {
  try {
    const savedVer = localStorage.getItem(STORAGE_KEYS.RESET_VERSION);
    const targetVer =
      typeof window !== "undefined" && (window as any).__SANDBOX_RESET_VERSION__
        ? String((window as any).__SANDBOX_RESET_VERSION__)
        : String(sandboxVersion.resetVersion);

    if (savedVer !== targetVer) {
      localStorage.removeItem(STORAGE_KEYS.UNLOCKED_LEVEL);
      localStorage.removeItem(STORAGE_KEYS.CURRENT_LEVEL);
      for (let i = 1; i <= 6; i++) {
        localStorage.removeItem(`${STORAGE_KEYS.LEVEL_PREFIX}${i}`);
      }
      localStorage.removeItem("blockdl-welcome-shown");
      localStorage.setItem(STORAGE_KEYS.RESET_VERSION, targetVer);
      return true;
    }
  } catch (err) {
    console.warn("Could not check reset version:", err);
  }
  return false;
}

export function useVGG16Sandbox() {
  const { setNodes, setEdges, nodes, edges, saveToHistory, clearHistory } = useFlowStore();

  // Run CLI reset check before reading initial state
  const isResetApplied = checkAndApplyCliReset();

  const [currentLevelId, setCurrentLevelId] = useState<number>(() => {
    if (isResetApplied) return 1;
    try {
      const saved = localStorage.getItem(STORAGE_KEYS.CURRENT_LEVEL);
      const parsed = saved ? parseInt(saved, 10) : 1;
      return parsed >= 1 && parsed <= 6 ? parsed : 1;
    } catch {
      return 1;
    }
  });

  const [unlockedLevel, setUnlockedLevel] = useState<number>(() => {
    if (isResetApplied) return 1;
    try {
      const saved = localStorage.getItem(STORAGE_KEYS.UNLOCKED_LEVEL);
      const parsed = saved ? parseInt(saved, 10) : 1;
      return parsed >= 1 && parsed <= 6 ? parsed : 1;
    } catch {
      return 1;
    }
  });

  const [validationResult, setValidationResult] = useState<ValidationResult | null>(null);
  const [showValidationDialog, setShowValidationDialog] = useState(false);
  const [showReferencePanel, setShowReferencePanel] = useState(false);
  const [showResetConfirmDialog, setShowResetConfirmDialog] = useState(false);

  const currentLevel = VGG16_LEVELS.find((l) => l.id === currentLevelId) || VGG16_LEVELS[0];

  // Load level data into the canvas
  const loadLevel = useCallback(
    (levelId: number) => {
      const targetLevel = VGG16_LEVELS.find((l) => l.id === levelId);
      if (!targetLevel) return;

      try {
        const savedDataStr = localStorage.getItem(`${STORAGE_KEYS.LEVEL_PREFIX}${levelId}`);
        if (savedDataStr) {
          const parsed = JSON.parse(savedDataStr);
          if (Array.isArray(parsed.nodes) && Array.isArray(parsed.edges)) {
            setNodes(parsed.nodes);
            setEdges(parsed.edges);
            saveToHistory();
            return;
          }
        }
      } catch (err) {
        console.warn("Could not read level data from localStorage:", err);
      }

      // If no saved data, load canonical initial state
      const initial = targetLevel.getInitialState();
      setNodes(initial.nodes);
      setEdges(initial.edges);
      saveToHistory();
    },
    [setNodes, setEdges, saveToHistory]
  );

  // Initialize level on first mount
  useEffect(() => {
    loadLevel(currentLevelId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Save student's current canvas state periodically or on change
  useEffect(() => {
    if (nodes.length > 0) {
      try {
        localStorage.setItem(
          `${STORAGE_KEYS.LEVEL_PREFIX}${currentLevelId}`,
          JSON.stringify({ nodes, edges })
        );
      } catch (err) {
        console.warn("Could not save level state:", err);
      }
    }
  }, [nodes, edges, currentLevelId]);

  // Navigate to level
  const goToLevel = useCallback(
    (targetLevelId: number) => {
      if (targetLevelId < 1 || targetLevelId > 6) return;
      if (targetLevelId > unlockedLevel) return; // Locked!

      // Save current level before switching
      try {
        localStorage.setItem(
          `${STORAGE_KEYS.LEVEL_PREFIX}${currentLevelId}`,
          JSON.stringify({ nodes, edges })
        );
        localStorage.setItem(STORAGE_KEYS.CURRENT_LEVEL, String(targetLevelId));
      } catch (err) {
        console.warn("Error saving state before switching level:", err);
      }

      setCurrentLevelId(targetLevelId);
      loadLevel(targetLevelId);
    },
    [currentLevelId, unlockedLevel, nodes, edges, loadLevel]
  );

  // Reset current single level to its canonical starting state (for the single step in case of confusion)
  const resetCurrentLevel = useCallback(() => {
    const initial = currentLevel.getInitialState();
    setNodes(initial.nodes);
    setEdges(initial.edges);
    saveToHistory();

    try {
      localStorage.removeItem(`${STORAGE_KEYS.LEVEL_PREFIX}${currentLevelId}`);
    } catch (err) {
      console.warn("Error removing level cache:", err);
    }

    setShowResetConfirmDialog(false);
  }, [currentLevel, currentLevelId, setNodes, setEdges, saveToHistory]);

  // Programmatic full reset (used by CLI reset or internal reload)
  const resetEntireSandbox = useCallback(() => {
    try {
      localStorage.removeItem(STORAGE_KEYS.UNLOCKED_LEVEL);
      localStorage.removeItem(STORAGE_KEYS.CURRENT_LEVEL);
      for (let i = 1; i <= 6; i++) {
        localStorage.removeItem(`${STORAGE_KEYS.LEVEL_PREFIX}${i}`);
      }
      localStorage.removeItem("blockdl-welcome-shown");
    } catch (err) {
      console.warn("Error clearing localStorage:", err);
    }

    setUnlockedLevel(1);
    setCurrentLevelId(1);

    const initial = VGG16_LEVELS[0].getInitialState();
    setNodes(initial.nodes);
    setEdges(initial.edges);
    clearHistory();
    saveToHistory();
  }, [setNodes, setEdges, clearHistory, saveToHistory]);

  // Run validation
  const validateCurrentLevel = useCallback(() => {
    const result = validateVGG16Level(currentLevelId, nodes, edges);
    setValidationResult(result);
    setShowValidationDialog(true);

    if (result.isValid) {
      const nextLevel = currentLevelId + 1;
      if (nextLevel <= 6 && nextLevel > unlockedLevel) {
        setUnlockedLevel(nextLevel);
        try {
          localStorage.setItem(STORAGE_KEYS.UNLOCKED_LEVEL, String(nextLevel));
        } catch (err) {
          console.warn("Error saving unlocked level:", err);
        }
      }
    }
  }, [currentLevelId, nodes, edges, unlockedLevel]);

  // Advance to next level after successful validation
  const proceedToNextLevel = useCallback(() => {
    setShowValidationDialog(false);
    if (currentLevelId < 6) {
      goToLevel(currentLevelId + 1);
    }
  }, [currentLevelId, goToLevel]);

  return {
    currentLevel,
    currentLevelId,
    unlockedLevel,
    allLevels: VGG16_LEVELS,
    validationResult,
    showValidationDialog,
    setShowValidationDialog,
    showReferencePanel,
    setShowReferencePanel,
    showResetConfirmDialog,
    setShowResetConfirmDialog,
    goToLevel,
    resetCurrentLevel,
    resetEntireSandbox,
    validateCurrentLevel,
    proceedToNextLevel,
  };
}
