import { useState, useEffect, useMemo } from "react";
import { AppShell } from "./components/AppShell";
import { CanvasEditor } from "./components/CanvasEditor";
import BlockPalette from "./components/BlockPalette";
import { CodeViewer } from "./components/CodeViewer";
import { VGG16LevelBar } from "./components/VGG16LevelBar";
import { VGG16ReferencePanel } from "./components/VGG16ReferencePanel";
import { VGG16CanvasBlueprint } from "./components/VGG16CanvasBlueprint";
import { VGG16FullscreenViewer } from "./components/VGG16FullscreenViewer";
import { ValidationDialog } from "./components/ValidationDialog";
import { useFlowStore } from "./lib/flow-store";
import { useVGG16Sandbox } from "./lib/useVGG16Sandbox";
import { parseGraphToDAG } from "./lib/dag-parser";
import { generatePyTorchCode } from "./lib/code-generation";
import type { Node, Edge } from "@xyflow/react";
import "./App.css";

function App() {
  const { nodes, edges, setNodes, setEdges, initializeHistory } = useFlowStore();

  const {
    currentLevel,
    currentLevelId,
    unlockedLevel,
    allLevels,
    validationResult,
    showValidationDialog,
    setShowValidationDialog,
    showReferencePanel,
    setShowReferencePanel,
    showResetConfirmDialog,
    setShowResetConfirmDialog,
    goToLevel,
    resetCurrentLevel,
    validateCurrentLevel,
    proceedToNextLevel,
  } = useVGG16Sandbox();

  // Fullscreen Viewer state
  const [showFullscreenViewer, setShowFullscreenViewer] = useState(false);
  const [fullscreenViewType, setFullscreenViewType] = useState<"blocks" | "features">("blocks");

  const handleOpenFullscreen = (view: "blocks" | "features" = "blocks") => {
    setFullscreenViewType(view);
    setShowFullscreenViewer(true);
  };

  // Initialize history when app starts
  useEffect(() => {
    initializeHistory();
  }, [initializeHistory]);

  const handleClearAll = () => {
    setNodes([]);
    setEdges([]);
  };

  const handleImportProject = (data: { nodes: Node[]; edges: Edge[] }) => {
    setNodes(data.nodes);
    setEdges(data.edges);
  };

  const levelBarContent = (
    <VGG16LevelBar
      levels={allLevels}
      currentLevelId={currentLevelId}
      unlockedLevel={unlockedLevel}
      onSelectLevel={goToLevel}
      onValidate={validateCurrentLevel}
      onReset={resetCurrentLevel}
      onOpenGuide={() => setShowReferencePanel(true)}
      showResetConfirm={showResetConfirmDialog}
      setShowResetConfirm={setShowResetConfirmDialog}
    />
  );

  const blueprintWidgetContent = (
    <VGG16CanvasBlueprint
      level={currentLevel}
      onOpenFullModal={() => setShowReferencePanel(true)}
    />
  );

  // Genera il codice PyTorch dal grafo per l'esportazione verso Ultralytics
  const generatedPyTorchCode = useMemo(() => {
    try {
      const dagResult = parseGraphToDAG(nodes, edges);
      if (dagResult.isValid) {
        return generatePyTorchCode(dagResult.orderedNodes);
      }
    } catch {
      // Ignora errori intermedi mentre lo studente monta il grafo
    }
    return "";
  }, [nodes, edges]);

  return (
    <>
      <AppShell
        palette={<BlockPalette />}
        canvas={<CanvasEditor />}
        codeViewer={<CodeViewer />}
        levelBar={levelBarContent}
        blueprintWidget={blueprintWidgetContent}
        nodes={nodes}
        edges={edges}
        onImportProject={handleImportProject}
        onClearAll={handleClearAll}
      />

      {/* Guide & Blueprint Modal */}
      <VGG16ReferencePanel
        open={showReferencePanel}
        onOpenChange={setShowReferencePanel}
        level={currentLevel}
        unlockedLevel={unlockedLevel}
        onOpenFullscreen={handleOpenFullscreen}
      />

      {/* Dedicated Fullscreen Reference Viewer with Zoom Controls */}
      <VGG16FullscreenViewer
        open={showFullscreenViewer}
        onClose={() => setShowFullscreenViewer(false)}
        level={currentLevel}
        initialView={fullscreenViewType}
      />

      {/* Diagnostic Validation Feedback Dialog */}
      <ValidationDialog
        open={showValidationDialog}
        onOpenChange={setShowValidationDialog}
        result={validationResult}
        onProceed={proceedToNextLevel}
        isLastLevel={currentLevelId === 6}
        generatedCode={generatedPyTorchCode}
      />
    </>
  );
}

export default App;
