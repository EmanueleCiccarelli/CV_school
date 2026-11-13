import type { Node, Edge } from "@xyflow/react";
import graphlib from "graphlib";
import { VGG16_LEVELS } from "./vgg16LevelsConfig";

export interface ValidationResult {
  isValid: boolean;
  title: string;
  message: string;
  details?: string[];
  blockName?: string;
  nextLevelUnlocked?: boolean;
}

interface NormalizedLayer {
  nodeId: string;
  type: string;
  params: Record<string, unknown>;
  absorbedActivation?: boolean;
  activation?: string;
}

interface ExpectedLayerSpec {
  block: string;
  type: string;
  name: string;
  // Checks non-activation structural parameters. Returns error description or null.
  checkStructure: (layer: NormalizedLayer) => string | null;
  // Checks activation function specifically. Returns boolean: true if activation is ReLU.
  checkActivation?: (layer: NormalizedLayer) => boolean;
}

// Generate the expected linear sequence of layer specs up to a given level
function getExpectedSpecsForLevel(levelId: number): ExpectedLayerSpec[] {
  const specs: ExpectedLayerSpec[] = [];

  // Input
  specs.push({
    block: "Input",
    type: "Input",
    name: "Input",
    checkStructure: (l) => {
      const h = Number(l.params.height) || 0;
      const w = Number(l.params.width) || 0;
      const inputType = String(l.params.inputType || "");
      if (h !== 224 || w !== 224) return "wrong_dimensions";
      if (inputType !== "image_color" && inputType !== "image_custom") return "wrong_type";
      return null;
    },
  });

  // Level 1: C1 (2x Conv64 + MaxPool)
  if (levelId >= 1) {
    specs.push(
      createConvSpec("Blocco C1", "Conv2D #1", 64),
      createConvSpec("Blocco C1", "Conv2D #2", 64),
      createMaxPoolSpec("Blocco C1", "MaxPool2D")
    );
  }

  // Level 2: C2 (2x Conv128 + MaxPool)
  if (levelId >= 2) {
    specs.push(
      createConvSpec("Blocco C2", "Conv2D #1", 128),
      createConvSpec("Blocco C2", "Conv2D #2", 128),
      createMaxPoolSpec("Blocco C2", "MaxPool2D")
    );
  }

  // Level 3: C3 (3x Conv256 + MaxPool)
  if (levelId >= 3) {
    specs.push(
      createConvSpec("Blocco C3", "Conv2D #1", 256),
      createConvSpec("Blocco C3", "Conv2D #2", 256),
      createConvSpec("Blocco C3", "Conv2D #3", 256),
      createMaxPoolSpec("Blocco C3", "MaxPool2D")
    );
  }

  // Level 4: C4 (3x Conv512 + MaxPool)
  if (levelId >= 4) {
    specs.push(
      createConvSpec("Blocco C4", "Conv2D #1", 512),
      createConvSpec("Blocco C4", "Conv2D #2", 512),
      createConvSpec("Blocco C4", "Conv2D #3", 512),
      createMaxPoolSpec("Blocco C4", "MaxPool2D")
    );
  }

  // Level 5: C5 (3x Conv512 + MaxPool)
  if (levelId >= 5) {
    specs.push(
      createConvSpec("Blocco C5", "Conv2D #1", 512),
      createConvSpec("Blocco C5", "Conv2D #2", 512),
      createConvSpec("Blocco C5", "Conv2D #3", 512),
      createMaxPoolSpec("Blocco C5", "MaxPool2D")
    );
  }

  // Level 6: Classifier
  if (levelId >= 6) {
    specs.push({
      block: "Classifier",
      type: "Flatten",
      name: "Flatten",
      checkStructure: (l) => (l.type === "Flatten" ? null : "wrong_type"),
    });
    specs.push(createDenseSpec("Classifier", "Dense #1", 4096, true));
    specs.push(createDropoutSpec("Classifier", "Dropout #1", 0.5));
    specs.push(createDenseSpec("Classifier", "Dense #2", 4096, true));
    specs.push(createDropoutSpec("Classifier", "Dropout #2", 0.5));
    specs.push({
      block: "Classifier",
      type: "Dense|Output",
      name: "Output Dense",
      checkStructure: (l) => {
        if (l.type === "Dense") {
          const units = Number(l.params.units);
          if (units !== 1000) return "wrong_units";
          if (!isBiasValid(l.params.use_bias)) return "wrong_bias";
          return null;
        }
        if (l.type === "Output") {
          const numClasses = Number(l.params.numClasses) || Number(l.params.units);
          return numClasses === 1000 ? null : "wrong_units";
        }
        return "wrong_type";
      },
    });
  }

  return specs;
}

function isConvStrideValid(val: unknown): boolean {
  if (val === undefined || val === null || val === "") return true; // default in BlockDL is (1,1)
  const s = String(val).replace(/\s/g, "");
  if (s === "(1,1)" || s === "1" || s === "(1)" || s === "1,1") return true;
  const match = s.match(/^\(?(\d+),(\d+)\)?$/);
  if (match) {
    return Number(match[1]) === 1 && Number(match[2]) === 1;
  }
  const num = Number(s);
  return num === 1;
}

function isConvPaddingValid(val: unknown): boolean {
  if (val === undefined || val === null || val === "") return true; // default in BlockDL is same
  const s = String(val).trim().toLowerCase();
  return s === "same";
}

function isBiasValid(val: unknown): boolean {
  if (val === undefined || val === null || val === "") return true; // default in BlockDL is true
  if (typeof val === "boolean") return val;
  const s = String(val).trim().toLowerCase();
  return s === "true" || s === "1";
}

function isMaxPoolSizeValid(val: unknown): boolean {
  if (val === undefined || val === null) return true; // default is (2,2)
  const s = String(val).replace(/\s/g, "");
  if (s === "") return false;
  if (s === "(2,2)" || s === "2" || s === "(2)" || s === "2,2") return true;
  const match = s.match(/^\(?(\d+),(\d+)\)?$/);
  if (match) {
    return Number(match[1]) === 2 && Number(match[2]) === 2;
  }
  const num = Number(s);
  return num === 2;
}

function isMaxPoolStrideValid(val: unknown): boolean {
  if (val === undefined || val === null || val === "") return true; // defaults to pool_size (2x2)
  const s = String(val).replace(/\s/g, "");
  if (s === "(2,2)" || s === "2" || s === "(2)" || s === "2,2") return true;
  const match = s.match(/^\(?(\d+),(\d+)\)?$/);
  if (match) {
    return Number(match[1]) === 2 && Number(match[2]) === 2;
  }
  const num = Number(s);
  return num === 2;
}

function isMaxPoolPaddingValid(val: unknown): boolean {
  if (val === undefined || val === null || val === "") return true; // default is valid
  const s = String(val).trim().toLowerCase();
  return s === "valid";
}

function createConvSpec(block: string, name: string, expectedFilters: number): ExpectedLayerSpec {
  return {
    block,
    type: "Conv2D",
    name,
    checkStructure: (layer) => {
      if (layer.type !== "Conv2D") return "wrong_type";
      const filters = Number(layer.params.filters);
      if (filters !== expectedFilters) return "wrong_filters";
      const kernel = String(layer.params.kernel_size || "").replace(/\s/g, "");
      if (kernel !== "(3,3)" && kernel !== "3" && kernel !== "3,3") return "wrong_kernel";
      if (!isConvStrideValid(layer.params.strides)) return "wrong_strides";
      if (!isConvPaddingValid(layer.params.padding)) return "wrong_padding";
      if (!isBiasValid(layer.params.use_bias)) return "wrong_bias";
      return null;
    },
    checkActivation: (layer) => {
      const act = String(layer.activation || layer.params.activation || "linear").toLowerCase();
      return act === "relu";
    },
  };
}

function createMaxPoolSpec(block: string, name: string): ExpectedLayerSpec {
  return {
    block,
    type: "MaxPool2D",
    name,
    checkStructure: (layer) => {
      if (layer.type !== "MaxPool2D") return "wrong_type";
      if (!isMaxPoolSizeValid(layer.params.pool_size)) return "wrong_pool_size";
      if (!isMaxPoolStrideValid(layer.params.strides)) return "wrong_strides";
      if (!isMaxPoolPaddingValid(layer.params.padding)) return "wrong_padding";
      return null;
    },
  };
}

function createDenseSpec(
  block: string,
  name: string,
  expectedUnits: number,
  requiresRelu = true
): ExpectedLayerSpec {
  return {
    block,
    type: "Dense",
    name,
    checkStructure: (layer) => {
      if (layer.type !== "Dense") return "wrong_type";
      const units = Number(layer.params.units);
      if (units !== expectedUnits) return "wrong_units";
      if (!isBiasValid(layer.params.use_bias)) return "wrong_bias";
      return null;
    },
    checkActivation: requiresRelu
      ? (layer) => {
          const act = String(layer.activation || layer.params.activation || "linear").toLowerCase();
          return act === "relu";
        }
      : undefined,
  };
}

function createDropoutSpec(block: string, name: string, expectedRate = 0.5): ExpectedLayerSpec {
  return {
    block,
    type: "Dropout",
    name,
    checkStructure: (layer) => {
      if (layer.type !== "Dropout") return "wrong_type";
      const rate = Number(layer.params.rate);
      if (Math.abs(rate - expectedRate) > 0.05) return "wrong_rate";
      return null;
    },
  };
}

export function validateVGG16Level(
  levelId: number,
  nodes: Node[],
  edges: Edge[]
): ValidationResult {
  const currentLevelConfig = VGG16_LEVELS.find((l) => l.id === levelId);
  const blockName = currentLevelConfig ? currentLevelConfig.blockName : `Livello ${levelId}`;

  // 1. Basic empty check
  if (!nodes || nodes.length === 0) {
    return {
      isValid: false,
      title: "Canvas Vuoto",
      message: "Il canvas è vuoto! Trascina i layer necessari dalla palette a sinistra per iniziare a costruire il blocco.",
      blockName,
    };
  }

  // 2. Build graph and check structural integrity
  const graph = new graphlib.Graph({ directed: true });
  const nodeMap = new Map<string, Node>();

  nodes.forEach((n) => {
    nodeMap.set(n.id, n);
    graph.setNode(n.id);
  });

  edges.forEach((e) => {
    graph.setEdge(e.source, e.target);
  });

  // Check cycles
  if (!graphlib.alg.isAcyclic(graph)) {
    return {
      isValid: false,
      title: "Verifica Fallita",
      message: "La configurazione della rete non è corretta. Ricontrolla attentamente il diagramma di riferimento e verifica i collegamenti tra i blocchi (è presente un loop/ciclo chiuso).",
      details: [
        "La VGG16 è una rete strettamente sequenziale (feed-forward) senza ritorni all'indietro.",
        "Confronta i blocchi inseriti con l'immagine di riferimento.",
      ],
      blockName,
    };
  }

  // Find Input nodes
  const inputLayerNodes = nodes.filter((n) => (n.data as { type?: string })?.type === "Input");
  if (inputLayerNodes.length !== 1) {
    return {
      isValid: false,
      title: "Verifica Fallita",
      message: "La configurazione della rete non è corretta. Ricontrolla attentamente il diagramma di riferimento e verifica che la rete inizi con un singolo blocco di Input.",
      details: [
        "Verifica che sia presente il blocco di Input iniziale della rete.",
        "Confronta la struttura con l'immagine di riferimento.",
      ],
      blockName,
    };
  }

  const rootInputNode = inputLayerNodes[0];

  // Check inEdges to Input
  const inEdges = graph.inEdges(rootInputNode.id) || [];
  if (inEdges.length > 0) {
    return {
      isValid: false,
      title: "Verifica Fallita",
      message: "La configurazione della rete non è corretta. Ricontrolla attentamente il diagramma di riferimento e verifica i collegamenti (l'Input non deve avere collegamenti in ingresso).",
      blockName,
    };
  }

  // Traverse linear chain from Input
  const rawChain: Node[] = [];
  let currentNodeId: string | null = rootInputNode.id;
  const visited = new Set<string>();

  while (currentNodeId) {
    if (visited.has(currentNodeId)) break;
    visited.add(currentNodeId);

    const node = nodeMap.get(currentNodeId);
    if (!node) break;
    rawChain.push(node);

    const outEdges: graphlib.Edge[] = (graph.outEdges(currentNodeId) || []) as graphlib.Edge[];
    if (outEdges.length === 0) {
      currentNodeId = null; // Reached end of pipeline
    } else if (outEdges.length === 1) {
      currentNodeId = outEdges[0].w;
    } else {
      // Branching detected
      return {
        isValid: false,
        title: "Verifica Fallita",
        message: "La configurazione della rete non è corretta. Ricontrolla attentamente il diagramma di riferimento e verifica i collegamenti tra i blocchi (la VGG16 è una catena lineare, ogni blocco deve avere una sola uscita).",
        details: [
          "Rimuovi eventuali archi duplicati o diramazioni.",
          "Confronta con la reference visiva dell'architettura.",
        ],
        blockName,
      };
    }
  }

  // Check for orphan / disconnected nodes
  const unvisitedNodes = nodes.filter((n) => !visited.has(n.id));
  if (unvisitedNodes.length > 0) {
    return {
      isValid: false,
      title: "Verifica Fallita",
      message: "La configurazione della rete non è corretta. Ricontrolla attentamente il diagramma di riferimento e verifica che tutti i blocchi siano connessi alla sequenza principale.",
      details: [
        "Ci sono blocchi sul canvas che non risultano collegati al flusso della rete.",
        "Collega i blocchi mancanti oppure elimina i blocchi isolati con l'icona cestino.",
      ],
      blockName,
    };
  }

  // 3. Normalize layer sequence (absorbing standalone Activation('relu') into previous Conv/Dense)
  const normalizedLayers: NormalizedLayer[] = [];
  for (let i = 0; i < rawChain.length; i++) {
    const node = rawChain[i];
    const data = node.data as { type: string; params: Record<string, unknown> };
    const type = data.type;
    const params = { ...(data.params || {}) };

    if (type === "Activation") {
      const actFunc = String(params.activation_function || "relu");
      if (normalizedLayers.length > 0) {
        const prev = normalizedLayers[normalizedLayers.length - 1];
        if (prev.type === "Conv2D" || prev.type === "Dense") {
          prev.activation = actFunc;
          prev.absorbedActivation = true;
          continue;
        }
      }
    }

    normalizedLayers.push({
      nodeId: node.id,
      type,
      params,
      activation: String(params.activation || "linear"),
    });
  }

  // 4. Compare with expected specs for the level
  const expectedSpecs = getExpectedSpecsForLevel(levelId);

  let hasStructuralError = false;
  let hasParameterError = false;
  let hasActivationError = false;

  // Check if chain length matches
  if (normalizedLayers.length !== expectedSpecs.length) {
    hasStructuralError = true;
  } else {
    // Check each layer in the chain
    for (let i = 0; i < expectedSpecs.length; i++) {
      const spec = expectedSpecs[i];
      const actual = normalizedLayers[i];

      // Check type matching
      if (spec.type.includes("|")) {
        const allowed = spec.type.split("|");
        if (!allowed.includes(actual.type)) {
          hasStructuralError = true;
          break;
        }
      } else if (actual.type !== spec.type) {
        hasStructuralError = true;
        break;
      }

      // Check structure & parameters (filters, kernel, strides, padding, use_bias, pool_size, units, rate)
      const structErr = spec.checkStructure(actual);
      if (structErr) {
        if (structErr === "wrong_type") {
          hasStructuralError = true;
          break;
        } else {
          hasParameterError = true;
        }
      }

      // Check activation
      if (spec.checkActivation) {
        const isActOk = spec.checkActivation(actual);
        if (!isActOk) {
          hasActivationError = true;
        }
      }
    }
  }

  // 5. Evaluate and report with appropriate educational feedback:
  // A. Structural error: wrong layer count, wrong types, missing blocks
  if (hasStructuralError) {
    return {
      isValid: false,
      title: "Verifica Fallita: Struttura Rete Non Valida",
      message:
        "La configurazione della rete non corrisponde all'architettura VGG16 per questo livello. Ricontrolla attentamente il diagramma di riferimento e verifica i blocchi che hai inserito sul canvas.",
      details: [
        "Confronta la struttura costruita sul canvas con la reference visiva della VGG16 (apri 'Guida & Blueprint').",
        "Verifica che il numero di blocchi, il loro tipo e l'ordine sequenziale corrispondano alla sezione evidenziata nel diagramma.",
      ],
      blockName,
    };
  }

  // B. Parameter error: structure is correct, but parameters inside blocks are wrong!
  if (hasParameterError) {
    return {
      isValid: false,
      title: "Verifica Fallita: Parametri Non Corretti",
      message:
        "La struttura della rete è complessivamente corretta, ma i parametri dei vari blocchi sono sbagliati. Controlla in rete i parametri corretti per una VGG16.",
      details: [
        "La sequenza e la tipologia di blocchi che hai inserito sul canvas sono corrette per questo livello.",
        "Tuttavia, uno o più parametri interni ai blocchi (come dimensioni, passo/stride, padding o bias) non corrispondono alla specifica originale della VGG16.",
        "Effettua una ricerca in rete per verificare i parametri esatti previsti per i layer di una VGG16 e correggili nei rispettivi blocchi sul canvas.",
      ],
      blockName,
    };
  }

  // C. Activation error: structure & parameters are right, but activation is wrong
  if (hasActivationError) {
    return {
      isValid: false,
      title: "Funzione di Attivazione Non Valida",
      message:
        "Attenzione: le funzioni di attivazione lineari (o assenti) non sono sufficienti per permettere a una rete neurale profonda come la VGG16 di apprendere pattern complessi.",
      details: [
        "Una sequenza di strati puramente lineari collasserebbe matematicamente in un singolo strato lineare, perdendo tutta la capacità espressiva.",
        "Fai una ricerca online (es. 'VGG16 activation function') per scoprire quale specifica funzione di attivazione non-lineare utilizza l'architettura VGG16 e impostala nei blocchi convoluzionali o densi (puoi anche consultare la legenda della reference visiva).",
      ],
      blockName,
    };
  }

  // 6. Success!
  const isFinalLevel = levelId === 6;
  return {
    isValid: true,
    title: isFinalLevel ? "🎉 VGG16 Costruita con Successo!" : `✅ ${blockName} Completato!`,
    message: isFinalLevel
      ? "Straordinario! Hai completato con successo l'intera architettura VGG16! Il codice PyTorch generato nel pannello destro è ora pronto per essere addestrato."
      : `Ottimo lavoro! Il ${blockName} è stato costruito correttamente secondo l'architettura VGG16. Livello successivo sbloccato!`,
    nextLevelUnlocked: true,
    blockName,
  };
}
