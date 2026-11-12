import type { Node, Edge } from "@xyflow/react";

export interface BoundingBox {
  left: number;   // percentage
  top: number;    // percentage
  width: number;  // percentage
  height: number; // percentage
}

export interface LevelRequirement {
  name: string;
  type: string;
  paramsDesc: string;
}

export interface VGG16Level {
  id: number;
  title: string;
  blockName: string;
  badge: string;
  objective: string;
  didacticExplanation: string;
  tensorShapeInfo: {
    input: string;
    output: string;
    formula: string;
  };
  requirements: LevelRequirement[];
  boxes: {
    blocksImage: BoundingBox;
    featuresImage: BoundingBox;
  };
  getInitialState: () => { nodes: Node[]; edges: Edge[] };
}

// Helper node builders
const createEdge = (source: string, target: string): Edge => ({
  id: `e-${source}-${target}`,
  source,
  target,
  type: "smoothstep",
  style: { strokeWidth: 2, stroke: "#64748b" },
});

const createInputNode = (id = "vgg-input", x = 100, y = 50): Node => ({
  id,
  type: "layerNode",
  position: { x, y },
  data: {
    type: "Input",
    params: {
      inputType: "image_color",
      height: 224,
      width: 224,
      channels: 3,
    },
  },
});

const createConvNode = (
  id: string,
  x: number,
  y: number,
  filters: number,
  activation = "relu"
): Node => ({
  id,
  type: "layerNode",
  position: { x, y },
  data: {
    type: "Conv2D",
    params: {
      filters,
      kernel_size: "(3,3)",
      strides: "(1,1)",
      padding: "same",
      activation,
      use_bias: "true",
    },
  },
});

const createMaxPoolNode = (id: string, x: number, y: number): Node => ({
  id,
  type: "layerNode",
  position: { x, y },
  data: {
    type: "MaxPool2D",
    params: {
      pool_size: "(2,2)",
      strides: "(2,2)",
      padding: "valid",
    },
  },
});

export const createFlattenNode = (id: string, x: number, y: number): Node => ({
  id,
  type: "layerNode",
  position: { x, y },
  data: {
    type: "Flatten",
    params: {},
  },
});

export const createDenseNode = (
  id: string,
  x: number,
  y: number,
  units: number,
  activation = "relu"
): Node => ({
  id,
  type: "layerNode",
  position: { x, y },
  data: {
    type: "Dense",
    params: {
      units,
      activation,
      use_bias: "true",
    },
  },
});

export const createDropoutNode = (
  id: string,
  x: number,
  y: number,
  rate = 0.5
): Node => ({
  id,
  type: "layerNode",
  position: { x, y },
  data: {
    type: "Dropout",
    params: {
      rate,
    },
  },
});

// Canonical Block definitions for progressive loading
export const getCanonicalC1 = () => {
  const nodes = [
    createInputNode("vgg-input", 100, 50),
    createConvNode("c1-conv-1", 100, 180, 64),
    createConvNode("c1-conv-2", 100, 310, 64),
    createMaxPoolNode("c1-pool", 100, 440),
  ];
  const edges = [
    createEdge("vgg-input", "c1-conv-1"),
    createEdge("c1-conv-1", "c1-conv-2"),
    createEdge("c1-conv-2", "c1-pool"),
  ];
  return { nodes, edges };
};

export const getCanonicalC2 = () => {
  const c1 = getCanonicalC1();
  const nodes = [
    ...c1.nodes,
    createConvNode("c2-conv-1", 380, 50, 128),
    createConvNode("c2-conv-2", 380, 180, 128),
    createMaxPoolNode("c2-pool", 380, 310),
  ];
  const edges = [
    ...c1.edges,
    createEdge("c1-pool", "c2-conv-1"),
    createEdge("c2-conv-1", "c2-conv-2"),
    createEdge("c2-conv-2", "c2-pool"),
  ];
  return { nodes, edges };
};

export const getCanonicalC3 = () => {
  const c2 = getCanonicalC2();
  const nodes = [
    ...c2.nodes,
    createConvNode("c3-conv-1", 660, 50, 256),
    createConvNode("c3-conv-2", 660, 180, 256),
    createConvNode("c3-conv-3", 660, 310, 256),
    createMaxPoolNode("c3-pool", 660, 440),
  ];
  const edges = [
    ...c2.edges,
    createEdge("c2-pool", "c3-conv-1"),
    createEdge("c3-conv-1", "c3-conv-2"),
    createEdge("c3-conv-2", "c3-conv-3"),
    createEdge("c3-conv-3", "c3-pool"),
  ];
  return { nodes, edges };
};

export const getCanonicalC4 = () => {
  const c3 = getCanonicalC3();
  const nodes = [
    ...c3.nodes,
    createConvNode("c4-conv-1", 940, 50, 512),
    createConvNode("c4-conv-2", 940, 180, 512),
    createConvNode("c4-conv-3", 940, 310, 512),
    createMaxPoolNode("c4-pool", 940, 440),
  ];
  const edges = [
    ...c3.edges,
    createEdge("c3-pool", "c4-conv-1"),
    createEdge("c4-conv-1", "c4-conv-2"),
    createEdge("c4-conv-2", "c4-conv-3"),
    createEdge("c4-conv-3", "c4-pool"),
  ];
  return { nodes, edges };
};

export const getCanonicalC5 = () => {
  const c4 = getCanonicalC4();
  const nodes = [
    ...c4.nodes,
    createConvNode("c5-conv-1", 1220, 50, 512),
    createConvNode("c5-conv-2", 1220, 180, 512),
    createConvNode("c5-conv-3", 1220, 310, 512),
    createMaxPoolNode("c5-pool", 1220, 440),
  ];
  const edges = [
    ...c4.edges,
    createEdge("c4-pool", "c5-conv-1"),
    createEdge("c5-conv-1", "c5-conv-2"),
    createEdge("c5-conv-2", "c5-conv-3"),
    createEdge("c5-conv-3", "c5-pool"),
  ];
  return { nodes, edges };
};

export const VGG16_LEVELS: VGG16Level[] = [
  {
    id: 1,
    title: "Livello 1: Blocco Convoluzionale C1",
    blockName: "Blocco C1",
    badge: "Blocco C1",
    objective:
      "Osserva la porzione evidenziata nel diagramma di riferimento: partendo dal nodo di Input, identifica quanti e quali blocchi compongono il Blocco C1, imposta i relativi iperparametri e collegali in sequenza.",
    didacticExplanation:
      "I primi layer convoluzionali estraggono le feature visive primitive (bordi, gradienti e contrasti cromatici). La VGG16 adotta una strategia precisa con filtri 3x3 e pooling: consulta la reference visiva per dedurre numero di filtri e dimensioni dei tensori intermedi.",
    tensorShapeInfo: {
      input: "3 × 224 × 224 (RGB)",
      output: "Vedi diagramma 3D",
      formula: "Il pooling dimezza le dimensioni spaziali della mappa di feature.",
    },
    requirements: [
      {
        name: "Nodo Input",
        type: "Input",
        paramsDesc: "Immagine a colori standard VGG16 (224 × 224 × 3)",
      },
      {
        name: "Layer Convoluzionali C1",
        type: "Conv2D",
        paramsDesc: "Deduci numero di layer, filtri e padding dal diagramma",
      },
      {
        name: "Layer di Pooling C1",
        type: "MaxPool2D",
        paramsDesc: "Dimezza la risoluzione spaziale",
      },
    ],
    boxes: {
      blocksImage: { left: 4.38, top: 19.31, width: 11.43, height: 48.82 },
      featuresImage: { left: 10.0, top: 5.92, width: 16.11, height: 91.72 },
    },
    getInitialState: () => ({
      nodes: [createInputNode("vgg-input", 100, 50)],
      edges: [],
    }),
  },
  {
    id: 2,
    title: "Livello 2: Blocco Convoluzionale C2",
    blockName: "Blocco C2",
    badge: "Blocco C2",
    objective:
      "Il blocco C1 è stato pre-caricato sul canvas. Consulta la bounding box sul diagramma per costruire e collegare il blocco C2.",
    didacticExplanation:
      "Osserva l'evoluzione dei canali rispetto al blocco C1: man mano che la risoluzione spaziale si dimezza, l'architettura VGG16 adegua il numero di filtri per combinare bordi e gradienti in texture più ricche.",
    tensorShapeInfo: {
      input: "Uscita da C1",
      output: "Vedi diagramma 3D",
      formula: "Confronta il raddoppio dei canali e il dimezzamento della griglia spaziale.",
    },
    requirements: [
      {
        name: "Layer Convoluzionali C2",
        type: "Conv2D",
        paramsDesc: "Verifica nel diagramma il numero di layer e la profondità dei filtri",
      },
      {
        name: "Layer di Pooling C2",
        type: "MaxPool2D",
        paramsDesc: "Operazione di subsampling per ridurre la risoluzione",
      },
    ],
    boxes: {
      blocksImage: { left: 20.39, top: 19.31, width: 11.4, height: 48.82 },
      featuresImage: { left: 18.33, top: 26.63, width: 14.44, height: 55.23 },
    },
    getInitialState: () => getCanonicalC1(),
  },
  {
    id: 3,
    title: "Livello 3: Blocco Convoluzionale C3",
    blockName: "Blocco C3",
    badge: "Blocco C3",
    objective:
      "I blocchi C1 e C2 sono pronti. Fai molta attenzione alla composizione interna del blocco C3 evidenziato nel diagramma: quanti layer convoluzionali sono presenti?",
    didacticExplanation:
      "A partire dal blocco C3, la VGG16 incrementa la profondità convoluzionale per catturare strutture geometriche complesse. Osserva con attenzione la composizione nel blueprint prima di collegare i blocchi.",
    tensorShapeInfo: {
      input: "Uscita da C2",
      output: "Vedi diagramma 3D",
      formula: "Le feature map si riducono ulteriormente mentre crescono i canali semantici.",
    },
    requirements: [
      {
        name: "Layer Convoluzionali C3",
        type: "Conv2D",
        paramsDesc: "Controlla quanti layer convoluzionali consecutivi compongono C3",
      },
      {
        name: "Layer di Pooling C3",
        type: "MaxPool2D",
        paramsDesc: "Riduzione della dimensione spaziale",
      },
    ],
    boxes: {
      blocksImage: { left: 34.67, top: 19.31, width: 14.85, height: 48.82 },
      featuresImage: { left: 28.33, top: 38.46, width: 14.44, height: 36.49 },
    },
    getInitialState: () => getCanonicalC2(),
  },
  {
    id: 4,
    title: "Livello 4: Blocco Convoluzionale C4",
    blockName: "Blocco C4",
    badge: "Blocco C4",
    objective:
      "Collega e realizza il blocco C4: verifica nel diagramma di riferimento la profondità massima di filtri raggiunta e il numero di convoluzioni previste.",
    didacticExplanation:
      "In questo stadio la rete estrae feature visive ad alto livello semantico (parti di oggetti e pattern complessi). Si raggiunge il valore massimo di filtri previsto nella famiglia VGG.",
    tensorShapeInfo: {
      input: "Uscita da C3",
      output: "Vedi diagramma 3D",
      formula: "Risoluzione ridotta a un quarto rispetto all'ingresso di C2.",
    },
    requirements: [
      {
        name: "Layer Convoluzionali C4",
        type: "Conv2D",
        paramsDesc: "Profondità filtri e numero di layer desunti dal diagramma",
      },
      {
        name: "Layer di Pooling C4",
        type: "MaxPool2D",
        paramsDesc: "Pooling finale del blocco C4",
      },
    ],
    boxes: {
      blocksImage: { left: 50.68, top: 19.31, width: 14.82, height: 48.82 },
      featuresImage: { left: 40.56, top: 43.39, width: 17.22, height: 28.6 },
    },
    getInitialState: () => getCanonicalC3(),
  },
  {
    id: 5,
    title: "Livello 5: Blocco Convoluzionale C5",
    blockName: "Blocco C5",
    badge: "Blocco C5",
    objective:
      "Completa l'estrattore di feature con il blocco C5: consulta la sezione evidenziata nella reference visiva per finalizzare la parte convoluzionale della VGG16.",
    didacticExplanation:
      "C5 è l'ultimo blocco dell'estrattore di feature. Al termine del pooling finale, la mappa spaziale viene compressa a 7x7 celle, ciascuna con centinaia di canali descrittori ad altissima astrazione semantica.",
    tensorShapeInfo: {
      input: "Uscita da C4",
      output: "7 × 7 con profondità finale",
      formula: "La mappa 7x7 rappresenta l'ingresso per la testata di classificazione.",
    },
    requirements: [
      {
        name: "Layer Convoluzionali C5",
        type: "Conv2D",
        paramsDesc: "Ultima terna convoluzionale dell'estrattore di feature",
      },
      {
        name: "Layer di Pooling C5",
        type: "MaxPool2D",
        paramsDesc: "Pooling finale prima del classificatore",
      },
    ],
    boxes: {
      blocksImage: { left: 66.7, top: 19.31, width: 14.85, height: 48.82 },
      featuresImage: { left: 55.56, top: 45.36, width: 17.78, height: 24.65 },
    },
    getInitialState: () => getCanonicalC4(),
  },
  {
    id: 6,
    title: "Livello 6: Classificatore Fully Connected",
    blockName: "Classifier (FC)",
    badge: "Testata Classificatore",
    objective:
      "L'estrattore di feature è completo. Ora realizza la testata densa (Fully Connected) per la classificazione: appiattisci il volume tensoriale in un vettore 1D e aggiungi i layer densi, inserendo opportune tecniche di regolarizzazione.",
    didacticExplanation:
      "Per passare dalle feature map 3D alla previsione delle 1000 classi, serve appiattire il tensore e connetterlo a strati densi. A causa dell'elevatissimo numero di connessioni nei layer densi, la VGG16 introduce layer di Dropout per contrastare l'overfitting.",
    tensorShapeInfo: {
      input: "Mappa 7 × 7 (da C5)",
      output: "1000 classi (ImageNet)",
      formula: "Flattening -> Strati densi intermedi -> Previsione sulle 1000 classi.",
    },
    requirements: [
      {
        name: "Layer di Flattening",
        type: "Flatten",
        paramsDesc: "Conversione da volume 3D a vettore 1D",
      },
      {
        name: "Layer Densi (Fully Connected)",
        type: "Dense",
        paramsDesc: "Strati di neuroni completamente connessi (vedi schema per numero di unità)",
      },
      {
        name: "Regolarizzazione Dropout",
        type: "Dropout",
        paramsDesc: "Disattivazione casuale dei neuroni per prevenire l'overfitting",
      },
      {
        name: "Layer di Uscita",
        type: "Dense o Output",
        paramsDesc: "Classificazione finale sulle classi del dataset ImageNet",
      },
    ],
    boxes: {
      blocksImage: { left: 84.4, top: 19.31, width: 11.43, height: 48.82 },
      featuresImage: { left: 74.44, top: 45.36, width: 25.0, height: 18.74 },
    },
    getInitialState: () => getCanonicalC5(),
  },
];
