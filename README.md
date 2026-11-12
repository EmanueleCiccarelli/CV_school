# Laboratorio di Computer Vision: da BlockDL a Ultralytics YOLO

Questo progetto è strutturato per un laboratorio didattico di Computer Vision (ad esempio per una classe di quinta liceo).  
L'obiettivo è consentire agli studenti di:
1. **Costruire visivamente una rete neurale a blocchi** tramite [BlockDL](https://blockdl.com/editor).
2. **Convertire il codice esportato in un modello `.pt` nativo** compatibile con la piattaforma [Ultralytics](https://platform.ultralytics.com/) ed il comando `yolo train`.
3. **Addestrare il modello** (in locale o sincronizzato con il cloud Ultralytics) su due compiti:
   - *Fase 1 (Classificazione)*: CNN semplice su versione ridotta di MNIST (cifre 1 e 7) e variante colorata real-world per app mobile.
   - *Fase 2 (Object Detection)*: Costruzione manuale di una **VGG-16**, trasferimento dei pesi pre-addestrati da `torchvision` (ImageNet), taglio dei layer densi e collegamento a una **YOLO Detect Head** multi-scala (C3, C4, C5) senza neck.

---

## 📁 Struttura della Codebase

```
VGG for school/
│
├── convert_vgg_to_detector.py   # Script coordinatore principale: valida BlockDL e genera il detector
├── vgg_validator.py             # Modulo di validazione rigorosa VGG-16 e diagnostica didattica BlockDL
├── detector_builder.py          # Costruttore dell'Object Detector Ultralytics (ImageNet + YOLO Detect Head)
├── main.py                      # Script di conversione CNN per compiti di classificazione
├── convert_to_ultralytics.py    # Engine per la classificazione
├── prepare_mnist_1_7.py         # Scarica e prepara MNIST (1 e 7) in formato Ultralytics
├── prepare_mnist_color_1_7.py   # MNIST colorato ad alto contrasto per rilevamento dal vivo su smartphone
├── predict.py                   # Script di inferenza e test interattivo (anteprima ASCII)
├── requirements.txt             # Dipendenze necessarie (PyTorch, Ultralytics, PyYAML)
├── README.md                    # Questa guida
│
└── network to convert/          # Cartella dove gli studenti inseriscono il file .py esportato da BlockDL
    └── vgg16_blockdl.py         # File di riferimento VGG-16 generato da BlockDL
```

---

## 🚀 Fase 2: Da VGG-16 (BlockDL) a YOLO Object Detector

Gli studenti compongono graficamente l'architettura VGG-16 su BlockDL (13 convoluzioni in 5 blocchi e 5 MaxPool) ed esportano il file `.py` in `network to convert/`.

### 1. Conversione e Validazione Architetturale

Eseguire da terminale:
```bash
python convert_vgg_to_detector.py
```

Lo script coordina in modo modulare la pipeline didattica:
1. **Analisi e Validazione Architetturale Rigorosa (`vgg_validator.py`)**:
   - Controlla che i 13 layer convoluzionali e i 5 MaxPool corrispondano all'architettura standard VGG-16.
   - Se ci sono errori (es. numero filtri errato, padding mancante, blocchi incompleti), stampa un **report diagnostico dettagliato** con istruzioni chiare per lo studente su quale blocco correggere su BlockDL.
2. **Presentazione dell'Adattamento Didattico**:
   - Riconosce ed esclude i 3 layer lineari finali di classificazione.
   - Mostra allo studente l'estrazione delle feature map multi-scala C3 (stride 8, 256 ch), C4 (stride 16, 512 ch) e C5 (stride 32, 512 ch).
   - Collega la **YOLO Detect Head** multi-scala (senza neck) direttamente alle tre feature maps.
   - Spiega l'importanza del **Transfer Learning** con pesi ImageNet per consentire l'addestramento immediato (fine-tuning).
3. **Generazione ed Esportazione (`detector_builder.py`)**:
   - Inizializza l'architettura nativa Ultralytics `DetectionModel` con i pesi ImageNet pre-allenati.
   - Salva il file `vgg16_detector.pt` ed esegue un test automatico di forward pass per confermare la piena operatività.

---

### 2. Addestramento dell'Object Detector su Ultralytics

#### Su Ultralytics Platform (Cloud):
* **Su Ubuntu / Linux (Bash):**
  ```bash
  export ULTRALYTICS_API_KEY="tua_api_key_ultralytics"
  yolo train model=vgg16_detector.pt data=ul://tuo-dataset task=detect imgsz=640 project=tuo-progetto
  ```

* **Su Windows (PowerShell):**
  ```powershell
  $env:ULTRALYTICS_API_KEY="tua_api_key_ultralytics"
  yolo train model=vgg16_detector.pt data=ul://tuo-dataset task=detect imgsz=640 project=tuo-progetto
  ```

* **Su Windows (Prompt dei comandi CMD):**
  ```cmd
  set ULTRALYTICS_API_KEY=tua_api_key_ultralytics
  yolo train model=vgg16_detector.pt data=ul://tuo-dataset task=detect imgsz=640 project=tuo-progetto
  ```

#### In locale offline (es. con dataset di test coco8):
```bash
yolo train model=vgg16_detector.pt data=coco8.yaml task=detect imgsz=640 epochs=10
```

---

## 🎨 Fase 1: Classificazione MNIST (1 e 7) per App Smartphone

Per la fase introduttiva di classificazione:

1. **Generare il dataset colorato**:
   ```bash
   python prepare_mnist_color_1_7.py
   ```
   Crea `mnist_color_1_7.zip` con variazioni cromatiche ad alto contrasto (inchiostro su carta bianca, penna blu, lavagna, ecc.) per funzionare da fotocamera smartphone nel mondo reale.
2. **Convertire la CNN semplice da BlockDL**:
   ```bash
   python main.py
   ```
3. **Testare l'inferenza**:
   ```bash
   python predict.py
   ```
