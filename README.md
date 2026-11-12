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
├── blockdl-main/                # Web app didattica interattiva a 6 livelli con validazione progressiva
│   ├── src/                     # Sorgenti React + Canvas + Validatore integrato
│   └── vite-detector-plugin.ts  # Plugin Vite con API per lanciare in automatico la conversione .pt
│
├── convert_vgg_to_detector.py   # Script coordinatore principale: valida BlockDL e genera il detector
├── vgg_validator.py             # Modulo di validazione rigorosa VGG-16 e diagnostica didattica BlockDL
├── detector_builder.py          # Costruttore dell'Object Detector Ultralytics (ImageNet + YOLO Detect Head)
├── main.py                      # Script di conversione CNN per compiti di classificazione
├── convert_to_ultralytics.py    # Engine per la classificazione
├── prepare_mnist_1_7.py         # Scarica e prepara MNIST (1 e 7) in formato Ultralytics
├── prepare_mnist_color_1_7.py   # MNIST colorato ad alto contrasto per rilevamento dal vivo su smartphone
├── predict.py                   # Script di inferenza e test interattivo (anteprima ASCII)
├── reset_environment.py         # Script per il reset totale dell'ambiente (PT, YAML, network, sandbox)
├── requirements.txt             # Dipendenze necessarie (PyTorch, Ultralytics, PyYAML)
├── README.md                    # Questa guida
│
└── network to convert/          # Cartella dove viene salvato il file .py esportato da BlockDL
    └── vgg16_blockdl.py         # File di riferimento VGG-16 generato da BlockDL
```

---

## 🚀 Fase 2: Da VGG-16 (BlockDL) a YOLO Object Detector

### 1. Avvio dell'Ambiente Interattivo (BlockDL Progressive Sandbox)

Gli studenti avviano l'ambiente interattivo guidato a 6 livelli:

```bash
cd blockdl-main
npm run dev
```

Nel browser (all'indirizzo `http://localhost:5173`):
1. Gli studenti costruiscono la rete blocco per blocco (Livello 1: C1 $\to$ Livello 2: C2 $\to$ Livello 3: C3 $\to$ Livello 4: C4 $\to$ Livello 5: C5 $\to$ Livello 6: Classificatore FC).
2. La sandbox valida automaticamente ciascun livello evidenziando eventuali incongruenze graficamente rispetto al blueprint VGG-16.
3. **Completamento del Livello 6**: una volta completata e verificata l'intera architettura, la sandbox:
   - Salva il codice PyTorch in `network to convert/vgg16_blockdl.py`.
   - **Avvia automaticamente** la trasformazione in Object Detector Ultralytics (`vgg16_detector.pt`).
   - Mostra agli studenti l'adattamento della loro rete (rimozione dei 3 layer densi finali, collegamento delle feature map multi-scala C3-C5 alla YOLO Detect Head, e calibrazione pesi ImageNet per il transfer learning).
   - Fornisce il pulsante **Scarica .pt** e i comandi pronti all'uso per proseguire l'addestramento su Ultralytics Platform!

*(In alternativa, è sempre possibile lanciare la conversione da terminale con `python convert_vgg_to_detector.py`).*

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

## 👨‍🏫 Reset Totale dell'Ambiente di Laboratorio (Solo da Terminale)

Per consentire al docente di eseguire simulazioni e prove e poi **riportare l'intero ambiente allo stato iniziale pulito** prima di distribuirlo agli studenti:

```bash
python reset_environment.py
```

### Caratteristiche di Sicurezza & Operazioni:
- **Protezione da esecuzioni accidentali**: richiede una digitazione esplicita di conferma da terminale (`RESET` / `s` / `yes`), evitando cancellazioni per errore. *(Per script automatici è possibile usare `python reset_environment.py --force`).*
- **Pulizia Modelli**: elimina tutti i modelli `.pt` e file `.yaml` generati.
- **Svuotamento Cartella**: cancella tutti i file esportati all'interno di `network to convert/`.
- **Pulizia Cache**: rimuove cartelle temporanee di training o download (`runs/`, `datasets/`, `__pycache__`).
- **Reset Sandbox BlockDL**: azzera la versione di reset del browser; al prossimo avvio o refresh, tutti i livelli (2..6) tornano bloccati e il canvas si riapre pulito sul Livello 1.

*(Disponibile anche dalla cartella `blockdl-main` tramite il comando `npm run reset:all`).*

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
