# Laboratorio di Computer Vision: da BlockDL a Ultralytics YOLO

Questo progetto è strutturato per un laboratorio didattico di Computer Vision (ad esempio per una classe di quinta liceo).  
L'obiettivo è consentire agli studenti di:
1. **Costruire visivamente una rete neurale a blocchi** tramite [BlockDL](https://blockdl.com/editor): costruzione progressiva guidata a 6 livelli dell'architettura **VGG-16**.
2. **Convertire il codice esportato in un Object Detector `.pt` nativo** compatibile con la piattaforma [Ultralytics](https://platform.ultralytics.com/) ed il comando `yolo train`: trasferimento dei pesi pre-addestrati da ImageNet, rimozione dei layer densi di classificazione e collegamento diretto a una **YOLO Detect Head** multi-scala (C3, C4, C5) senza neck intermedio.
3. **Eseguire il Fine-Tuning del modello** sia sulla piattaforma cloud [Ultralytics HUB / Platform](https://platform.ultralytics.com/) sia in locale tramite terminale.

---

## 📁 Struttura della Codebase

```
VGG for school/
│
├── blockdl-main/                # Web app didattica interattiva a 6 livelli con validazione progressiva
│   ├── src/                     # Sorgenti React + Canvas + Validatore integrato VGG-16
│   └── vite-detector-plugin.ts  # Plugin Vite con API per lanciare in automatico la conversione .pt
│
├── VGG16_Ultralytics_Colab_Training.ipynb # Notebook Google Colab per fine-tuning guidato con TPU/GPU
├── colab_assets/                # Screenshot e risorse visive per la guida Colab
├── run.py                       # Mini-server locale & launcher Python (zero Node.js richiesto)
├── convert_vgg_to_detector.py   # Script coordinatore principale: valida BlockDL e genera il detector
├── vgg_validator.py             # Modulo di validazione rigorosa VGG-16 e diagnostica didattica BlockDL
├── detector_builder.py          # Costruttore dell'Object Detector Ultralytics (ImageNet + YOLO Detect Head)
├── weights_vault.py             # Sistema di protezione e packaging Anti-Cheat per i pesi pre-allenati
├── reset_environment.py         # Script per il reset totale e pulizia anti-cheat dell'ambiente
├── requirements.txt             # Dipendenze Python (PyTorch, Ultralytics, PyYAML)
├── README.md                    # Questa guida
│
└── network to convert/          # Cartella dove viene salvato il file .py esportato da BlockDL
    └── vgg16_blockdl.py         # File di riferimento VGG-16 generato da BlockDL
```

---

## 🛠️ Prerequisiti & Avvio Immediato (Zero Node.js per gli studenti)

L'interfaccia grafica didattica BlockDL è ora **pre-compilata** all'interno del progetto.  
**Agli studenti serve unicamente Python**: non è necessario installare Node.js né eseguire `npm install` o comandi Node!

### 1. Installazione Dipendenze Python
Assicurarsi di avere **Python 3.9+** installato. Dalla cartella principale del progetto (`VGG for school`):
```bash
pip install -r requirements.txt
```
*(Installa PyTorch, Ultralytics e PyYAML necessari per la validazione della rete e la conversione in `.pt`).*

### 2. Avvio con Singolo Comando
Dalla cartella principale del progetto, avviare il laboratorio didattico con:
```bash
python run.py
```
* **Apertura automatica**: il browser predefinito si apre direttamente su `http://localhost:5173`.
* **Zero dipendenze esterne**: il server interno Python (basato sulla libreria standard) serve i file statici del frontend e gestisce le API locali di validazione e generazione automatica dell'Object Detector (`vgg16_detector.pt`).
* Per arrestare il server è sufficiente premere `Ctrl+C` nel terminale.

> ℹ️ **Nota per docenti o sviluppatori (opzionale):**  
> Se si desidera apportare modifiche al codice sorgente React/TypeScript del frontend, la cartella `blockdl-main/` contiene tutti i sorgenti originali. In quel caso (e solo per sviluppo frontend) è possibile usare Node.js con `cd blockdl-main && npm install` e lanciare `npm run dev` oppure ricompilare con `npm run build`.

---

## 🚀 Da VGG-16 (BlockDL) a YOLO Object Detector

### 1. Costruzione Interattiva (BlockDL Progressive Sandbox)

Gli studenti avviano il laboratorio con:

```bash
python run.py
```

Nel browser (all'indirizzo `http://localhost:5173`):
1. Gli studenti costruiscono la rete blocco per blocco (Livello 1: C1 $\to$ Livello 2: C2 $\to$ Livello 3: C3 $\to$ Livello 4: C4 $\to$ Livello 5: C5 $\to$ Livello 6: Classificatore FC).
2. La sandbox valida automaticamente ciascun livello evidenziando eventuali incongruenze graficamente rispetto al blueprint VGG-16.
3. **Completamento del Livello 6**: una volta completata e verificata l'intera architettura, la sandbox:
   - Salva il codice PyTorch in `network to convert/vgg16_blockdl.py`.
   - **Avvia automaticamente** la trasformazione in Object Detector Ultralytics (`vgg16_detector.pt`).
   - Mostra agli studenti l'adattamento della loro rete (rimozione dei layer densi finali, collegamento delle feature map multi-scala C3-C5 alla YOLO Detect Head, e calibrazione pesi ImageNet per il transfer learning).
   - Fornisce il pulsante **Scarica .pt** e la guida pratica con i parametri ideali di fine-tuning per proseguire l'addestramento su Ultralytics Platform!

*(In alternativa, è sempre possibile lanciare la conversione da terminale con `python convert_vgg_to_detector.py`).*

---

### 2. Addestramento dell'Object Detector su Ultralytics

#### 💡 Guida al Fine-Tuning: Quanti parametri bloccare (`freeze`)?

Quando si esegue il **fine-tuning** su un nuovo dataset, la rete non deve essere addestrata tutta da zero:

* **Riepilogo dei parametri del modello:**
  * **Parametri totali:** `20,367,616` (~20.4M)
  * **Backbone VGG-16 (`model.0`):** `14,723,136` parametri (**72.3%** dell'intera rete) $\to$ **da congelare**
  * **YOLO Detect Head (`model.4`):** `5,644,480` parametri (**27.7%** dell'intera rete) $\to$ **da addestrare**

* **Il numero ideale di layer da bloccare (`freeze`):**
  Nel riquadro **"Advanced Settings"** di Ultralytics Platform (o aggiungendo `freeze=1` da riga di comando), impostare:
  ```yaml
  freeze: 1
  ```
  * **Cosa fa:** blocca il blocco iniziale (`model.0`), congelando tutti i **14,723,136 parametri** del backbone sequenziale VGG-16 (le feature visive già apprese da ImageNet). In questo modo, l'addestramento si concentra unicamente sui **5,644,480 parametri** della testa di rilevamento YOLO!

* **🎮 Spazio alla sperimentazione: gioca con gli iperparametri!**
  Gli studenti sono incoraggiati a provare diverse combinazioni nella finestra di training:
  * **`epochs`**: prova a impostare 20–30 epoche invece delle 100 predefinite di default.
  * **`lr0` (Learning Rate Iniziale)**: prova a ridurlo da `0.01` a `0.001` per osservare come cambia la stabilità della curva di perdita (*loss*).
  * **`batch`**: sperimenta con batch size di 8, 16 oppure `-1` (calcolo automatico in base alla VRAM).
  * **Data Augmentation**: sperimenta attivando o disattivando mosaic, flip o variazioni cromatiche (HSV).

---

#### Su Ultralytics Platform (Cloud):
* **Su Ubuntu / Linux (Bash):**
  ```bash
  export ULTRALYTICS_API_KEY="tua_api_key_ultralytics"
  yolo train model=vgg16_detector.pt data=ul://tuo-dataset task=detect imgsz=640 freeze=1 project=tuo-progetto
  ```

* **Su Windows (PowerShell):**
  ```powershell
  $env:ULTRALYTICS_API_KEY="tua_api_key_ultralytics"
  yolo train model=vgg16_detector.pt data=ul://tuo-dataset task=detect imgsz=640 freeze=1 project=tuo-progetto
  ```

* **Su Windows (Prompt dei comandi CMD):**
  ```cmd
  set ULTRALYTICS_API_KEY=tua_api_key_ultralytics
  yolo train model=vgg16_detector.pt data=ul://tuo-dataset task=detect imgsz=640 freeze=1 project=tuo-progetto
  ```

#### In locale offline (es. con dataset di test coco8):
```bash
yolo train model=vgg16_detector.pt data=coco8.yaml task=detect imgsz=640 freeze=1 epochs=15
```

#### Su Google Colab (con accelerazione TPU v5e / GPU T4):
È disponibile il notebook interattivo pre-configurato **`VGG16_Ultralytics_Colab_Training.ipynb`**:
* **Guida visiva integrata**: spiega agli studenti passo-passo come attivare la TPU in Colab e impostare `freeze: 1` nelle Advanced Settings di Ultralytics HUB.
* **Addestramento cloud collegato a HUB**: consente di eseguire il fine-tuning direttamente su Colab senza saturare la memoria del PC locale, con streaming delle metriche e salvataggio dei pesi direttamente sulla dashboard Ultralytics HUB.

---

## 👨‍🏫 Reset Totale dell'Ambiente di Laboratorio (Solo da Terminale)

Per consentire al docente di eseguire simulazioni e prove e poi **riportare l'intero ambiente allo stato iniziale pulito** prima di distribuirlo agli studenti:

```bash
python reset_environment.py
```

### Caratteristiche di Sicurezza & Operazioni:
- **Protezione da esecuzioni accidentali**: richiede una digitazione esplicita di conferma da terminale (`RESET` / `s` / `yes`), evitando cancellazioni per errore. *(Per script automatici è possibile usare `python reset_environment.py --force`).*
- **Blindatura Anti-Cheat dei Checkpoint**: i pesi pre-allenati sono protetti e offuscati in `checkpoints/.vault/model_weights.bin`, impedendo agli studenti di prenderli con drag-and-drop.
- **Pulizia Modelli**: elimina tutti i modelli `.pt` e file `.yaml` generati dalla cartella principale.
- **Svuotamento Cartella Studenti**: cancella tutti i file all'interno di `network to convert/`.
- **Pulizia Cache**: rimuove cartelle temporanee di training o download (`runs/`, `datasets/`, `__pycache__`).
- **Reset Sandbox BlockDL**: azzera la versione di reset del browser; al prossimo avvio o refresh, tutti i livelli (2..6) tornano bloccati e il canvas si riapre pulito sul Livello 1.

*(Disponibile anche dalla cartella `blockdl-main` tramite il comando `npm run reset:all`).*
