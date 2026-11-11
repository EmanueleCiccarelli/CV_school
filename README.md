# Laboratorio di Computer Vision: da BlockDL a Ultralytics YOLO

Questo progetto è strutturato per un laboratorio didattico di Computer Vision (ad esempio per una classe di quinta liceo).  
L'obiettivo è consentire agli studenti di:
1. **Costruire visivamente una rete neurale a blocchi** tramite [BlockDL](https://blockdl.com/editor).
2. **Convertire il codice esportato in un modello `.pt` nativo** compatibile con la piattaforma [Ultralytics](https://platform.ultralytics.com/) ed il comando `yolo train`.
3. **Addestrare il modello** (in locale o sincronizzato con il cloud Ultralytics) su compiti di classificazione di immagini:
   - *Fase 1 (introduttiva)*: CNN semplice su versione ridotta di MNIST (discriminazione binaria cifre 1 e 7).
   - *Fase 2 (avanzata)*: Costruzione manuale e training di una **VGG-16** su dataset personalizzato.

---

## 📁 Struttura della Codebase

```
VGG for school/
│
├── main.py                     # Script principale per gli studenti: esegue la conversione con 1 comando
├── convert_to_ultralytics.py   # Engine di conversione (AST Sanitizer, FX Parser, Ultralytics Exporter)
├── requirements.txt            # Dipendenze necessarie (PyTorch, Ultralytics, PyYAML)
├── README.md                   # Questa guida
│
└── network to convert/         # Cartella dove gli studenti inseriscono il file .py esportato da BlockDL
    └── pytorch_model_sequential (1).py
```

---

## 🚀 Guida Rapida (Quick Start)

### 1. Installazione Requisiti

Funziona sia su **Ubuntu (Linux)** sia su **Windows** (PowerShell o CMD).

```bash
pip install -r requirements.txt
```

> **Nota per aule/laboratori con sole CPU (senza scheda NVIDIA):**
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
> pip install ultralytics pyyaml
> ```

---

### 2. Disegnare ed Esportare la Rete da BlockDL

1. Aprire l'editor grafico su [BlockDL](https://blockdl.com/editor).
2. Comporre i layer (es. Conv2D, MaxPool2D, Flatten, Linear).
3. Cliccare su **Export -> PyTorch** e scaricare il file Python (es. `pytorch_model_sequential.py`).
4. Spostare il file scaricato dentro la cartella:
   ```
   network to convert/
   ```

---

### 3. Conversione in Modello Ultralytics (.pt)

Basta eseguire da terminale:

```bash
python main.py
```

Lo script:
- Seleziona in automatico il primo file presente in `network to convert/`.
- Neutralizza le dimensioni fittizie dei layer densi (`nn.Linear`) ricalcolandole dinamicamente.
- Rimuove eventuali `log_softmax` o `softmax` finali (Ultralytics calcola la `CrossEntropyLoss` direttamente sui logits).
- Adatta il primo layer convoluzionale a 3 canali RGB (standard di Ultralytics).
- Salva il modello pronto: `model.pt` (e la sua architettura `model.yaml`).

---

## 🏋️ Addestramento con Ultralytics

Al termine della conversione, `main.py` mostra a schermo i comandi esatti da copiare e incollare.

### Opzione A: Con Ultralytics Platform (Cloud & Dashboard)

#### Su Ubuntu / Linux (Bash):
```bash
export ULTRALYTICS_API_KEY="la_tua_api_key_fornita_da_ultralytics"
yolo train model=model.pt data=ul://emanuele-ciccarelli/tuo-dataset task=classify imgsz=28 project=emanuele-ciccarelli/progetto
```

#### Su Windows (PowerShell):
```powershell
$env:ULTRALYTICS_API_KEY="la_tua_api_key_fornita_da_ultralytics"
yolo train model=model.pt data=ul://emanuele-ciccarelli/tuo-dataset task=classify imgsz=28 project=emanuele-ciccarelli/progetto
```

#### Su Windows (Prompt dei comandi - CMD):
```cmd
set ULTRALYTICS_API_KEY=la_tua_api_key_fornita_da_ultralytics
yolo train model=model.pt data=ul://emanuele-ciccarelli/tuo-dataset task=classify imgsz=28 project=emanuele-ciccarelli/progetto
```

---

### Opzione B: Addestramento Locale con Dataset su Disco

Organizzare le cartelle delle immagini in questo modo:
```
dataset/
├── train/
│   ├── 1/
│   └── 7/
└── val/
    ├── 1/
    └── 7/
```

Avviare il comando:
```bash
yolo train model=model.pt data=dataset task=classify imgsz=28 epochs=10
```

---

## ⚙️ Parametri Opzionali di `main.py`

Sebbene `python main.py` funzioni con zero argomenti, sono disponibili flag opzionali per scenari avanzati:

- **Specificare un file particolare**:
  ```bash
  python main.py --file "network to convert/mia_rete.py"
  ```
- **Risoluzione immagine per VGG-16 (default 28 per MNIST)**:
  ```bash
  python main.py --imgsz 224
  ```
- **Personalizzare il nome del file di output**:
  ```bash
  python main.py --output "vgg16_custom.pt"
  ```
- **Specificare i nomi delle classi**:
  ```bash
  python main.py --classes uno sette
  ```

---

## 💡 Perché questo Engine è Robusto per gli Studenti?

1. **Nessun calcolo manuale del Flatten**: In BlockDL gli studenti spesso sbagliano il calcolo dell'ingresso di `nn.Linear` (ad esempio inserendo numeri a caso come `100352`). L'AST sanitizer e il calcolatore dinamico determinano i canali esatti in automatico.
2. **Compatibilità di loss**: Rimuove automaticamente `log_softmax` o `softmax` finali inseriti per abitudine dagli studenti, prevenendo errori di calcolo gradiente con Ultralytics.
3. **Immunità alla variazione di risoluzione**: Inserisce un layer `AdaptiveAvgPool2d` prima del `Flatten`, garantendo che la rete non vada in errore anche se Ultralytics adatta la risoluzione (es. da 28 a 32 pixel).
4. **Cross-Platform nativo**: Usa la libreria standard `pathlib.Path` per gestire i percorsi in modo trasparente sia con i separatori `/` di Linux/Ubuntu sia con `\` di Windows.

