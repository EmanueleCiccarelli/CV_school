# Pre-Training Pipeline per VGG16-BN + Multi-Scale Detect Head
### (100% Compatibile con Ultralytics Platform e CLI `yolo train`)

Pipeline autonoma e modulare per il pre-training a due stadi dell'Object Detector basato su **VGG16-BN** con la stessa architettura **multi-scala nativa di Ultralytics** utilizzata per gli studenti.

---

## 1. Architettura Unificata del Rilevatore

La rete (definita in [`model.py`](model.py) e sincronizzata con [`detector_builder.py`](../detector_builder.py)) utilizza l'architettura nativa `DetectionModel` di Ultralytics:

1. **Backbone (`vgg16_bn.features`)**:
   - 44 layer totali (13 `Conv2d`, 13 `BatchNorm2d`, 13 `ReLU`, 5 `MaxPool2d`).
   - Pesi pre-addestrati ImageNet (`DEFAULT`).
   - Disattivati `avgpool` e i layer densi di classificazione.

2. **Feature Maps Multi-Scala**:
   - **C3** (layer 24): uscita MaxPool blocco 3 (stride 8, 256 canali) per oggetti **piccoli**.
   - **C4** (layer 34): uscita MaxPool blocco 4 (stride 16, 512 canali) per oggetti **medi**.
   - **C5** (layer 44): uscita MaxPool blocco 5 (stride 32, 512 canali) per oggetti **grandi**.

3. **Testa Multi-Scala `Detect` (con blocchi convoluzionali intermedi 3x3)**:
   - **Ramo coordinate `cv2`**: per ciascuna delle 3 scale, include **due blocchi convoluzionali intermedi 3x3 Conv-BN** seguiti dalla convoluzione 1x1 finale.
   - **Ramo classi `cv3`**: per ciascuna delle 3 scale, include **due blocchi convoluzionali intermedi 3x3 Conv-BN** seguiti dalla proiezione 1x1 a $nc$ classi.
   - **Modulo DFL (Distribution Focal Loss)**: per una stima continua e accurata dei margini dei box.

> **Vantaggio Fondamentale**: il checkpoint `.pt` generato è nativamente un modello Ultralytics: gli studenti possono trascinarlo direttamente su **Ultralytics Platform** o addestrarlo via terminale con `yolo train model=vgg16_detector.pt data=...` con zero problemi di compatibilità!

---

## 2. Strategia di Addestramento a Due Stadi

Modulo: [`train.py`](train.py)

### Protezione del BatchNorm Congelato (`Frozen BatchNorm Protection`)
Ad ogni epoca, un callback dedicato forza tutti i layer `BatchNorm2d` dei blocchi congelati in modalità `.eval()`, impedendo a `running_mean` e `running_var` di alterare le statistiche calibrate di ImageNet durante il batching di detection.

### Stage 1: Head Warm-up (Default: 5 epoche)
- **Backbone [0:44]**: 100% congelato (`freeze=1`).
- **BatchNorm**: tutti i 13 layer BN del backbone in `.eval()`.
- **Ottimizzatore**: AdamW sui parametri della testa `Detect` (`lr0 = 1e-3`, `weight_decay = 1e-4`).
- **Obiettivo**: calibrare le convoluzioni 3x3 della testa e il modulo DFL prima di toccare i filtri del backbone.

### Stage 2: Partial Backbone Fine-Tuning (Default: 25 epoche)
- **Backbone [0:34] (conv1 .. conv4)**: 100% congelato con BatchNorm in `.eval()`.
- **Backbone [34:44] (blocco conv5)**: scongelato per specializzare i filtri convoluzionali di alto livello.
- **Testa Detect**: scongelata.
- **Parametri di apprendimento**:
  - `lr0 = 1e-4`, decadimento con `CosineAnnealingLR` (`cos_lr=True`, `lrf=0.01`) fino a `1e-6`.
  - Mixed Precision con `torch.cuda.amp` abilitata di default.

---

## 3. Chirurgia dei Pesi per Classi Arbitrarie

Modulo: [`export_checkpoint.py`](export_checkpoint.py)

La funzione `prepare_custom_class_checkpoint()` permette di convertire il checkpoint pre-allenato (es. 20 classi Pascal VOC o 80 classi COCO) per un nuovo numero di classi $N$ (es. 2 classi):

1. **Copia 100% dei pesi di localizzazione e feature**:
   - 170 tensori su 176 vengono copiati identicamente: tutto il backbone `vgg16_bn`, l'intero ramo coordinate `cv2` (tutte le 3 scale), il layer `dfl` e tutti i blocchi intermedi 3x3 del ramo classificazione `cv3`.
2. **Re-inizializzazione stabile della sola proiezione classi 1x1**:
   - Solo i 6 tensori finali (`cv3.*.2.weight` e `cv3.*.2.bias` per le 3 scale) vengono re-inizializzati per $N$ classi.
   - Ai bias viene assegnato il **prior logit** $b = \ln(p / (1 - p)) \approx -4.595$ ($p = 0.01$), garantendo convergenza immediata e prevenendo gradienti esplosivi nella BCE loss.
3. **Aggiornamento metadati**:
   - Aggiorna `nc`, `model.names` e `model.yaml['nc']` nel checkpoint `.pt`.

---

## 4. Integrazione Trasparente con il Flusso Studenti

Nel modulo [`detector_builder.py`](../detector_builder.py):
- Quando uno studente completa il Livello 6 di BlockDL, `build_and_export_detector()` verifica automaticamente se è presente un checkpoint pre-addestrato (es. `checkpoints/best.pt`).
- Se presente, inietta automaticamente i pesi pre-allenati del backbone e della testa YOLO nel modello dello studente.
- **Pedagogia preservata ("White Lie")**: per gli studenti il messaggio di output rimane pulito e coerente con la VGG16, indicando che la loro rete è stata calibrata e arricchita con Transfer Learning per il training immediato su Ultralytics Platform.

---

## 5. Guida ai Comandi Rapidi

### Test Veloce (Dry-Run in ~15 secondi su dataset test coco8)
```bash
python pretrain_pipeline/train.py --dry-run
```

### Avvio Pre-Training Completo su MS COCO (Default: 80 classi)
```bash
python pretrain_pipeline/train.py
# Oppure specificando esplicitamente il dataset:
python pretrain_pipeline/train.py --data coco.yaml
```

### Alternativa Rapida: Pascal VOC (20 classi, completamento in ~1 ora)
```bash
python pretrain_pipeline/train.py --data VOC.yaml
```

### Esecuzione di un Singolo Stadio
```bash
# Esegui solo Stage 1 (Warm-up della testa per 10 epoche)
python pretrain_pipeline/train.py --stage 1 --epochs-stage1 10

# Esegui solo Stage 2 (Fine-tuning per 30 epoche)
python pretrain_pipeline/train.py --stage 2 --epochs-stage2 30
```

### Adattamento Checkpoint a un Dataset Personalizzato (es. 2 classi)
```bash
python pretrain_pipeline/export_checkpoint.py \
    --input checkpoints/best.pt \
    --output checkpoints/vgg16_detector_custom_2cls.pt \
    --classes 2
```
