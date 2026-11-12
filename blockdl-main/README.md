# BlockDL - VGG16 Progressive Sandbox Didattica

![](banner.png)

Piattaforma didattica interattiva basata su **BlockDL** per guidare studenti delle scuole superiori (quinta liceo) nella costruzione passo-passo dell'architettura **VGG16**, prevenendo errori cumulativi tramite validazione a singoli blocchi funzionali e generazione di codice **PyTorch (`nn.Module`)**.

---

## 🎯 Struttura dei Livelli Didattici

La sandbox è suddivisa in 6 livelli progressivi a sblocco sequenziale:

1. **Livello 1 (Blocco Conv C1):** Input 3×224×224 (RGB) $\to$ 2x `Conv2D(64)` + `ReLU` + `MaxPool2D(2×2)`.
2. **Livello 2 (Blocco Conv C2):** Pre-carica C1 $\to$ 2x `Conv2D(128)` + `ReLU` + `MaxPool2D(2×2)`.
3. **Livello 3 (Blocco Conv C3):** Pre-carica C1-C2 $\to$ 3x `Conv2D(256)` + `ReLU` + `MaxPool2D(2×2)`.
4. **Livello 4 (Blocco Conv C4):** Pre-carica C1-C3 $\to$ 3x `Conv2D(512)` + `ReLU` + `MaxPool2D(2×2)`.
5. **Livello 5 (Blocco Conv C5):** Pre-carica C1-C4 $\to$ 3x `Conv2D(512)` + `ReLU` + `MaxPool2D(2×2)`.
6. **Livello 6 (Classificatore FC):** Pre-carica Feature Extractor C1-C5 $\to$ `Flatten` $\to$ `Dense(4096)` + `Dropout(0.5)` $\to$ `Dense(4096)` + `Dropout(0.5)` $\to$ `Dense(1000)` / `Output(1000)`.

---

## 🚀 Avvio Rapido Locale

L'applicazione funziona interamente in locale nel browser senza dipendenze cloud obbligatorie.

### 1. Installazione Dipendenze
```bash
npm install
```

### 2. Avvio del Server di Sviluppo
```bash
npm run dev
```
Apri il browser su: **`http://localhost:5173`**

---

## 👨‍🏫 Comandi da Terminale per il Docente (Reset Ambiente)

Per evitare che gli studenti possano azzerare accidentalmente la sessione, il **Reset Globale dell'Ambiente** è disponibile **esclusivamente da riga di comando**:

### A. Reset Completo dell'Ambiente per la Classe
Riblocca tutti i livelli dal 2 al 6, cancella le cache di salvataggio nel browser e ripristina il canvas pulito al Livello 1:
```bash
npm run reset-sandbox
```
*(Da eseguire nel terminale prima di distribuire il progetto o iniziare una nuova lezione).*

### B. Reset + Avvio Immediato del Server
Esegue il reset pulito di tutti i progressi e lancia contemporaneamente il server di sviluppo:
```bash
npm run dev:reset
```

---

## 🛠️ Altri Script Utili

- **Verifica Build di Produzione:**
  ```bash
  npm run build
  ```
- **Anteprima Build:**
  ```bash
  npm run preview
  ```

---

## 📐 Reference Visive & Bounding Box

Durante l'esecuzione, gli studenti hanno a disposizione sia nel canvas (widget compatto in basso a sinistra) sia nella finestra *"Guida & Blueprint"* due reference visive ufficiali della VGG16:
- **`vgg16_blocks_refined.jpeg`**: schema a blocchi con convoluzioni, pooling e classificatore.
- **`vgg16_features_maps.webp`**: diagramma 3D dei volumi tensoriali e risoluzioni intermedie.

Su ciascuna immagine viene proiettata una **Bounding Box evidenziata** che delimita esattamente il blocco che lo studente deve costruire per superare il livello attivo.

---

## 📄 Licenza

Distribuito sotto licenza Mozilla Public License 2.0 - vedi [LICENSE.md](LICENSE.md) per i dettagli.
