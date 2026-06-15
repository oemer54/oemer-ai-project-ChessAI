# ♟ ChessAI – KI zur Bewertung von Schachstellungen

Uni-Projekt von Ömer Öksüz  
Kurs: Projektmanagement / Advanced AI and Data Science

---

## Projektbeschreibung

Die KI bewertet Schachstellungen anhand eines FEN-Strings und empfiehlt den besten Zug.  
Das Modell wird mit Stockfish-Bewertungen trainiert (Scikit-learn, Gradient Boosting).

**Workflow:**
```
Lichess → FEN kopieren → Terminal → KI gibt Bewertung + Empfehlung aus
```

---

## Installation

### 1. Python-Pakete installieren

```bash
pip install python-chess scikit-learn pandas numpy
```

### 2. Stockfish installieren

**Windows:**
- https://stockfishchess.org/download/ → Windows-Version herunterladen
- `stockfish.exe` in denselben Ordner wie die Python-Dateien legen
- In `data_generator.py` den Pfad anpassen: `STOCKFISH_PATH = "stockfish.exe"`

**Linux/Mac:**
```bash
# Ubuntu/Debian
sudo apt install stockfish

# Mac
brew install stockfish
```

---

## Verwendung

### Schritt 1: Trainingsdaten generieren
```bash
python data_generator.py
```
→ Erzeugt `training_data.csv` (ca. 5–10 Minuten)

### Schritt 2: Modell trainieren
```bash
python train_model.py
```
→ Erzeugt `model.pkl` und `scaler.pkl`

### Schritt 3: KI verwenden
```bash
python chess_ai.py
```

---

## Beispiel-Session

```
FEN oder Befehl> rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1

    a  b  c  d  e  f  g  h
  ┌─────────────────────────┐
8 │ ♜  ♞  ♝  ♛  ♚  ♝  ♞  ♜ │ 8
7 │ ♟  ♟  ♟  ♟  ♟  ♟  ♟  ♟ │ 7
...

Am Zug: Schwarz
→ Analysiere alle möglichen Züge...

─────────────────────────────────────────────
  Empfehlung:  e7 -> e5
  Bewertung:   +0.31 (Weiß hat kleinen Vorteil)
─────────────────────────────────────────────
```

### FEN von Lichess holen

1. https://lichess.org/editor aufrufen
2. Stellung aufbauen
3. Unten den FEN-String kopieren
4. In das Terminal einfügen

---

## Dateistruktur

```
chessai/
├── data_generator.py   # Trainingsdaten mit Stockfish erzeugen
├── train_model.py      # ML-Modell trainieren
├── chess_ai.py         # Hauptprogramm (Terminal)
├── README.md           # Diese Datei
├── training_data.csv   # (wird generiert)
├── model.pkl           # (wird generiert)
└── scaler.pkl          # (wird generiert)
```

---

## Technologie

| Tool | Zweck |
|------|-------|
| Python | Programmiersprache |
| python-chess | Schach-Logik & FEN-Parsing |
| Stockfish | Trainingsdaten-Generierung |
| Scikit-learn | ML-Modell (Gradient Boosting) |
| Lichess | FEN-Strings generieren |

---

## Nicht-Ziele (bewusst ausgelassen)

- ❌ Vollständige Schach-Engine
- ❌ Reinforcement Learning
- ❌ Professionelle Turnier-KI
- ❌ Komplexes Deep Learning
