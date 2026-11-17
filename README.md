# ♟ ChessAI

Uni-Projekt von Ömer Öksüz (Advanced AI and Data Science).
Frage: **Kann ein Transformer Schachzüge finden, ohne dass eine Suche läuft?**

FEN-String (z.B. von Lichess) ins Terminal einfügen, die KI empfiehlt einen Zug.

## Zwei Ansätze im Vergleich

| | Alt (Semester 1) | Neu (Semester 2) |
|---|---|---|
| Modell lernt | Stellung -> Bewertung (Zahl) | Stellung -> Zug (Wahrscheinlichkeit je Zug) |
| Eingabe | 25 selbst gebaute Features | Brett als 68 Tokens (64 Felder + 4 Rochade-Flags) |
| Zugwahl | Minimax (Tiefe 3) probiert Züge durch | direkt, eine Vorhersage (oder Hybrid mit Suche, siehe unten) |
| Dateien | `train_model.py` | `train_models.py`, `chess_model.py` |

Neu werden zwei Modelle auf denselben Daten trainiert und verglichen:
**MLP** (einfaches Netz, Baseline) und **Transformer** (PyTorch `TransformerEncoder`).
Gemessen wird Top-1 und Top-5 Accuracy auf getrennten Testdaten.

Details der Umsetzung:
- Ist Schwarz am Zug, wird das Brett gespiegelt. Das Modell sieht immer "Weiß am Zug".
- Zug-Vokabular: 64 x 64 = 4096 (Startfeld, Zielfeld). Bauernumwandlung immer zur Dame.
- Legal-Move-Masking: illegale Züge werden vor dem Softmax gesperrt (Training und Vorhersage).
- **Hybrid** (`mode hybrid`, Standard im Terminal): Der Transformer liefert die 6 wahrscheinlichsten Züge, dazu kommen alle Schachgebote. Jeder Kandidat wird mit Alpha-Beta-Suche (Tiefe 4, Materialbewertung, Schlagzug-Nachsuche) nachgerechnet. So wird z. B. ein Matt in 1 nie übersehen.
- Transformer-Ausgabe (`--head pair`): ein Wert pro (Startfeld, Zielfeld)-Paar. Alternative `--head linear`: Mittelwert aller Tokens -> Linear(4096).

## Installation

```bash
python -m pip install python-chess scikit-learn pandas numpy torch matplotlib
```
Stockfish von https://stockfishchess.org/download/ herunterladen und `stockfish.exe` in den Projektordner legen (nicht im Repo, zu groß).

## Ablauf

```bash
# 1) Trainingsdaten: Stellung -> bester Stockfish-Zug (parallel, speichert laufend)
python data_generator_moves.py --n 20000        # später z.B. --n 100000, es wird ergänzt

# 2) MLP und Transformer trainieren und vergleichen
python train_models.py                          # Test vorab: --limit 5000 --epochs 2

# 3) Terminal-Programm
python chess_ai.py
```

Befehle im Programm: `<FEN>`, `mode hybrid|transformer|mlp|minimax`, `all` (alle Engines vergleichen), `depth 3` (Suchtiefe), `test` (Matt-in-1-Teststellung), `quit`.

Altes Modell (nur für `minimax`): `python data_generator.py` und `python train_model.py` erzeugen `model.pkl` und `scaler.pkl`.

## Dateien

```
chess_model.py            Tokenisierung, Modelle (MLP, Transformer), Vorhersage
data_generator_moves.py   Daten: Stellung -> Stockfish-Zug (neu)
train_models.py           Training und Vergleich MLP gegen Transformer (neu)
chess_ai.py               Terminal-Programm, alle Engines (inkl. Hybrid)
data_generator.py         Daten: Stellung -> Bewertung (alt)
train_model.py            Gradient Boosting auf 25 Features (alt)
```

## Ergebnisse

Test auf Stellungen, die das Modell nie gesehen hat (10 % der Daten). Zufälliger legaler Zug zum Vergleich: Top-1 4,2 %, Top-5 18,9 %. 5 Epochen, CPU.

| Daten | Modell | Parameter | Top-1 | Top-5 |
|---|---|---|---|---|
| 30.000 | MLP | 2,79 Mio | 27,5 % | 60,8 % |
| 30.000 | Transformer | 0,57 Mio | 39,7 % | 73,0 % |
| 100.000 | MLP | 2,79 Mio | 38,5 % | 73,9 % |
| 100.000 | Transformer | 0,57 Mio | **41,6 %** | **75,3 %** |

Beobachtungen:
- Der Transformer ist bei wenig Daten deutlich besser (+12 Prozentpunkte bei 30.000). Bei 100.000 Stellungen holt der MLP stark auf, der Transformer liegt aber weiter vorn, mit einem Fünftel der Parameter.
- Der MLP überanpasst sich (Trainingsfehler 1,66, Testfehler 2,18). Beim Transformer liegen beide nah beieinander (2,06 / 2,05).
- Matt-in-1-Test (`test`): Weder MLP noch Transformer finden Dh5xf7# allein (Transformer: Qxe5, Bxg8, Qe2). Mit Suche (Hybrid) wird es gefunden.
- Hybrid gegen den Transformer allein (Tiefe 4, 8 Partien mit zufälliger Eröffnung): 7,5 von 8 Punkten. Das ist eine kleine Stichprobe und kein Stärkemaß gegen Menschen.

Rohdaten: `results.json` (100.000), `results_30k.json` (30.000), Diagramm `results.png`.

## Nicht-Ziele

Keine vollständige Engine, kein Reinforcement Learning, kein MCTS / AlphaZero-Training.
