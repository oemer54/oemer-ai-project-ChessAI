"""
train_model.py
--------------
Trainiert ein ML-Modell (Gradient Boosting) auf den Stockfish-Trainingsdaten.
Features: Material-Zählung aus der FEN-Stellung.

Verwendung:
    python train_model.py

Voraussetzung:
    training_data.csv (erzeugt von data_generator.py)

Ausgabe:
    model.pkl       (trainiertes Modell)
    scaler.pkl      (Feature-Scaler)
"""

import chess
import pandas as pd
import numpy as np
import pickle
import os

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score


# ──────────────────────────────────────────────
# FEATURE ENGINEERING
# ──────────────────────────────────────────────

# Figurenwerte (in Centipawns)
PIECE_VALUES = {
    chess.PAWN:   100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK:   500,
    chess.QUEEN:  900,
    chess.KING:   0,
}

def fen_to_features(fen: str) -> list[float]:
    """
    Extrahiert numerische Features aus einem FEN-String.

    Features (insgesamt 25):
      0–11  Figurenanzahl (6 weiß, 6 schwarz)
      12    Materialvorteil (weiß - schwarz) in Centipawns
      13    Anzahl legaler Züge (Mobilität) für den am Zug befindlichen Spieler
      14    Anzahl Freibauern weiß
      15    Anzahl Freibauern schwarz
      16    Weiß im Schach (0/1)
      17    Schwarz im Schach (0/1)
      18    Figurenanzahl gesamt (Spielphase)
      19    Rochade-Rechte weiß Königsseite (0/1)
      20    Rochade-Rechte weiß Damenseite (0/1)
      21    Rochade-Rechte schwarz Königsseite (0/1)
      22    Rochade-Rechte schwarz Damenseite (0/1)
      23    Am Zug: Weiß=1, Schwarz=-1
      24    Doppelbauern weiß (negativ)
    """
    board = chess.Board(fen)
    features = []

    # Figurenanzahl pro Typ und Farbe
    piece_counts = {}
    for color in [chess.WHITE, chess.BLACK]:
        for ptype in [chess.PAWN, chess.KNIGHT, chess.BISHOP,
                      chess.ROOK, chess.QUEEN, chess.KING]:
            count = len(board.pieces(ptype, color))
            piece_counts[(color, ptype)] = count
            features.append(count)

    # Materialvorteil in Centipawns
    white_material = sum(PIECE_VALUES[p] * piece_counts[(chess.WHITE, p)]
                         for p in PIECE_VALUES)
    black_material = sum(PIECE_VALUES[p] * piece_counts[(chess.BLACK, p)]
                         for p in PIECE_VALUES)
    features.append(white_material - black_material)

    # Mobilität (legale Züge des aktuellen Spielers)
    features.append(len(list(board.legal_moves)))

    # Freibauern zählen
    def count_passed_pawns(color):
        count = 0
        enemy = not color
        pawns = board.pieces(chess.PAWN, color)
        enemy_pawns = board.pieces(chess.PAWN, enemy)
        for sq in pawns:
            file = chess.square_file(sq)
            rank = chess.square_rank(sq)
            passed = True
            for esq in enemy_pawns:
                ef = chess.square_file(esq)
                er = chess.square_rank(esq)
                if abs(ef - file) <= 1:
                    if color == chess.WHITE and er > rank:
                        passed = False
                        break
                    elif color == chess.BLACK and er < rank:
                        passed = False
                        break
            if passed:
                count += 1
        return count

    features.append(count_passed_pawns(chess.WHITE))
    features.append(count_passed_pawns(chess.BLACK))

    # Schach
    features.append(1 if board.is_check() and board.turn == chess.BLACK else 0)
    features.append(1 if board.is_check() and board.turn == chess.WHITE else 0)

    # Gesamtfigurenanzahl (Spielphase)
    total_pieces = sum(len(board.pieces(p, c))
                       for p in chess.PIECE_TYPES
                       for c in [chess.WHITE, chess.BLACK])
    features.append(total_pieces)

    # Rochade-Rechte
    features.append(1 if board.has_kingside_castling_rights(chess.WHITE) else 0)
    features.append(1 if board.has_queenside_castling_rights(chess.WHITE) else 0)
    features.append(1 if board.has_kingside_castling_rights(chess.BLACK) else 0)
    features.append(1 if board.has_queenside_castling_rights(chess.BLACK) else 0)

    # Am Zug
    features.append(1 if board.turn == chess.WHITE else -1)

    # Doppelbauern weiß
    doubled = 0
    for file in range(8):
        pawns_on_file = sum(1 for sq in board.pieces(chess.PAWN, chess.WHITE)
                            if chess.square_file(sq) == file)
        if pawns_on_file > 1:
            doubled += pawns_on_file - 1
    features.append(doubled)

    return features


# ──────────────────────────────────────────────
# TRAINING
# ──────────────────────────────────────────────
def load_data(filepath: str):
    df = pd.read_csv(filepath)
    print(f"[OK] {len(df)} Einträge geladen.")

    # Ausreißer entfernen (Mattwerte)
    df = df[df["score_cp"].between(-3000, 3000)]
    print(f"[OK] Nach Filterung: {len(df)} Einträge.")

    print("→ Extrahiere Features (kann einen Moment dauern)...")
    X = np.array([fen_to_features(fen) for fen in df["fen"]])
    y = df["score_cp"].values
    return X, y


def train(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print("→ Trainiere Gradient Boosting Modell...")
    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)
    print(f"\n── Modell-Evaluation ──────────────────")
    print(f"  MAE  (Ø Fehler): {mae:.1f} Centipawns")
    print(f"  R²   (Güte):     {r2:.3f}  (1.0 = perfekt)")

    return model, scaler


def save_model(model, scaler):
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print("\n[OK] Modell gespeichert: model.pkl, scaler.pkl")


# ──────────────────────────────────────────────
# HAUPTPROGRAMM
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  ChessAI – Modell-Training")
    print("=" * 50)

    CSV_FILE = "training_data.csv"
    if not os.path.exists(CSV_FILE):
        print(f"[FEHLER] '{CSV_FILE}' nicht gefunden.")
        print("  → Zuerst ausführen: python data_generator.py")
        exit(1)

    X, y = load_data(CSV_FILE)
    model, scaler = train(X, y)
    save_model(model, scaler)
    print("\n✓ Fertig! Weiter mit: python chess_ai.py")
