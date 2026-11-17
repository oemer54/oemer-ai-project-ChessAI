"""
data_generator.py
-----------------
Generiert Trainingsdaten mit Stockfish.
Verbesserung: Stockfish spielt zuerst den besten Gegenzug, dann wird die
entstehende Stellung bewertet. So lernt das Modell einen Zug tiefer zu denken.

Verwendung:
    python data_generator.py

Ausgabe:
    training_data.csv  (FEN, Bewertung in Centipawns)
"""

import chess
import chess.engine
import chess.pgn
import csv
import random
import os
import sys

# ──────────────────────────────────────────────
# KONFIGURATION
# ──────────────────────────────────────────────
STOCKFISH_PATH = "stockfish"   # Pfad zu Stockfish (muss installiert sein)
OUTPUT_FILE    = "training_data.csv"
NUM_POSITIONS  = 2000          # Anzahl der zu generierenden Stellungen
DEPTH          = 10            # Analysetiefe für Stockfish
TIME_LIMIT     = 0.05          # Sekunden pro Stellung


def random_fen_positions(n: int) -> list[str]:
    """Erzeugt n zufällige Schachstellungen via zufällige Züge."""
    positions = []
    while len(positions) < n:
        board = chess.Board()
        # Zufällig 5–40 Züge spielen
        num_moves = random.randint(5, 40)
        for _ in range(num_moves):
            if board.is_game_over():
                break
            move = random.choice(list(board.legal_moves))
            board.push(move)
        if not board.is_game_over():
            positions.append(board.fen())
    return positions


def evaluate_positions(fens: list[str], stockfish_path: str) -> list[tuple[str, float]]:
    """
    Bewertet jede Stellung mit Stockfish – aber erst NACH dem besten Gegenzug.
    So lernt das Modell einen Zug tiefer zu denken (ohne Minimax).
    """
    data = []
    try:
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    except FileNotFoundError:
        print(f"[FEHLER] Stockfish nicht gefunden unter: '{stockfish_path}'")
        print("  → Bitte Stockfish installieren: https://stockfishchess.org/download/")
        print("  → Oder Pfad in STOCKFISH_PATH anpassen.")
        sys.exit(1)

    total = len(fens)
    for i, fen in enumerate(fens):
        board = chess.Board(fen)

        # Schritt 1: Besten Zug für den aktuellen Spieler finden
        result = engine.play(board, chess.engine.Limit(depth=DEPTH, time=TIME_LIMIT))
        best_move = result.move

        # Schritt 2: Besten Zug spielen
        if best_move and best_move in board.legal_moves:
            board.push(best_move)

        # Schritt 3: Falls Spiel noch läuft → besten Gegenzug spielen
        if not board.is_game_over() and len(list(board.legal_moves)) > 0:
            reply = engine.play(board, chess.engine.Limit(depth=DEPTH, time=TIME_LIMIT))
            if reply.move and reply.move in board.legal_moves:
                board.push(reply.move)

        # Schritt 4: Entstehende Stellung bewerten
        if board.is_game_over():
            outcome = board.outcome()
            if outcome and outcome.winner == chess.WHITE:
                cp = 10000
            elif outcome and outcome.winner == chess.BLACK:
                cp = -10000
            else:
                cp = 0
        else:
            info = engine.analyse(board, chess.engine.Limit(depth=DEPTH, time=TIME_LIMIT))
            score = info["score"].white()
            if score.is_mate():
                cp = 10000 if score.mate() > 0 else -10000
            else:
                cp = score.score()

        # Original-FEN speichern (nicht die nach den Zügen!)
        data.append((fen, cp))

        # Fortschritt anzeigen
        if (i + 1) % 100 == 0 or (i + 1) == total:
            print(f"  {i+1}/{total} Stellungen bewertet...")

    engine.quit()
    return data


def save_csv(data: list[tuple[str, float]], filepath: str):
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fen", "score_cp"])
        writer.writerows(data)
    print(f"[OK] {len(data)} Stellungen gespeichert in '{filepath}'")


# ──────────────────────────────────────────────
# HAUPTPROGRAMM
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  ChessAI – Trainingsdaten-Generator")
    print("=" * 50)
    print(f"\n→ Erzeuge {NUM_POSITIONS} zufällige Schachstellungen...")
    fens = random_fen_positions(NUM_POSITIONS)

    print(f"→ Bewerte Stellungen mit Stockfish (Tiefe {DEPTH})...")
    data = evaluate_positions(fens, STOCKFISH_PATH)

    save_csv(data, OUTPUT_FILE)
    print("\n✓ Fertig! Weiter mit: python train_model.py")