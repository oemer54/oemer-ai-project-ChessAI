"""
chess_ai.py
-----------
Hauptprogramm: FEN-String eingeben → KI bewertet Stellung + empfiehlt besten Zug.

Verwendung:
    python chess_ai.py

Voraussetzung:
    model.pkl und scaler.pkl (erzeugt von train_model.py)
"""

import chess
import pickle
import numpy as np
import os
import sys

# ──────────────────────────────────────────────
# FEATURE-FUNKTION (identisch zu train_model.py)
# ──────────────────────────────────────────────
PIECE_VALUES = {
    chess.PAWN:   100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK:   500,
    chess.QUEEN:  900,
    chess.KING:   0,
}

def fen_to_features(fen: str) -> list[float]:
    board = chess.Board(fen)
    features = []

    piece_counts = {}
    for color in [chess.WHITE, chess.BLACK]:
        for ptype in [chess.PAWN, chess.KNIGHT, chess.BISHOP,
                      chess.ROOK, chess.QUEEN, chess.KING]:
            count = len(board.pieces(ptype, color))
            piece_counts[(color, ptype)] = count
            features.append(count)

    white_material = sum(PIECE_VALUES[p] * piece_counts[(chess.WHITE, p)] for p in PIECE_VALUES)
    black_material = sum(PIECE_VALUES[p] * piece_counts[(chess.BLACK, p)] for p in PIECE_VALUES)
    features.append(white_material - black_material)
    features.append(len(list(board.legal_moves)))

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
                        passed = False; break
                    elif color == chess.BLACK and er < rank:
                        passed = False; break
            if passed:
                count += 1
        return count

    features.append(count_passed_pawns(chess.WHITE))
    features.append(count_passed_pawns(chess.BLACK))
    features.append(1 if board.is_check() and board.turn == chess.BLACK else 0)
    features.append(1 if board.is_check() and board.turn == chess.WHITE else 0)

    total_pieces = sum(len(board.pieces(p, c))
                       for p in chess.PIECE_TYPES
                       for c in [chess.WHITE, chess.BLACK])
    features.append(total_pieces)
    features.append(1 if board.has_kingside_castling_rights(chess.WHITE) else 0)
    features.append(1 if board.has_queenside_castling_rights(chess.WHITE) else 0)
    features.append(1 if board.has_kingside_castling_rights(chess.BLACK) else 0)
    features.append(1 if board.has_queenside_castling_rights(chess.BLACK) else 0)
    features.append(1 if board.turn == chess.WHITE else -1)

    doubled = 0
    for file in range(8):
        pawns_on_file = sum(1 for sq in board.pieces(chess.PAWN, chess.WHITE)
                            if chess.square_file(sq) == file)
        if pawns_on_file > 1:
            doubled += pawns_on_file - 1
    features.append(doubled)

    return features


# ──────────────────────────────────────────────
# HILFSFUNKTIONEN
# ──────────────────────────────────────────────
def score_to_text(score_cp: float, turn: chess.Color) -> str:
    """Erklärt die Bewertung in Worten."""
    # Score immer aus Sicht von Weiß (Centipawns)
    if abs(score_cp) < 30:
        return "Ausgeglichene Stellung"
    elif score_cp > 0:
        adv = "Weiß"
    else:
        adv = "Schwarz"

    abs_score = abs(score_cp)
    if abs_score < 100:
        strength = "kleiner Vorteil"
    elif abs_score < 300:
        strength = "klarer Vorteil"
    elif abs_score < 600:
        strength = "großer Vorteil"
    else:
        strength = "entscheidender Vorteil"

    return f"{adv} hat {strength}"


def score_display(score_cp: float) -> str:
    """Formatiert Score als +X.XX / -X.XX (in Bauern, nicht Centipawns)."""
    val = score_cp / 100
    return f"{val:+.2f}"


def best_move(board: chess.Board, model, scaler) -> tuple[chess.Move, float]:
    """
    Findet den besten Zug durch Suche aller legalen Züge.
    Bewertet jede entstehende Stellung mit dem ML-Modell.
    """
    best     = None
    best_score = -np.inf if board.turn == chess.WHITE else np.inf

    for move in board.legal_moves:
        board.push(move)
        fen = board.fen()
        board.pop()

        features = np.array([fen_to_features(fen)])
        features = scaler.transform(features)
        score    = model.predict(features)[0]

        if board.turn == chess.WHITE:
            if score > best_score:
                best_score = score
                best = move
        else:
            if score < best_score:
                best_score = score
                best = move

    return best, best_score


def move_to_readable(move: chess.Move) -> str:
    """Konvertiert einen Zug in lesbares Format: e2 -> e4"""
    src = chess.square_name(move.from_square)
    dst = chess.square_name(move.to_square)
    promo = ""
    if move.promotion:
        promo = f" ({chess.piece_name(move.promotion)})"
    return f"{src} -> {dst}{promo}"


def draw_board(board: chess.Board):
    """Einfache ASCII-Darstellung des Bretts."""
    print()
    print("    a  b  c  d  e  f  g  h")
    print("  ┌─────────────────────────┐")
    for rank in range(7, -1, -1):
        row = f"{rank+1} │"
        for file in range(8):
            sq = chess.square(file, rank)
            piece = board.piece_at(sq)
            if piece is None:
                symbol = " · "
            else:
                symbol = f" {piece.unicode_symbol()} "
            row += symbol
        row += f"│ {rank+1}"
        print(row)
    print("  └─────────────────────────┘")
    print("    a  b  c  d  e  f  g  h")
    print()


# ──────────────────────────────────────────────
# MODELL LADEN
# ──────────────────────────────────────────────
def load_model():
    for f in ["model.pkl", "scaler.pkl"]:
        if not os.path.exists(f):
            print(f"[FEHLER] '{f}' nicht gefunden.")
            print("  → Zuerst ausführen: python train_model.py")
            sys.exit(1)

    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


# ──────────────────────────────────────────────
# HAUPTSCHLEIFE
# ──────────────────────────────────────────────
HELP_TEXT = """
Befehle:
  <FEN>    FEN-String eingeben und analysieren
  new      Neue Partie (Startstellung)
  board    Aktuelles Brett anzeigen
  help     Diese Hilfe anzeigen
  quit     Programm beenden

Tipp: FEN-Strings bekommst du auf lichess.org
  → Stelle aufbauen → Share & export → FEN kopieren
"""

EXAMPLE_FENS = [
    ("Startstellung",     "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    ("Nach 1.e4",         "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"),
    ("Sizilianisch",      "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2"),
]

def main():
    print("=" * 55)
    print("  ♟  ChessAI – KI-Stellungsbewertung  ♟")
    print("=" * 55)
    print("→ Lade Modell...")
    model, scaler = load_model()
    print("✓ Modell geladen!\n")
    print(HELP_TEXT)

    current_board = chess.Board()

    while True:
        try:
            user_input = input("FEN oder Befehl> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAuf Wiedersehen!")
            break

        if not user_input:
            continue

        cmd = user_input.lower()

        # ── Befehle ──────────────────────────────
        if cmd in ("quit", "exit", "q"):
            print("Auf Wiedersehen!")
            break

        elif cmd == "help":
            print(HELP_TEXT)

        elif cmd == "new":
            current_board = chess.Board()
            print("[OK] Neue Partie gestartet (Startstellung).")
            draw_board(current_board)

        elif cmd == "board":
            draw_board(current_board)

        elif cmd == "examples":
            print("\nBeispiel-FEN-Strings:")
            for name, fen in EXAMPLE_FENS:
                print(f"  {name}:\n  {fen}\n")

        # ── FEN-Analyse ──────────────────────────
        else:
            fen = user_input
            try:
                board = chess.Board(fen)
                current_board = board
            except ValueError:
                print(f"[FEHLER] Ungültiger FEN-String: '{fen}'")
                print("  Tipp: FEN von lichess.org kopieren")
                continue

            # Brett anzeigen
            draw_board(board)

            # Spielsituation prüfen
            if board.is_checkmate():
                print("♟  SCHACHMATT! Das Spiel ist vorbei.")
                continue
            if board.is_stalemate():
                print("♟  PATT! Das Spiel ist unentschieden.")
                continue
            if board.is_insufficient_material():
                print("♟  Remis (unzureichendes Material).")
                continue

            turn_str = "Weiß" if board.turn == chess.WHITE else "Schwarz"
            print(f"Am Zug: {turn_str}")
            if board.is_check():
                print("⚠  Im Schach!")

            # Bewertung der aktuellen Stellung
            features = np.array([fen_to_features(fen)])
            features_scaled = scaler.transform(features)
            score = model.predict(features_scaled)[0]

            # Bester Zug
            print("→ Analysiere alle möglichen Züge...")
            move, move_score = best_move(board, model, scaler)

            # Ausgabe
            print()
            print("─" * 45)
            if move:
                print(f"  Empfehlung:  {move_to_readable(move)}")
            print(f"  Bewertung:   {score_display(score)} ({score_to_text(score, board.turn)})")
            print("─" * 45)
            print()
            print("  (Positive Werte = Vorteil für Weiß,")
            print("   Negative Werte = Vorteil für Schwarz)")
            print()


if __name__ == "__main__":
    main()
