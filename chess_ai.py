"""
chess_ai.py
-----------
Terminal-Programm: FEN-String eingeben -> Zugempfehlung.

Drei Engines (je nachdem, welche Modelldateien vorhanden sind):
  transformer   Transformer sagt direkt die Züge voraus     (model_transformer.pt)
  mlp           einfaches Netz, Baseline                    (model_mlp.pt)
  minimax       altes Modell (25 Features) + Minimax (Tiefe 3) (model.pkl, scaler.pkl)
  hybrid        Transformer-Kandidaten + Suche (Alpha-Beta)    (nutzt model_transformer.pt)

Verwendung:
    python chess_ai.py
"""

import os
import pickle
import sys
import time

import chess

try:
    import numpy as np
except ImportError:
    np = None

try:
    import torch  # noqa: F401
    from chess_model import load_model, predict_moves
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Matt-in-1-Teststellung: Dh5xf7# ist richtig
MINIMAX_DEPTH = 3   # per Befehl "depth 2" änderbar (kleiner = schneller)

TACTIC_FEN = "rnbq1knr/1ppp2pp/5p2/p1b1p2Q/2B1P3/7P/PPPP1PP1/RNB1K1NR w - - 0 1"


# ──────────────────────────────────────────────
# ALTES MODELL: 25 Features + Minimax
# (identisch zu train_model.py)
# ──────────────────────────────────────────────
PIECE_VALUES = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
                chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0}


def fen_to_features(fen: str) -> list:
    board = chess.Board(fen)
    features = []
    piece_counts = {}
    for color in [chess.WHITE, chess.BLACK]:
        for ptype in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]:
            count = len(board.pieces(ptype, color))
            piece_counts[(color, ptype)] = count
            features.append(count)
    white_material = sum(PIECE_VALUES[p] * piece_counts[(chess.WHITE, p)] for p in PIECE_VALUES)
    black_material = sum(PIECE_VALUES[p] * piece_counts[(chess.BLACK, p)] for p in PIECE_VALUES)
    features.append(white_material - black_material)
    features.append(len(list(board.legal_moves)))

    def count_passed_pawns(color):
        count = 0
        pawns = board.pieces(chess.PAWN, color)
        enemy_pawns = board.pieces(chess.PAWN, not color)
        for sq in pawns:
            file, rank = chess.square_file(sq), chess.square_rank(sq)
            passed = True
            for esq in enemy_pawns:
                ef, er = chess.square_file(esq), chess.square_rank(esq)
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
    features.append(1 if board.is_check() and board.turn == chess.BLACK else 0)
    features.append(1 if board.is_check() and board.turn == chess.WHITE else 0)
    features.append(sum(len(board.pieces(p, c)) for p in chess.PIECE_TYPES for c in [chess.WHITE, chess.BLACK]))
    features.append(1 if board.has_kingside_castling_rights(chess.WHITE) else 0)
    features.append(1 if board.has_queenside_castling_rights(chess.WHITE) else 0)
    features.append(1 if board.has_kingside_castling_rights(chess.BLACK) else 0)
    features.append(1 if board.has_queenside_castling_rights(chess.BLACK) else 0)
    features.append(1 if board.turn == chess.WHITE else -1)
    doubled = 0
    for file in range(8):
        n = sum(1 for sq in board.pieces(chess.PAWN, chess.WHITE) if chess.square_file(sq) == file)
        if n > 1:
            doubled += n - 1
    features.append(doubled)
    return features


def evaluate(board, gb_model, scaler) -> float:
    """Bewertung einer Stellung in Centipawns (Weiß-Sicht)."""
    if board.is_checkmate():
        return -10000 if board.turn == chess.WHITE else 10000
    if board.is_stalemate() or board.is_insufficient_material():
        return 0
    x = scaler.transform(np.array([fen_to_features(board.fen())]))
    return float(gb_model.predict(x)[0])


def minimax(board, depth, alpha, beta, gb_model, scaler) -> float:
    """Minimax mit Alpha-Beta-Pruning. Weiß maximiert, Schwarz minimiert."""
    if depth == 0 or board.is_game_over():
        return evaluate(board, gb_model, scaler)
    if board.turn == chess.WHITE:
        best = -float("inf")
        for move in board.legal_moves:
            board.push(move)
            best = max(best, minimax(board, depth - 1, alpha, beta, gb_model, scaler))
            board.pop()
            alpha = max(alpha, best)
            if beta <= alpha:
                break
        return best
    best = float("inf")
    for move in board.legal_moves:
        board.push(move)
        best = min(best, minimax(board, depth - 1, alpha, beta, gb_model, scaler))
        board.pop()
        beta = min(beta, best)
        if beta <= alpha:
            break
    return best


def best_move_minimax(board, gb_model, scaler, depth=3):
    best, best_score = None, -float("inf") if board.turn == chess.WHITE else float("inf")
    for move in board.legal_moves:
        board.push(move)
        score = minimax(board, depth - 1, -float("inf"), float("inf"), gb_model, scaler)
        board.pop()
        if (board.turn == chess.WHITE and score > best_score) or (board.turn == chess.BLACK and score < best_score):
            best_score, best = score, move
    return best, best_score


# ──────────────────────────────────────────────
# HYBRID: Transformer schlägt Kandidaten vor, Suche prüft sie nach
# ──────────────────────────────────────────────
MATE = 100000          # Matt-Wert; Matt in N Zügen = MATE - Halbzüge
HYBRID_DEPTH = 4       # Suchtiefe in Halbzügen (per Befehl "depth 3" änderbar)
HYBRID_K = 6           # so viele Kandidaten liefert der Transformer
PROB_BONUS = 50        # Centipawns Bonus für 100 % Transformer-Wahrscheinlichkeit (Tie-Break)


def material_stm(board) -> int:
    """Materialwert aus Sicht des Spielers am Zug (Centipawns)."""
    total = 0
    for ptype, val in PIECE_VALUES.items():
        total += val * (len(board.pieces(ptype, chess.WHITE)) - len(board.pieces(ptype, chess.BLACK)))
    return total if board.turn == chess.WHITE else -total


def _order(board, moves):
    """Schlagzüge zuerst (wertvolles Opfer, billiger Angreifer): macht Alpha-Beta schneller."""
    def key(m):
        if board.is_capture(m):
            victim = board.piece_type_at(m.to_square) or chess.PAWN   # en passant
            return 10 * PIECE_VALUES[victim] - PIECE_VALUES[board.piece_type_at(m.from_square)]
        return -1 + (1000 if m.promotion else 0)
    return sorted(moves, key=key, reverse=True)


def quiesce(board, alpha, beta, qdepth):
    """Nur noch Schlagzüge weiterrechnen, damit am Suchende keine Figur 'hängt'."""
    stand = material_stm(board)
    if qdepth == 0 or stand >= beta:
        return stand
    alpha = max(alpha, stand)
    for m in _order(board, list(board.generate_legal_captures())):
        board.push(m)
        score = -quiesce(board, -beta, -alpha, qdepth - 1)
        board.pop()
        if score >= beta:
            return score
        alpha = max(alpha, score)
    return alpha


def negamax(board, depth, alpha, beta, ply):
    """Alpha-Beta in Negamax-Form: Wert aus Sicht des Spielers am Zug."""
    if board.is_checkmate():
        return -(MATE - ply)
    if board.is_stalemate() or board.is_insufficient_material():
        return 0
    if depth <= 0:
        return quiesce(board, alpha, beta, 4)
    best = -float("inf")
    for m in _order(board, list(board.legal_moves)):
        board.push(m)
        score = -negamax(board, depth - 1, -beta, -alpha, ply + 1)
        board.pop()
        best = max(best, score)
        alpha = max(alpha, score)
        if alpha >= beta:
            break
    return best


def best_moves_hybrid(board, model, depth=HYBRID_DEPTH, k=HYBRID_K):
    """
    1. Transformer liefert die k wahrscheinlichsten Züge.
    2. Dazu kommen alle Schachgebote (taktisch wichtig, Matt wird so nie übersehen).
    3. Jeder Kandidat wird mit Alpha-Beta nachgerechnet.
    Rückgabe: Liste (Zug, Suchwert, Transformer-Wahrscheinlichkeit), beste zuerst.
    """
    cands = {m: p for m, p in predict_moves(model, board, k=k)}
    for m in board.legal_moves:
        if m not in cands and board.gives_check(m):
            cands[m] = 0.0
    results = []
    for m, p in cands.items():
        board.push(m)
        score = -negamax(board, depth - 1, -float("inf"), float("inf"), 1)
        board.pop()
        results.append((m, score, p))
    results.sort(key=lambda r: r[1] + PROB_BONUS * r[2], reverse=True)
    return results


def hybrid_score_text(score) -> str:
    if abs(score) > MATE - 100:
        n = (MATE - abs(score) + 1) // 2
        return f"Matt in {n}" if score > 0 else f"wird in {n} matt gesetzt"
    return f"{score / 100:+.2f} Bauern"


# ──────────────────────────────────────────────
# AUSGABE-HILFEN
# ──────────────────────────────────────────────
def move_readable(move: chess.Move) -> str:
    s = f"{chess.square_name(move.from_square)} -> {chess.square_name(move.to_square)}"
    return s + (f" ({chess.piece_name(move.promotion)})" if move.promotion else "")


def score_text(cp: float) -> str:
    if abs(cp) < 30:
        return "Ausgeglichene Stellung"
    side = "Weiß" if cp > 0 else "Schwarz"
    a = abs(cp)
    size = "kleiner" if a < 100 else "klarer" if a < 300 else "großer" if a < 600 else "entscheidender"
    return f"{side} hat {size} Vorteil"


def draw_board(board: chess.Board):
    print()
    print("    a  b  c  d  e  f  g  h")
    print("  ┌─────────────────────────┐")
    for rank in range(7, -1, -1):
        row = f"{rank + 1} │"
        for file in range(8):
            piece = board.piece_at(chess.square(file, rank))
            row += " · " if piece is None else f" {piece.unicode_symbol()} "
        print(row + f"│ {rank + 1}")
    print("  └─────────────────────────┘")
    print("    a  b  c  d  e  f  g  h")
    print()


# ──────────────────────────────────────────────
# ENGINES LADEN
# ──────────────────────────────────────────────
def load_engines() -> dict:
    engines = {}
    if HAS_TORCH:
        for name, path in [("transformer", "model_transformer.pt"), ("mlp", "model_mlp.pt")]:
            if os.path.exists(path):
                engines[name] = load_model(path)
        base = engines.get("transformer") or engines.get("mlp")
        if base is not None:
            engines["hybrid"] = base
    elif any(os.path.exists(p) for p in ("model_transformer.pt", "model_mlp.pt")):
        print("[HINWEIS] Modelle gefunden, aber PyTorch fehlt: python -m pip install torch")
    if os.path.exists("model.pkl") and os.path.exists("scaler.pkl") and np is not None:
        with open("model.pkl", "rb") as f:
            gb = pickle.load(f)
        with open("scaler.pkl", "rb") as f:
            sc = pickle.load(f)
        engines["minimax"] = (gb, sc)
    return engines


def analyse(name: str, engine, board: chess.Board, detail: bool = True):
    """Eine Engine auf die Stellung anwenden und das Ergebnis ausgeben."""
    t0 = time.time()
    if name == "minimax":
        gb, sc = engine
        move, score = best_move_minimax(board, gb, sc, depth=MINIMAX_DEPTH)
        print(f"  [{name}]  Empfehlung: {move_readable(move)} ({board.san(move)})")
        print(f"            Bewertung: {score / 100:+.2f} ({score_text(score)})   [{time.time() - t0:.1f}s]")
    elif name == "hybrid":
        res = best_moves_hybrid(board, engine, depth=HYBRID_DEPTH, k=HYBRID_K)
        move = res[0][0]
        if not detail:
            print(f"  ➜ Empfehlung: {board.san(move)}   ({move_readable(move)})")
            print(f"    Bewertung nach dem Zug: {hybrid_score_text(res[0][1])}   [{time.time() - t0:.1f}s]")
            return
        print(f"  [{name}]  Empfehlung: {move_readable(move)} ({board.san(move)})   [{time.time() - t0:.1f}s, Tiefe {HYBRID_DEPTH}]")
        for i, (m, sc, p) in enumerate(res[:3], 1):
            print(f"            {i}. {board.san(m):8s} {hybrid_score_text(sc):18s} (Netz: {p:5.1%})")
    else:
        preds = predict_moves(engine, board, k=3)
        print(f"  [{name}]  Empfehlung: {move_readable(preds[0][0])} ({board.san(preds[0][0])})   [{time.time() - t0:.2f}s]")
        for i, (m, p) in enumerate(preds, 1):
            print(f"            {i}. {board.san(m):8s} {p:6.1%}")


HELP = """
Befehle:
  <FEN>          Stellung analysieren (FEN von lichess.org kopieren)
  mode <name>    Engine wählen: hybrid | transformer | mlp | minimax
  all            Vergleich: alle Engines mit je 3 Vorschlägen zeigen (an/aus)
  depth <1-5>    Suchtiefe von hybrid (Standard 4) bzw. minimax (Standard 3); bei langsamem PC kleiner
  test           Matt-in-1-Teststellung analysieren
  help           diese Hilfe
  quit           beenden
"""


def main():
    global MINIMAX_DEPTH, HYBRID_DEPTH
    print("=" * 55)
    print("  ♟  ChessAI – Zugempfehlung  ♟")
    print("=" * 55)
    engines = load_engines()
    if not engines:
        print("[FEHLER] Keine Modelle gefunden.")
        print("  -> Neu: python data_generator_moves.py, dann python train_models.py")
        print("  -> Alt: python train_model.py")
        sys.exit(1)

    names = list(engines)
    mode = "hybrid" if "hybrid" in engines else names[0]
    show_all = False
    print("Verfügbare Engines:", ", ".join(names))
    print(HELP)

    while True:
        try:
            line = input(f"[{mode}] FEN oder Befehl> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAuf Wiedersehen!")
            break
        if not line:
            continue
        cmd = line.lower()

        if cmd in ("quit", "exit", "q"):
            print("Auf Wiedersehen!")
            break
        if cmd == "help":
            print(HELP)
            continue
        if cmd.startswith("mode"):
            parts = cmd.split()
            if len(parts) == 2 and parts[1] in engines:
                mode = parts[1]
                print(f"[OK] Engine: {mode}")
            else:
                print(f"[FEHLER] Verfügbar: {', '.join(names)}")
            continue
        if cmd.startswith("depth"):
            parts = cmd.split()
            if len(parts) == 2 and parts[1].isdigit() and 1 <= int(parts[1]) <= 5:
                if mode == "minimax":
                    MINIMAX_DEPTH = min(int(parts[1]), 4)
                    print(f"[OK] Minimax-Tiefe: {MINIMAX_DEPTH}")
                else:
                    HYBRID_DEPTH = int(parts[1])
                    print(f"[OK] Hybrid-Suchtiefe: {HYBRID_DEPTH}")
            else:
                print("[FEHLER] Beispiel: depth 2")
            continue
        if cmd == "all":
            show_all = not show_all
            print(f"[OK] Vergleich aller Engines: {'an' if show_all else 'aus'}")
            continue

        fen = TACTIC_FEN if cmd == "test" else line
        if cmd == "test":
            print("Teststellung: Weiß am Zug, richtig wäre Dh5xf7# (h5 -> f7, Matt in 1)")
        try:
            board = chess.Board(fen)
        except ValueError:
            print("[FEHLER] Ungültiger FEN-String. Tipp: FEN von lichess.org kopieren")
            continue

        draw_board(board)
        if board.is_checkmate():
            print("♟  SCHACHMATT! Das Spiel ist vorbei.")
            continue
        if board.is_stalemate() or board.is_insufficient_material():
            print("♟  Remis.")
            continue
        print(f"Am Zug: {'Weiß' if board.turn == chess.WHITE else 'Schwarz'}" + ("   ⚠ Im Schach!" if board.is_check() else ""))
        print("─" * 50)
        for name in (names if show_all else [mode]):
            analyse(name, engines[name], board, detail=show_all or name != "hybrid")
        print("─" * 50)
        print()


if __name__ == "__main__":
    main()
