"""
data_generator_moves.py
-----------------------
Trainingsdaten für den Transformer: Stellung -> bester Zug laut Stockfish.

- Läuft parallel (mehrere Stockfish-Prozesse)
- Speichert laufend in die CSV (bei Abbruch geht nichts verloren)
- Kann fortgesetzt werden: einfach nochmal starten, es macht dort weiter

Verwendung:
    python data_generator_moves.py                  (20.000 Stellungen)
    python data_generator_moves.py --n 100000       (mehr Daten)
    python data_generator_moves.py --stockfish stockfish.exe

Ausgabe:
    moves_data.csv   (fen, best_move, score_cp)
"""

import argparse
import csv
import multiprocessing as mp
import os
import random
import sys
import time

import chess
import chess.engine

DEPTH = 8
TIME_LIMIT = 0.05

_engine = None


def init_worker(stockfish_path: str):
    """Jeder Worker startet seinen eigenen Stockfish (1 Thread)."""
    global _engine
    random.seed(time.time_ns() ^ os.getpid())
    _engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    _engine.configure({"Threads": 1, "Hash": 16})


def random_position() -> chess.Board:
    """Zufällige Stellung durch 6-60 zufällige Halbzüge."""
    while True:
        board = chess.Board()
        for _ in range(random.randint(6, 60)):
            if board.is_game_over():
                break
            board.push(random.choice(list(board.legal_moves)))
        if not board.is_game_over():
            return board


def work(_):
    """Eine Stellung erzeugen und von Stockfish bewerten lassen."""
    board = random_position()
    info = _engine.analyse(board, chess.engine.Limit(depth=DEPTH, time=TIME_LIMIT))
    pv = info.get("pv")
    if not pv:
        return None
    score = info["score"].white()
    if score.is_mate():
        cp = 10000 if score.mate() > 0 else -10000
    else:
        cp = score.score()
    return board.fen(), pv[0].uci(), cp


def load_existing(path: str) -> set:
    seen = set()
    if os.path.exists(path):
        with open(path, newline="") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row:
                    seen.add(row[0])
    return seen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20000, help="Ziel-Anzahl Stellungen")
    parser.add_argument("--out", default="moves_data.csv")
    parser.add_argument("--stockfish", default="stockfish.exe")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    args = parser.parse_args()

    seen = load_existing(args.out)
    start_count = len(seen)
    if start_count >= args.n:
        print(f"[OK] '{args.out}' hat schon {start_count} Stellungen.")
        return

    new_file = start_count == 0
    print("=" * 50)
    print("  ChessAI – Datengenerator (Stellung -> Zug)")
    print("=" * 50)
    print(f"Ziel: {args.n} | schon vorhanden: {start_count} | Worker: {args.workers}")

    try:
        test = chess.engine.SimpleEngine.popen_uci(args.stockfish)
        test.quit()
    except FileNotFoundError:
        print(f"[FEHLER] Stockfish nicht gefunden: '{args.stockfish}'")
        print("  -> Mit --stockfish den Pfad angeben, z.B. --stockfish stockfish.exe")
        sys.exit(1)

    t0 = time.time()
    written = 0
    with open(args.out, "a", newline="") as f, \
         mp.Pool(args.workers, initializer=init_worker, initargs=(args.stockfish,)) as pool:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["fen", "best_move", "score_cp"])

        try:
            for result in pool.imap_unordered(work, range(args.n * 3), chunksize=20):
                if result is None or result[0] in seen:
                    continue
                seen.add(result[0])
                writer.writerow(result)
                written += 1

                if written % 500 == 0:
                    f.flush()
                    rate = written / (time.time() - t0)
                    total = start_count + written
                    left = (args.n - total) / rate if rate > 0 else 0
                    print(f"  {total}/{args.n}  ({rate:.1f}/s, noch ca. {left/60:.0f} Min.)")

                if start_count + written >= args.n:
                    break
        except KeyboardInterrupt:
            print("\n[Abbruch] Bisherige Daten sind gespeichert.")

        pool.terminate()

    print(f"[OK] Fertig: {start_count + written} Stellungen in '{args.out}'")


if __name__ == "__main__":
    mp.freeze_support()
    main()
