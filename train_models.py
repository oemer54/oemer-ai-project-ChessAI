"""
train_models.py
---------------
Trainiert MLP (Baseline) und Transformer auf  Stellung -> Stockfish-Zug  und vergleicht sie.

Verwendung:
    python train_models.py                       (Standard: 10 Epochen, beide Modelle)
    python train_models.py --epochs 3 --limit 5000     (schneller Test)
    python train_models.py --models transformer --head linear   (nur ein Modell / Variante)

Voraussetzung:
    moves_data.csv  (erzeugt von data_generator_moves.py)

Ausgabe:
    model_mlp.pt, model_transformer.pt   (gespeicherte Modelle für chess_ai.py)
    results.json                         (alle Zahlen für Doku / Präsentation)
    results.png                          (Diagramm, falls matplotlib installiert ist)
"""

import argparse
import csv
import json
import math
import os
import sys
import time

import chess
import numpy as np
import torch
import torch.nn.functional as F

from chess_model import (MLP, ChessTransformer, N_MOVES, SEQ_LEN, MAX_LEGAL, encode,
                         legal_indices, legal_mask, move_to_index, normalize,
                         predict_moves, save_model)

# Teststellung: Weiß am Zug, Dh5xf7# ist Matt in 1
TACTIC_FEN = "rnbq1knr/1ppp2pp/5p2/p1b1p2Q/2B1P3/7P/PPPP1PP1/RNB1K1NR w - - 0 1"
TACTIC_BEST = "h5f7"


# ──────────────────────────────────────────────
# DATEN
# ──────────────────────────────────────────────
def load_data(csv_path: str, limit: int):
    """CSV (fen, best_move, score) -> Tokens, Label, legale Züge. Ergebnis wird zwischengespeichert."""
    cache = csv_path + ".cache.npz"
    key = np.array([os.path.getsize(csv_path), limit or 0])
    if os.path.exists(cache):
        z = np.load(cache)
        if np.array_equal(z["key"], key):
            print(f"[OK] Daten aus Zwischenspeicher geladen ({len(z['labels'])} Stellungen)")
            return z["tokens"], z["labels"], z["legal"]

    tokens, labels, legal = [], [], []
    skipped = 0
    t0 = time.time()
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for i, row in enumerate(reader):
            if limit and len(labels) >= limit:
                break
            try:
                board = chess.Board(row[0])
                move = chess.Move.from_uci(row[1])
                board, move, _ = normalize(board, move)
                if move not in board.legal_moves:
                    skipped += 1
                    continue
                tokens.append(encode(board))
                labels.append(move_to_index(move))
                legal.append(legal_indices(board))
            except (ValueError, IndexError):
                skipped += 1
            if (i + 1) % 20000 == 0:
                print(f"  {i + 1} Zeilen verarbeitet ({time.time() - t0:.0f}s)")

    tokens = np.stack(tokens)
    labels = np.array(labels, dtype=np.int16)
    legal = np.stack(legal)
    np.savez(cache, tokens=tokens, labels=labels, legal=legal, key=key)
    print(f"[OK] {len(labels)} Stellungen vorbereitet, {skipped} übersprungen")
    return tokens, labels, legal


# ──────────────────────────────────────────────
# AUSWERTUNG
# ──────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, data, idx, batch, device):
    """Loss, Top-1 und Top-5 Accuracy (nur legale Züge zählen)."""
    tokens, labels, legal = data
    model.eval()
    loss_sum, c1, c5 = 0.0, 0, 0
    for s in range(0, len(idx), batch):
        b = idx[s:s + batch]
        tok = torch.from_numpy(tokens[b]).to(device)
        y = torch.from_numpy(labels[b]).long().to(device)
        mask = legal_mask(torch.from_numpy(legal[b]).to(device))
        logits = model(tok).masked_fill(~mask, -1e9)
        loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
        top5 = logits.topk(5, dim=1).indices
        c1 += (top5[:, 0] == y).sum().item()
        c5 += (top5 == y.unsqueeze(1)).any(dim=1).sum().item()
    n = len(idx)
    return loss_sum / n, c1 / n, c5 / n


def train_model(name, model, data, train_idx, val_idx, args, device):
    lr = args.lr_mlp if name == "mlp" else args.lr_transformer
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n=== {name}  ({n_params:,} Parameter, lr={lr}) ===")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps_per_epoch = math.ceil(len(train_idx) / args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=args.epochs * steps_per_epoch, pct_start=0.1)

    tokens, labels, legal = data
    rng = np.random.RandomState(args.seed)
    history = []
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = rng.permutation(train_idx)
        run_loss, seen = 0.0, 0
        for s in range(0, len(perm), args.batch):
            b = perm[s:s + args.batch]
            tok = torch.from_numpy(tokens[b]).to(device)
            y = torch.from_numpy(labels[b]).long().to(device)
            mask = legal_mask(torch.from_numpy(legal[b]).to(device))
            logits = model(tok).masked_fill(~mask, -1e9)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run_loss += loss.item() * len(b)
            seen += len(b)

        val_loss, top1, top5 = evaluate(model, data, val_idx, args.batch, device)
        h = {"epoch": epoch, "train_loss": run_loss / seen, "val_loss": val_loss,
             "val_top1": top1, "val_top5": top5}
        history.append(h)
        print(f"  Epoche {epoch:2d}/{args.epochs} | train_loss {h['train_loss']:.3f} | "
              f"val_loss {val_loss:.3f} | Top-1 {top1:6.1%} | Top-5 {top5:6.1%} | "
              f"{time.time() - t_start:.0f}s")

    model.cpu()
    return {"params": n_params, "train_seconds": time.time() - t_start,
            "val_loss": history[-1]["val_loss"], "val_top1": history[-1]["val_top1"],
            "val_top5": history[-1]["val_top5"], "history": history}


def tactic_test(models: dict):
    """Die Matt-in-1-Stellung: was empfiehlt jedes Modell?"""
    board = chess.Board(TACTIC_FEN)
    assert chess.Move.from_uci(TACTIC_BEST) in board.legal_moves
    board.push(chess.Move.from_uci(TACTIC_BEST))
    assert board.is_checkmate(), "Teststellung ist kein Matt"
    board = chess.Board(TACTIC_FEN)

    print("\n=== Matt-in-1-Test (richtig wäre Dh5xf7#) ===")
    out = {}
    for name, model in models.items():
        preds = predict_moves(model, board, k=3)
        line = ", ".join(f"{board.san(m)} ({p:.0%})" for m, p in preds)
        found = preds[0][0].uci() == TACTIC_BEST
        print(f"  {name:12s}: {line}   -> {'GEFUNDEN' if found else 'nicht gefunden'}")
        out[name] = {"top3": [(board.san(m), p) for m, p in preds], "found": found}
    return out


def plot_results(results, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib nicht installiert, kein Diagramm. Optional: python -m pip install matplotlib)")
        return
    names = list(results["models"])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for n in names:
        h = results["models"][n]["history"]
        ax[0].plot([e["epoch"] for e in h], [e["val_top1"] * 100 for e in h], marker="o", label=n)
    ax[0].axhline(results["random_top1"] * 100, color="gray", linestyle="--", label="zufälliger legaler Zug")
    ax[0].set_xlabel("Epoche")
    ax[0].set_ylabel("Top-1 Accuracy (%)")
    ax[0].set_title("Lernkurve (Testdaten)")
    ax[0].legend()
    x = np.arange(len(names))
    ax[1].bar(x - 0.2, [results["models"][n]["val_top1"] * 100 for n in names], 0.4, label="Top-1")
    ax[1].bar(x + 0.2, [results["models"][n]["val_top5"] * 100 for n in names], 0.4, label="Top-5")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(names)
    ax[1].set_ylabel("Accuracy (%)")
    ax[1].set_title("Vergleich am Ende")
    ax[1].legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    print(f"[OK] Diagramm gespeichert: {path}")


# ──────────────────────────────────────────────
# HAUPTPROGRAMM
# ──────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="moves_data.csv")
    p.add_argument("--out-dir", default=".")
    p.add_argument("--models", default="mlp,transformer")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--limit", type=int, default=0, help="nur die ersten N Stellungen (Test)")
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--lr-mlp", type=float, default=1e-3)
    p.add_argument("--lr-transformer", type=float, default=5e-4)
    p.add_argument("--head", default="pair", choices=["pair", "linear"])
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if not os.path.exists(args.data):
        print(f"[FEHLER] '{args.data}' nicht gefunden.")
        print("  -> Zuerst ausführen: python data_generator_moves.py")
        sys.exit(1)

    torch.manual_seed(args.seed)
    torch.set_num_threads(os.cpu_count() or 1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 50)
    print("  ChessAI – Training MLP und Transformer")
    print("=" * 50)
    print(f"Gerät: {device}")

    data = load_data(args.data, args.limit)
    n = len(data[1])
    if n < 1000:
        print(f"[WARNUNG] Nur {n} Stellungen: zu wenig für sinnvolle Ergebnisse.")

    # Trainings- und Testdaten strikt trennen
    perm = np.random.RandomState(args.seed).permutation(n)
    n_val = max(1, int(n * args.val_fraction))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    n_legal = (data[2] >= 0).sum(axis=1)[val_idx]
    random_top1 = float(np.mean(1.0 / n_legal))
    random_top5 = float(np.mean(np.minimum(5, n_legal) / n_legal))
    print(f"Training: {len(train_idx)} | Test: {len(val_idx)} Stellungen")
    print(f"Zum Vergleich, ein zufälliger legaler Zug trifft: Top-1 {random_top1:.1%}, Top-5 {random_top5:.1%}")

    wanted = [m.strip() for m in args.models.split(",")]
    results = {"n_train": int(len(train_idx)), "n_val": int(len(val_idx)),
               "random_top1": random_top1, "random_top5": random_top5,
               "epochs": args.epochs, "models": {}}
    trained = {}

    if "mlp" in wanted:
        model = MLP()
        results["models"]["mlp"] = train_model("mlp", model, data, train_idx, val_idx, args, device)
        save_model(model, os.path.join(args.out_dir, "model_mlp.pt"))
        trained["mlp"] = model

    if "transformer" in wanted:
        model = ChessTransformer(d_model=args.d_model, layers=args.layers, head=args.head)
        results["models"]["transformer"] = train_model("transformer", model, data, train_idx, val_idx, args, device)
        results["models"]["transformer"]["config"] = model.config
        save_model(model, os.path.join(args.out_dir, "model_transformer.pt"))
        trained["transformer"] = model

    results["tactic_test"] = tactic_test(trained)

    print("\n=== Ergebnis (Testdaten) ===")
    print(f"  {'Modell':12s} {'Parameter':>10s} {'Top-1':>8s} {'Top-5':>8s}")
    print(f"  {'zufällig':12s} {'-':>10s} {random_top1:8.1%} {random_top5:8.1%}")
    for name, r in results["models"].items():
        print(f"  {name:12s} {r['params']:10,d} {r['val_top1']:8.1%} {r['val_top5']:8.1%}")

    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\n[OK] Gespeichert: results.json, model_*.pt")
    plot_results(results, os.path.join(args.out_dir, "results.png"))


if __name__ == "__main__":
    main()
