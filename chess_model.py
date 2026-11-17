"""
chess_model.py
--------------
Gemeinsamer Code für Training (train_models.py) und Terminal (chess_ai.py):

  1. Tokenisierung:  Schachbrett -> 68 Zahlen (64 Felder + 4 Rochaderechte)
  2. Zug-Vokabular:  jeder Zug = (Startfeld, Zielfeld) = 64*64 = 4096 mögliche Züge
  3. Modelle:        MLP (Baseline) und Transformer
  4. predict_moves:  Top-k Züge mit Wahrscheinlichkeit für eine Stellung

Trick: Ist Schwarz am Zug, wird das Brett gespiegelt (Farben getauscht).
Das Modell sieht dadurch IMMER "Weiß am Zug" und muss nur eine Seite lernen.
"""

import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_MOVES = 64 * 64      # Zug-Vokabular: from*64 + to
SEQ_LEN = 68           # 64 Felder + 4 Rochade-Flags
VOCAB = 15             # 0 leer | 1-6 eigene Figuren | 7-12 gegnerische | 13/14 Flag nein/ja
MAX_LEGAL = 224        # mehr legale Züge gibt es in keiner Stellung (Maximum ist 218)


# ──────────────────────────────────────────────
# 1. TOKENISIERUNG
# ──────────────────────────────────────────────
def normalize(board: chess.Board, move: chess.Move = None):
    """Spiegelt Brett (und Zug), falls Schwarz am Zug ist. Gibt (board, move, flipped) zurück."""
    if board.turn == chess.WHITE:
        return board, move, False
    board = board.mirror()
    if move is not None:
        move = chess.Move(chess.square_mirror(move.from_square),
                          chess.square_mirror(move.to_square), move.promotion)
    return board, move, True


def encode(board: chess.Board) -> np.ndarray:
    """Brett (Weiß am Zug) -> 68 Tokens."""
    t = np.zeros(SEQ_LEN, dtype=np.int8)
    for sq, piece in board.piece_map().items():
        t[sq] = piece.piece_type + (0 if piece.color == chess.WHITE else 6)
    t[64] = 13 + int(board.has_kingside_castling_rights(chess.WHITE))
    t[65] = 13 + int(board.has_queenside_castling_rights(chess.WHITE))
    t[66] = 13 + int(board.has_kingside_castling_rights(chess.BLACK))
    t[67] = 13 + int(board.has_queenside_castling_rights(chess.BLACK))
    return t


def move_to_index(move: chess.Move) -> int:
    return move.from_square * 64 + move.to_square


def index_to_move(idx: int, board: chess.Board) -> chess.Move:
    """Umkehrung. Bauernumwandlung wird immer zur Dame (Unterverwandlungen sind extrem selten)."""
    from_sq, to_sq = divmod(int(idx), 64)
    promo = None
    if board.piece_type_at(from_sq) == chess.PAWN and chess.square_rank(to_sq) in (0, 7):
        promo = chess.QUEEN
    return chess.Move(from_sq, to_sq, promo)


def legal_indices(board: chess.Board) -> np.ndarray:
    """Alle legalen Zug-Indizes, aufgefüllt mit -1 auf feste Länge."""
    idx = sorted({move_to_index(m) for m in board.legal_moves})
    out = np.full(MAX_LEGAL, -1, dtype=np.int16)
    out[:len(idx)] = idx
    return out


def legal_mask(legal_idx: torch.Tensor) -> torch.Tensor:
    """(B, MAX_LEGAL) Indizes -> (B, N_MOVES) bool-Maske. Das ist das Legal-Move-Masking."""
    idx = legal_idx.long().clone()
    idx[idx < 0] = N_MOVES                       # Platzhalter-Spalte für die Auffüllung
    mask = torch.zeros(idx.shape[0], N_MOVES + 1, dtype=torch.bool, device=idx.device)
    mask.scatter_(1, idx, True)
    return mask[:, :N_MOVES]


# ──────────────────────────────────────────────
# 2. MODELLE
# ──────────────────────────────────────────────
class MLP(nn.Module):
    """Baseline: einfaches Netz ohne Attention. Sieht das Brett als langen Zahlenvektor."""

    def __init__(self, hidden: int = 512):
        super().__init__()
        self.config = {"type": "mlp", "hidden": hidden}
        self.net = nn.Sequential(
            nn.Linear(64 * 13 + 4, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, N_MOVES),
        )

    def forward(self, tok):
        board = F.one_hot(tok[:, :64].long(), 13).flatten(1).float()
        flags = (tok[:, 64:].long() - 13).float()
        return self.net(torch.cat([board, flags], dim=1))


class ChessTransformer(nn.Module):
    """
    Transformer-Encoder über 68 Tokens.
    head="pair":   Ausgabe = Wert für jedes (Startfeld, Zielfeld)-Paar. Passt zur Struktur von Zügen.
    head="linear": Mittelwert aller Tokens -> Linear(4096). So wie im ChatGPT-Chat beschrieben.
    """

    def __init__(self, d_model=128, layers=4, heads=4, ff=256, dropout=0.1, head="pair"):
        super().__init__()
        self.config = {"type": "transformer", "d_model": d_model, "layers": layers,
                       "heads": heads, "ff": ff, "dropout": dropout, "head": head}
        self.head_type = head
        self.d_model = d_model
        self.tok_emb = nn.Embedding(VOCAB, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, SEQ_LEN, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, heads, ff, dropout,
                                           batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        if head == "pair":
            self.from_proj = nn.Linear(d_model, d_model)
            self.to_proj = nn.Linear(d_model, d_model)
        else:
            self.out = nn.Linear(d_model, N_MOVES)

    def forward(self, tok):
        x = self.tok_emb(tok.long()) + self.pos_emb
        x = self.norm(self.encoder(x))
        if self.head_type == "pair":
            sq = x[:, :64]                                   # nur die 64 Felder
            f = self.from_proj(sq)
            t = self.to_proj(sq)
            logits = torch.einsum("bfd,btd->bft", f, t) / self.d_model ** 0.5
            return logits.flatten(1)                         # index = from*64 + to
        return self.out(x.mean(dim=1))


def build_model(config: dict) -> nn.Module:
    cfg = dict(config)
    kind = cfg.pop("type")
    return MLP(**cfg) if kind == "mlp" else ChessTransformer(**cfg)


def save_model(model: nn.Module, path: str):
    torch.save({"config": model.config, "state": model.state_dict()}, path)


def load_model(path: str) -> nn.Module:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = build_model(ckpt["config"])
    model.load_state_dict(ckpt["state"])
    model.eval()
    return model


# ──────────────────────────────────────────────
# 3. VORHERSAGE FÜR EINE STELLUNG
# ──────────────────────────────────────────────
@torch.no_grad()
def predict_moves(model: nn.Module, board: chess.Board, k: int = 3):
    """Gibt die k besten legalen Züge (Move, Wahrscheinlichkeit) für die echte Stellung zurück."""
    model.eval()
    nb, _, flipped = normalize(board)
    tok = torch.from_numpy(encode(nb)).unsqueeze(0)
    mask = legal_mask(torch.from_numpy(legal_indices(nb)).unsqueeze(0))
    logits = model(tok).masked_fill(~mask, float("-inf"))
    probs = torch.softmax(logits, dim=1)[0]
    top = torch.topk(probs, min(k, int(mask.sum())))
    result = []
    for p, idx in zip(top.values.tolist(), top.indices.tolist()):
        m = index_to_move(idx, nb)
        if flipped:
            m = chess.Move(chess.square_mirror(m.from_square),
                           chess.square_mirror(m.to_square), m.promotion)
        result.append((m, p))
    return result
