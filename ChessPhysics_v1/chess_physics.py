#!/usr/bin/env python3
"""
chess_physics.py  --  Chess as a hybrid physical system
=======================================================

Reads a PGN file and analyses every game as a *hard-core binary lattice gas*
with two mutually annihilating species (White, Black).

MODEL
-----
1. Lattice & exclusion   : 8x8 cells, at most ONE particle per cell (hard-core
                           exclusion; verified at every ply).
2. Two species           : White (+) and Black (-). Each particle type carries
                             * hierarchy rank  h  (P1 N2 B3 R4 Q5 K6)
                             * existence weight w (P1 N3 B3.25 R5 Q9, K = 0,
                               the king is not annihilable: it is the
                               absorbing boundary of the game).
3. Annihilation          : a capture is a W + B -> C reaction. Victim weight is
                           destroyed (released as 'reaction energy'); the pair
                           collapses to ONE particle on ONE cell, the cell is
                           stamped as a *decision-immune* site (absorbing,
                           Boolean flag that never returns to 0).
4. Boolean evolution     : state X_t in {0,1}^(12x64);  X_{t+1} = X_t XOR D(move).
                           Hamming weight |D| = Boolean activity of the ply
                           (quiet 2, capture 3, castling 4, ...). A second
                           Boolean field, the control field (attacked-by-W,
                           attacked-by-B), is evolved the same way.
5. Decision entropy      : at ply t the mover chooses among N_t legal moves.
                           Each move gets a heuristic 'energy' score s_m;
                           p_m = softmax(beta * s_m);
                           H_t = -sum p_m log2 p_m   (<= log2 N_t).
                           Surprisal of the played move  I_t = -log2 p_played.
                           Irreversible moves (captures, pawn pushes) are
                           ratchets: their surprisal is written permanently
                           into the state -> 'immune information' I_imm.
6. Extra observables     : density, interface length (unlike neighbours),
                           mixing index, reaction channels (legal captures),
                           contested cells (attacked by both), recurrences,
                           condensate fraction, occupation & scar heat-maps.

CONVENTIONS (bookkeeping, not literal chess physics)
---------------------------------------------------
* A capture is booked as a reaction that releases the victim's weight; the
  capturing piece survives on the board. "Immune" squares are a model-defined
  flag set at the capture square; they stay available to later play.
* "Immune information" is the cumulative surprisal of irreversible moves
  (captures and pawn pushes).
* Heuristic move energies are NOT engine-calibrated, and beta = 1 is not
  fitted. Quantities such as H and the surprisal depend on both.
* The control-field/piece-field flip ratio is descriptive only.

USAGE
-----
    pip install python-chess numpy matplotlib
    python chess_physics.py games.pgn                 # analyse all games
    python chess_physics.py games.pgn --game 2 --beta 1.5 --out results
    python chess_physics.py --demo                    # built-in Opera Game
Outputs per game: console report, <out>/game_N_plies.csv, <out>/game_N.png
"""

import argparse
import io
import math
import os
import csv
from collections import defaultdict

import numpy as np
import chess
import chess.pgn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------- constants
HIER = {chess.PAWN: 1, chess.KNIGHT: 2, chess.BISHOP: 3,
        chess.ROOK: 4, chess.QUEEN: 5, chess.KING: 6}
WEIGHT = {chess.PAWN: 1.0, chess.KNIGHT: 3.0, chess.BISHOP: 3.25,
          chess.ROOK: 5.0, chess.QUEEN: 9.0, chess.KING: 0.0}
INIT_MASS = sum(WEIGHT[p] * n for p, n in
                [(chess.PAWN, 8), (chess.KNIGHT, 2), (chess.BISHOP, 2),
                 (chess.ROOK, 2), (chess.QUEEN, 1)])      # per species = 39.5
DEMO_PGN = """[Event "Opera Game"]
[White "Morphy"]
[Black "Duke of Brunswick & Count Isouard"]
[Result "1-0"]

1. e4 e5 2. Nf3 d6 3. d4 Bg4 4. dxe5 Bxf3 5. Qxf3 dxe5 6. Bc4 Nf6 7. Qb3 Qe7
8. Nc3 c6 9. Bg5 b5 10. Nxb5 cxb5 11. Bxb5+ Nbd7 12. O-O-O Rd8 13. Rxd7 Rxd7
14. Rd1 Qe6 15. Bxd7+ Nxd7 16. Qb8+ Nxb8 17. Rd8# 1-0
"""


def popcount(x):
    return bin(x).count("1")


# ------------------------------------------------------------ Boolean fields
def state_bits(board):
    """12 x 64 Boolean occupation tensor (species x piece type, cell)."""
    X = np.zeros((12, 64), dtype=bool)
    for sq, pc in board.piece_map().items():
        X[(0 if pc.color else 6) + pc.piece_type - 1, sq] = True
    return X


def control_bits(board):
    """2 x 64 Boolean control field: cell attacked by White / by Black."""
    C = np.zeros((2, 64), dtype=bool)
    for sq in chess.SQUARES:
        C[0, sq] = board.is_attacked_by(chess.WHITE, sq)
        C[1, sq] = board.is_attacked_by(chess.BLACK, sq)
    return C


def centrality(sq):
    f, r = chess.square_file(sq), chess.square_rank(sq)
    return 3.5 - (abs(f - 3.5) + abs(r - 3.5)) / 2.0


def move_score(board, m):
    """Heuristic 'energy' of a move (higher = more favourable to the mover)."""
    mover = board.turn
    pc = board.piece_at(m.from_square)
    w = WEIGHT[pc.piece_type]
    s = 0.0
    if board.is_capture(m):
        if board.is_en_passant(m):
            s += WEIGHT[chess.PAWN]
        else:
            s += WEIGHT[board.piece_at(m.to_square).piece_type]
    if m.promotion:
        s += WEIGHT[m.promotion] - 1.0
    s += 0.15 * (centrality(m.to_square) - centrality(m.from_square))
    board.push(m)
    if board.is_check():
        s += 0.5
    if board.is_attacked_by(not mover, m.to_square):
        defended = board.is_attacked_by(mover, m.to_square)
        s -= (0.15 if defended else 0.5) * w
    board.pop()
    return s


def decision_entropy(board, played, beta):
    legal = list(board.legal_moves)
    n = len(legal)
    if n == 0:
        return 0, 0.0, 0.0, 0.0
    sc = np.array([move_score(board, m) for m in legal])
    z = beta * sc
    z -= z.max()
    p = np.exp(z)
    p /= p.sum()
    H = float(-(p * np.log2(np.clip(p, 1e-300, None))).sum())
    k = legal.index(played)
    return n, H, math.log2(n), float(-math.log2(p[k]))


# --------------------------------------------------------------- observables
def interface_stats(board):
    wocc = int(board.occupied_co[chess.WHITE])
    bocc = int(board.occupied_co[chess.BLACK])
    unlike = like = 0
    for sq in chess.scan_forward(wocc):
        unlike += popcount(chess.BB_KING_ATTACKS[sq] & bocc)
        like += popcount(chess.BB_KING_ATTACKS[sq] & wocc)
    for sq in chess.scan_forward(bocc):
        like += popcount(chess.BB_KING_ATTACKS[sq] & bocc)
    like //= 2
    mixing = unlike / (unlike + like) if (unlike + like) else 0.0
    return unlike, mixing


def species_mass(board, color):
    return sum(WEIGHT[p.piece_type] for p in board.piece_map().values()
               if p.color == color)


def mobility_other(board):
    b2 = board.copy(stack=False)
    b2.turn = not board.turn
    b2.ep_square = None
    return b2.pseudo_legal_moves.count()


# ------------------------------------------------------------------ analysis
def analyse_game(game, beta=1.0):
    board = game.board()
    X_prev = state_bits(board)
    C_prev = control_bits(board)
    heat_occ = np.zeros(64)
    scars = np.zeros(64)                 # capture counts per square
    immune_sites = np.zeros(64, dtype=bool)  # model flag: square has hosted a capture
    seen = defaultdict(int)
    seen[board.board_fen() + str(board.turn)] += 1

    cond_mass = 0.0
    n_annih = 0
    chain = 0
    max_chain = 0
    last_cap_sq = None
    I_imm = 0.0
    exclusion_violations = 0
    rows = []

    for ply, move in enumerate(game.mainline_moves(), start=1):
        mover = board.turn
        san = board.san(move)
        n, H, Hmax, surpr = decision_entropy(board, move, beta)
        channels = sum(1 for m in board.legal_moves if board.is_capture(m))
        pc = board.piece_at(move.from_square)
        is_cap = board.is_capture(move)
        irreversible = is_cap or pc.piece_type == chess.PAWN
        e_release = 0.0

        if is_cap:
            if board.is_en_passant(move):
                vic = chess.PAWN
                vsq = chess.square(chess.square_file(move.to_square),
                                   chess.square_rank(move.from_square))
            else:
                vic = board.piece_at(move.to_square).piece_type
                vsq = move.to_square
            e_release = WEIGHT[vic]
            cond_mass += e_release
            n_annih += 1
            scars[move.to_square] += 1
            immune_sites[move.to_square] = True
            chain = chain + 1 if last_cap_sq == move.to_square else 1
            max_chain = max(max_chain, chain)
            last_cap_sq = move.to_square
        else:
            chain = 0
            last_cap_sq = None
        if irreversible:
            I_imm += surpr

        board.push(move)

        X = state_bits(board)
        C = control_bits(board)
        if X.sum(axis=0).max() > 1:
            exclusion_violations += 1
        ham_state = int((X ^ X_prev).sum())
        ham_ctrl = int((C ^ C_prev).sum())
        X_prev, C_prev = X, C

        for sq in board.piece_map():
            heat_occ[sq] += 1
        key = board.board_fen() + str(board.turn)
        seen[key] += 1

        n_pieces = len(board.piece_map())
        interface, mixing = interface_stats(board)
        wm, bm = species_mass(board, chess.WHITE), species_mass(board, chess.BLACK)
        contested = int((C[0] & C[1]).sum())

        rows.append(dict(
            ply=ply, mover="W" if mover else "B", san=san,
            n_legal=n, H_dec=H, H_max=Hmax, surprisal=surpr,
            capture=int(is_cap), irreversible=int(irreversible),
            reaction_energy=e_release, condensate_mass=cond_mass,
            immune_fraction=cond_mass / (2 * INIT_MASS), I_immune=I_imm,
            hamming_state=ham_state, hamming_control=ham_ctrl,
            n_pieces=n_pieces, density=n_pieces / 64.0,
            w_mass=wm, b_mass=bm,
            order_param=(wm - bm) / (wm + bm) if (wm + bm) else 0.0,
            interface=interface, mixing=mixing,
            channels=channels, contested=contested,
            mobility_W=(board.legal_moves.count() if board.turn else mobility_other(board)),
            mobility_B=(board.legal_moves.count() if not board.turn else mobility_other(board)),
            check=int(board.is_check()),
        ))

    recurrences = sum(v - 1 for v in seen.values() if v > 1)
    info = dict(n_annih=n_annih, max_chain=max_chain, cond_mass=cond_mass,
                I_imm=I_imm, recurrences=recurrences,
                violations=exclusion_violations,
                result=game.headers.get("Result", "*"),
                final_board=board, heat_occ=heat_occ, scars=scars,
                immune_sites=immune_sites, n_sites=int(immune_sites.sum()))
    return rows, info


# ---------------------------------------------------------------- reporting
def col(rows, k):
    return np.array([r[k] for r in rows], dtype=float)


def interpret(rows, info):
    L = []
    T = len(rows)
    H, Hmax, S = col(rows, "H_dec"), col(rows, "H_max"), col(rows, "surprisal")
    cap = col(rows, "capture")
    dens = col(rows, "density")
    imm = col(rows, "immune_fraction")
    ham = col(rows, "hamming_state")
    hc = col(rows, "hamming_control")
    ch = col(rows, "channels")
    mix = col(rows, "mixing")

    L.append(f"Plies: {T}   Result: {info['result']}   "
             f"Exclusion violations: {info['violations']} (must be 0)")
    L.append("")
    L.append("1) HARD-CORE LATTICE GAS")
    L.append(f"   Density fell 0.500 -> {dens[-1]:.3f}. Because captures "
             "only remove pieces, the piece count is monotone "
             "non-increasing (irreversible in the sense that no piece is "
             "ever added, apart from promotions replacing pawns).")
    L.append("")
    L.append("2) CAPTURE REACTIONS (effective accounting, see docstring)")
    if info["n_annih"]:
        first = int(np.argmax(cap)) + 1
        L.append(f"   {info['n_annih']} capture events, first at ply "
                 f"{first}; reaction energy released = {info['cond_mass']:.2f} "
                 f"weight units (model-defined immune fraction "
                 f"phi = {imm[-1]:.3f}).")
        L.append(f"   Longest reaction cascade on one cell: "
                 f"{info['max_chain']} (exchange chains = chain reactions).")
        L.append(f"   {info['n_sites']} distinct squares are flagged as reaction sites "
                 "by model definition (a bookkeeping flag, not a claim that the "
                 "square is unavailable later).")
    else:
        L.append("   No captures occurred.")
    L.append("")
    L.append("3) DECISION ENTROPY")
    k = max(1, T // 4)
    slope = np.polyfit(np.arange(T), H, 1)[0] if T > 2 else 0.0
    L.append(f"   <H> = {H.mean():.2f} bits (max possible <log2 N> = "
             f"{Hmax.mean():.2f}); first quarter {H[:k].mean():.2f}, "
             f"last quarter {H[-k:].mean():.2f}; trend {slope:+.3f} bits/ply.")
    ratio = S.mean() - H.mean()
    L.append(f"   <surprisal> = {S.mean():.2f} bits; <surprisal> - <H> = "
             f"{ratio:+.2f} bits. The sign of this gap depends on beta and on "
             "the heuristic energy; it is not by itself a behavioural "
             "temperature.")
    L.append(f"   Cumulative surprisal of irreversible moves "
             f"(captures + pawn pushes): {info['I_imm']:.1f} bits "
             f"({100*info['I_imm']/max(S.sum(),1e-9):.0f}% of all "
             "information spent).")
    L.append("")
    L.append("4) BOOLEAN EVOLUTION")
    L.append(f"   Mean state Hamming flip/ply = {ham.mean():.2f} "
             f"(quiet=2, capture=3, castle=4); control-field flips/ply = "
             f"{hc.mean():.1f} (max {hc.max():.0f}).")
    gain = hc.mean() / max(ham.mean(), 1e-9)
    L.append(f"   Control-field / piece-field flip ratio = {gain:.2f} "
             "(descriptive; not a test of damage spreading).")
    if info["recurrences"]:
        L.append(f"   {info['recurrences']} repeated-position count.")
    L.append("")
    L.append("5) PHASES")
    third = max(1, T // 3)
    for name, sl in (("early", slice(0, third)),
                     ("middle", slice(third, 2 * third)),
                     ("late", slice(2 * third, T))):
        if len(dens[sl]) == 0:
            continue
        L.append(f"   {name:6s}: density {dens[sl].mean():.2f}, "
                 f"channels {ch[sl].mean():.1f}, H {H[sl].mean():.2f}, "
                 f"mixing {mix[sl].mean():.2f}, "
                 f"captures {int(cap[sl].sum())}")
    peak = int(np.argmax(ch)) + 1
    L.append(f"   Maximum legal-capture count {int(ch.max())} at ply {peak}.")
    return "\n".join(L)


def make_plots(rows, info, path, title):
    t = col(rows, "ply")
    fig, ax = plt.subplots(3, 2, figsize=(13, 11))
    fig.suptitle(title)
    a = ax[0, 0]
    a.plot(t, col(rows, "w_mass"), label="W mass")
    a.plot(t, col(rows, "b_mass"), label="B mass")
    a.plot(t, col(rows, "condensate_mass"), "k--", label="released weight (raw)")
    a.set_title("Existence weight and released weight")
    a.legend()
    a = ax[0, 1]
    a.plot(t, col(rows, "H_dec"), label="H_dec")
    a.plot(t, col(rows, "H_max"), ":", label="log2 N")
    a.plot(t, col(rows, "I_immune") / np.maximum(t, 1), "g--",
           label="cumulative irreversible-move surprisal / elapsed ply")
    a.set_title("Decision entropy (bits)")
    a.legend()
    a = ax[1, 0]
    a.plot(t, col(rows, "hamming_state"), label="state XOR")
    a.plot(t, col(rows, "hamming_control"), label="control XOR")
    a.set_title("Boolean activity (Hamming flips)")
    a.legend()
    a = ax[1, 1]
    a.plot(t, col(rows, "channels"), label="reaction channels")
    a.plot(t, col(rows, "contested"), label="contested cells")
    a.plot(t, col(rows, "interface"), label="interface length")
    a.set_title("Reactive interface")
    a.legend()
    for a, key, name in ((ax[2, 0], "heat_occ", "Occupation heat-map"),
                         (ax[2, 1], "scars", "Capture counts per square")):
        im = a.imshow(info[key].reshape(8, 8), origin="lower", cmap="magma")
        a.set_xticks(range(8)); a.set_xticklabels("abcdefgh")
        a.set_yticks(range(8)); a.set_yticklabels(range(1, 9))
        a.set_title(name)
        fig.colorbar(im, ax=a, fraction=0.046)
    for a in ax[:2].ravel():
        a.set_xlabel("ply")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pgn", nargs="?", help="PGN file (may contain many games)")
    ap.add_argument("--game", type=int, help="analyse only game N (1-based)")
    ap.add_argument("--beta", type=float, default=1.0,
                    help="inverse temperature of the move ensemble (default 1)")
    ap.add_argument("--out", default="chess_physics_out")
    ap.add_argument("--demo", action="store_true", help="use built-in game")
    args = ap.parse_args()
    if not args.pgn and not args.demo:
        ap.error("give a PGN file or use --demo")

    os.makedirs(args.out, exist_ok=True)
    fh = io.StringIO(DEMO_PGN) if args.demo else open(args.pgn, encoding="utf-8",
                                                      errors="replace")
    n = 0
    while True:
        game = chess.pgn.read_game(fh)
        if game is None:
            break
        n += 1
        if args.game and n != args.game:
            continue
        rows, info = analyse_game(game, args.beta)
        if not rows:
            continue
        h = game.headers
        title = f"Game {n}: {h.get('White','?')} - {h.get('Black','?')} ({info['result']})"
        print("=" * 72)
        print(title)
        print("=" * 72)
        print(interpret(rows, info))
        csv_path = os.path.join(args.out, f"game_{n}_plies.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        png = os.path.join(args.out, f"game_{n}.png")
        make_plots(rows, info, png, title)
        print(f"\n[saved] {csv_path}\n[saved] {png}\n")
    fh.close()


if __name__ == "__main__":
    main()
