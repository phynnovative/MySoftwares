import io, math, random
import numpy as np
import chess, chess.pgn
import chess_physics as cp

def metrics(rows, info):
    c = cp.col
    T = len(rows); k = max(1, T // 4)
    H, S = c(rows, "H_dec"), c(rows, "surprisal")
    ham, hc, ch = c(rows, "hamming_state"), c(rows, "hamming_control"), c(rows, "channels")
    return dict(T=T,
        H_mean=H.mean(), H_first=H[:k].mean(), H_last=H[-k:].mean(),
        dH=H[:k].mean() - H[-k:].mean(),
        S_minus_H=S.mean() - H.mean(),
        phi=c(rows, "immune_fraction")[-1], n_annih=int(c(rows, "capture").sum()),
        density_final=c(rows, "density")[-1],
        ctrl_ratio=hc.mean() / max(ham.mean(), 1e-9),
        peak_channels=ch.max(), peak_pos=(int(np.argmax(ch)) + 1) / T,
        I_imm_share=info["I_imm"] / max(S.sum(), 1e-9),
        max_chain=info["max_chain"])

def game_from_moves(moves):
    g = chess.pgn.Game(); node = g
    for m in moves: node = node.add_variation(m)
    return g

def random_moves(n, rng, beta=None):
    """n plies of play from the start position; uniform random if beta is None,
    else Boltzmann-sampled with the model's heuristic energies."""
    for _ in range(200):
        b = chess.Board(); mv = []
        while len(mv) < n and not b.is_game_over(claim_draw=False):
            legal = list(b.legal_moves)
            if beta is None:
                m = rng.choice(legal)
            else:
                s = np.array([cp.move_score(b, x) for x in legal]) * beta
                s -= s.max(); p = np.exp(s); p /= p.sum()
                m = legal[rng.choices(range(len(legal)), weights=p)[0]]
            b.push(m); mv.append(m)
        if len(mv) == n: return mv
    raise RuntimeError("could not reach length")

def one_null(args):
    n, seed, beta = args
    rng = random.Random(seed)
    mv = random_moves(n, rng, beta)
    rows, info = cp.analyse_game(game_from_moves(mv), beta=1.0)
    return metrics(rows, info)
