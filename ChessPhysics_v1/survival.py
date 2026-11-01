import glob, os, json
import numpy as np, chess, chess.pgn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import chess_physics as cp
W0 = 39.5
NAMES = {"1_morphy": "Morphy 1858", "2_byrne_fischer": "Byrne-Fischer 1956",
         "3_karpov_kasparov": "Karpov-Kasparov 1985", "4_kasparov_topalov": "Kasparov-Topalov 1999",
         "5_aronian_anand": "Aronian-Anand 2013"}
out = {}
for f in sorted(glob.glob("pgn_a/*.pgn")):
    key = os.path.basename(f)[:-4]
    g = chess.pgn.read_game(open(f)); b = g.board()
    ws, bs = [W0 / W0], [1.0]
    pt = {n: [] for n in ("P","N","B","R","Q")}
    first_loss = {}
    sym = {chess.PAWN:"P", chess.KNIGHT:"N", chess.BISHOP:"B", chess.ROOK:"R", chess.QUEEN:"Q"}
    def counts(bd):
        d = {n: 0 for n in pt}
        for p in bd.piece_map().values():
            if p.piece_type in sym: d[sym[p.piece_type]] += 1
        return d
    c0 = counts(b)
    for n in pt: pt[n].append(c0[n] / (16 if n=="P" else 4 if n!="Q" else 2))
    for m in g.mainline_moves():
        b.push(m)
        ws.append(cp.species_mass(b, chess.WHITE) / W0); bs.append(cp.species_mass(b, chess.BLACK) / W0)
        c = counts(b)
        for n in pt: pt[n].append(c[n] / (16 if n=="P" else 4 if n!="Q" else 2))
    # plies to reach thresholds: fraction of each species' weight remaining
    def half_life(s):
        a = np.array(s); i = np.where(a <= 0.5)[0]; return int(i[0]) if len(i) else None
    out[key] = dict(name=NAMES[key], plies=len(ws)-1,
                    W_final=ws[-1], B_final=bs[-1],
                    W_half=half_life(ws), B_half=half_life(bs),
                    pt_final={n: pt[n][-1] for n in pt},
                    ws=ws, bs=bs, pt=pt)
json.dump(out, open("survival.json", "w"))
fig, ax = plt.subplots(1, 5, figsize=(16, 3.2), sharey=True)
for a, (k, d) in zip(ax, out.items()):
    t = np.arange(len(d["ws"]))
    a.step(t, d["ws"], where="post", color="0.15", label="White")
    a.step(t, d["bs"], where="post", color="#1f77b4", label="Black")
    a.set_title(d["name"], fontsize=9); a.set_xlabel("ply"); a.set_ylim(0, 1.05)
ax[0].set_ylabel("surviving weight fraction"); ax[0].legend(fontsize=8, frameon=False)
fig.tight_layout(); fig.savefig("fig_survival.pdf"); fig.savefig("fig_survival.png", dpi=150)
for k, d in out.items():
    print(d["name"], d["plies"], "W_final=%.2f B_final=%.2f" % (d["W_final"], d["B_final"]), d["W_half"], d["B_half"], d["pt_final"])
