import glob, os, json, csv
import numpy as np, chess.pgn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import chess_physics as cp, study_lib as sl
NAMES = {"1_morphy": "Morphy 1858", "2_byrne_fischer": "Byrne-Fischer 1956",
         "3_karpov_kasparov": "Karpov-Kasparov 1985", "4_kasparov_topalov": "Kasparov-Topalov 1999",
         "5_aronian_anand": "Aronian-Anand 2013"}
real, rowsd = {}, {}
for f in sorted(glob.glob("pgn_a/*.pgn")):
    k = os.path.basename(f)[:-4]
    g = chess.pgn.read_game(open(f))
    rows, info = cp.analyse_game(g, 1.0)
    real[k] = {a: (float(b) if not isinstance(b, int) else b) for a, b in sl.metrics(rows, info).items()}
    real[k]["ply_first_capture"] = int(np.argmax(cp.col(rows, "capture"))) + 1
    real[k]["surprisal_mean"] = float(cp.col(rows, "surprisal").mean())
    real[k]["I_imm"] = float(info["I_imm"]); real[k]["released"] = float(info["cond_mass"])
    rowsd[k] = rows
json.dump(real, open("real_metrics.json", "w"), indent=1)
fig, ax = plt.subplots(3, 5, figsize=(16, 7.2), sharex="col")
for j, (k, rows) in enumerate(rowsd.items()):
    t = cp.col(rows, "ply")
    ax[0, j].step(t, cp.col(rows, "immune_fraction"), where="post", color="0.15")
    ax[0, j].set_ylim(0, 0.8); ax[0, j].set_title(NAMES[k], fontsize=9)
    ax[1, j].plot(t, cp.col(rows, "H_max"), color="0.7", lw=1, label=r"$\log_2 N_t$")
    ax[1, j].plot(t, cp.col(rows, "H_dec"), color="#1f77b4", lw=1.4, label=r"$H_t$")
    ax[1, j].plot(t, cp.col(rows, "surprisal"), color="0.15", lw=0.7, ls=":", label=r"$I_t$")
    ax[1, j].set_ylim(0, 9)
    ax[2, j].plot(t, cp.col(rows, "hamming_control"), color="#1f77b4", lw=1.2, label="control field")
    ax[2, j].plot(t, cp.col(rows, "hamming_state"), color="0.15", lw=1.2, label="piece field")
    ax[2, j].set_xlabel("ply"); ax[2, j].set_ylim(0, 28)
ax[0, 0].set_ylabel(r"immune fraction $\varphi$"); ax[1, 0].set_ylabel("bits"); ax[2, 0].set_ylabel("Hamming flips / ply")
ax[1, 0].legend(fontsize=7, frameon=False, loc="lower left"); ax[2, 0].legend(fontsize=7, frameon=False)
fig.tight_layout(); fig.savefig("fig_games.pdf"); fig.savefig("fig_games.png", dpi=140)
for k, v in real.items(): print(k, {a: round(b, 3) if isinstance(b, float) else b for a, b in v.items()})
