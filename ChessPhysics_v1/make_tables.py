import json, numpy as np, glob, os, chess.pgn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import chess_physics as cp
from scipy.stats import spearmanr
real = json.load(open("real_metrics.json")); nulls = json.load(open("nulls.json"))
surv = json.load(open("survival.json"))
KEYS = list(real)
SHORT = {"1_morphy": "Morphy--Isouard", "2_byrne_fischer": "Byrne--Fischer", "3_karpov_kasparov": "Karpov--Kasparov",
         "4_kasparov_topalov": "Kasparov--Topalov", "5_aronian_anand": "Aronian--Anand"}
YEAR = {"1_morphy": 1858, "2_byrne_fischer": 1956, "3_karpov_kasparov": 1985, "4_kasparov_topalov": 1999, "5_aronian_anand": 2013}
RES = {"1_morphy": "1--0", "2_byrne_fischer": "0--1", "3_karpov_kasparov": "0--1", "4_kasparov_topalov": "1--0", "5_aronian_anand": "0--1"}
# ---- Table 1: per-game observables
cols = [("T", "Plies", "%d"), ("n_annih", "Captures", "%d"), ("phi", r"$\varphi$", "%.3f"), ("density_final", r"$\rho_{\mathrm{end}}$", "%.3f"),
        ("H_mean", r"$\langle H\rangle$", "%.2f"), ("dH", r"$\Delta H$", "%.2f"), ("S_minus_H", r"$\langle I\rangle-\langle H\rangle$", "%.2f"),
        ("I_imm_share", r"$I_{\mathrm{imm}}/\sum I$", "%.2f"), ("ctrl_ratio", r"$R_{\mathrm{c/p}}$", "%.2f"), ("peak_channels", r"$C_{\max}$", "%d"),
        ("peak_pos", r"$t_{C}/T$", "%.2f")]
L = [r"\begin{tabular}{l" + "r" * len(cols) + "}", r"\toprule", "Game & " + " & ".join(c[1] for c in cols) + r" \\", r"\midrule"]
for k in KEYS:
    L.append(f"{SHORT[k]} ({YEAR[k]}) & " + " & ".join(c[2] % real[k][c[0]] for c in cols) + r" \\")
L += [r"\bottomrule", r"\end{tabular}"]
open("tab_games.tex", "w").write("\n".join(L) + "\n")
# ---- Table 2: z-scores vs nulls
M = [("H_mean", r"$\langle H\rangle$"), ("dH", r"$\Delta H$"), ("S_minus_H", r"$\langle I\rangle-\langle H\rangle$"),
     ("phi", r"$\varphi$"), ("ctrl_ratio", r"$R_{\mathrm{c/p}}$"), ("peak_channels", r"$C_{\max}$")]
Z = {"A": {}, "B": {}}; P = {"A": {}, "B": {}}; NM = {"A": {}, "B": {}}
for kind in "AB":
    for k in KEYS:
        for m, _ in M:
            x = np.array([r[m] for r in nulls[k][kind]], float); v = real[k][m]
            sd = x.std(ddof=1)
            Z[kind][(k, m)] = (v - x.mean()) / sd if sd > 0 else float("nan")
            nle, nge = (x <= v).sum(), (x >= v).sum(); n = len(x)
            P[kind][(k, m)] = min(1.0, 2 * (min(nle, nge) + 1) / (n + 1))
            NM[kind][(k, m)] = (x.mean(), sd)
def fmt(z, p):
    s = f"{z:+.1f}"
    return s + (r"$^{*}$" if abs(z) >= 3.3 else "")
L = [r"\begin{tabular}{l" + "r" * len(KEYS) + "}", r"\toprule", "Observable & " + " & ".join(SHORT[k] for k in KEYS) + r" \\", r"\midrule",
     r"\multicolumn{" + str(len(KEYS) + 1) + r"}{l}{\emph{Null A: uniformly random legal moves}} \\"]
for m, lab in M: L.append(lab + " & " + " & ".join(fmt(Z["A"][(k, m)], P["A"][(k, m)]) for k in KEYS) + r" \\")
L += [r"\midrule", r"\multicolumn{" + str(len(KEYS) + 1) + r"}{l}{\emph{Null B: moves sampled from the model ensemble ($\beta=1$)}} \\"]
for m, lab in M: L.append(lab + " & " + " & ".join(fmt(Z["B"][(k, m)], P["B"][(k, m)]) for k in KEYS) + r" \\")
L += [r"\bottomrule", r"\end{tabular}"]
open("tab_null.tex", "w").write("\n".join(L) + "\n")
# null means for text
nulltxt = {kind: {k: {m: NM[kind][(k, m)] for m, _ in M} for k in KEYS} for kind in "AB"}
json.dump({"Z": {kd: {f"{k}|{m}": v for (k, m), v in Z[kd].items()} for kd in "AB"},
           "P": {kd: {f"{k}|{m}": v for (k, m), v in P[kd].items()} for kd in "AB"},
           "nullmean": {kd: {k: {m: list(v) for m, v in d.items()} for k, d in nulltxt[kd].items()} for kd in "AB"}}, open("null_stats.json", "w"), indent=1)
# ---- Fig: z-score strips
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
cols_ = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
for a, kind, ttl in zip(ax, "AB", ["Null A: random legal moves", "Null B: model ensemble"]):
    for i, (m, lab) in enumerate(M):
        for j, k in enumerate(KEYS):
            a.plot(i + (j - 2) * 0.12, Z[kind][(k, m)], "o", color=cols_[j], ms=5, label=SHORT[k].replace("--", "-") if i == 0 else None)
    a.axhspan(-3.3, 3.3, color="0.9", zorder=0); a.axhline(0, color="0.5", lw=0.6)
    a.set_xticks(range(len(M))); a.set_xticklabels(['\u27e8H\u27e9', '\u0394H', '\u27e8I\u27e9\u2212\u27e8H\u27e9', '\u03c6', 'R c/p', 'C max'], fontsize=8)
    a.set_title(ttl, fontsize=9)
ax[0].set_ylabel("z-score of real game vs null"); ax[0].legend(fontsize=7, frameon=False, ncol=1, loc="upper left")
fig.tight_layout(); fig.savefig("fig_z.pdf"); fig.savefig("fig_z.png", dpi=140)
# ---- sacrifice plies: surprisal minus H, rank within game
SAC = {"1_morphy": (31, "16.Qb8+"), "2_byrne_fischer": (34, "17...Be6"), "4_kasparov_topalov": (47, "24.Rxd4"), "5_aronian_anand": (32, "16...Nde5")}
sac = {}
for k, (ply, lab) in SAC.items():
    g = chess.pgn.read_game(open(f"pgn_a/{k}.pgn")); rows, info = cp.analyse_game(g, 1.0)
    d = np.array([r["surprisal"] - r["H_dec"] for r in rows]); H = np.array([r["H_dec"] for r in rows]); S = np.array([r["surprisal"] for r in rows])
    i = ply - 1
    sac[k] = dict(ply=ply, move=lab, san=rows[i]["san"], H=float(H[i]), I=float(S[i]), diff=float(d[i]),
                  rank_pct=float((d <= d[i]).mean()), mover=rows[i]["mover"])
json.dump(sac, open("sacrifice.json", "w"), indent=1)
# ---- H5: survival rank order (final surviving fraction by type, mean over games)
types = ["P", "N", "B", "R", "Q"]
fin = {t: [surv[k]["pt_final"][t] for k in KEYS] for t in types}
mean_fin = {t: float(np.mean(fin[t])) for t in types}
rho, p = spearmanr([1, 2, 3, 4, 5], [mean_fin[t] for t in types])
json.dump({"mean_final": mean_fin, "per_game": fin, "spearman_rho": float(rho), "p": float(p)}, open("h5.json", "w"), indent=1)
print(json.dumps({"mean_final": mean_fin, "rho": rho}, indent=1))
for k, s in sac.items(): print(k, s)
for kind in "AB":
    print(kind)
    for m, _ in M: print(" ", m, [f"{Z[kind][(k,m)]:+.1f}" for k in KEYS], [f"{NM[kind][(KEYS[0],m)][0]:.2f}" for _ in [0]])
