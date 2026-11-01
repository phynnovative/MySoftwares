import json, sys, time
from multiprocessing import Pool
import study_lib as sl
LENGTHS = {"1_morphy":33, "2_byrne_fischer":82, "3_karpov_kasparov":80,
           "4_kasparov_topalov":87, "5_aronian_anand":46}
NA, NB = 1000, 1000
if __name__ == "__main__":
    jobs = []
    for gi, (k, n) in enumerate(LENGTHS.items()):
        for i in range(NA): jobs.append((k, "A", (n, 1000*gi + i, None)))
        for i in range(NB): jobs.append((k, "B", (n, 500000 + 1000*gi + i, 1.0)))
    t0 = time.time()
    with Pool(2) as p:
        res = p.map(sl.one_null, [j[2] for j in jobs], chunksize=4)
    out = {}
    for (k, kind, _), r in zip(jobs, res):
        out.setdefault(k, {}).setdefault(kind, []).append(r)
    json.dump(out, open("nulls.json", "w"))
    print("done", time.time() - t0)
