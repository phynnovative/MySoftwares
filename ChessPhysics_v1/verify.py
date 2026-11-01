import io, glob, os, chess, chess.pgn
from source_b import B
for f in sorted(glob.glob('pgn_a/*.pgn')):
    key = os.path.basename(f)[:-4]
    ga = chess.pgn.read_game(open(f))
    gb = chess.pgn.read_game(io.StringIO(B[key]))
    # replay legality is enforced by the parser; compare move sequences
    ma = [m.uci() for m in ga.mainline_moves()]
    mb = [m.uci() for m in gb.mainline_moves()]
    bd = ga.end().board()
    errs = [str(e) for e in ga.errors] + [str(e) for e in gb.errors]
    print(f"{key:20s} plies A={len(ma):3d} B={len(mb):3d} identical={ma==mb} "
          f"errors={errs} mate={bd.is_checkmate()} result={ga.headers['Result']}")
