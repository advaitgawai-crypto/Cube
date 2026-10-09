"""
Ultimate test for the Rubik's cube in notations.py

Run:  python test_ultimate.py          (notations.py must be in the same folder)

Colors: upper=0 front=1 lower=2 right=3 left=4 back=5
Every face is read from OUTSIDE the cube (back = as seen from behind).
Sections:
  A  hand-derivable result of each move from a solved cube
  B  every move vs an independent 3D sticker simulator (on scrambled cubes)
  C  group identities (x4 = identity, M x4, X = R M' L' ...)
  D  famous algorithms with known orders
  E  random scramble + inverse, sticker counts, fixed centers/corners
"""
import random, sys
import numpy as np
try:
    from notations import Cube
except ImportError:
    from notation import Cube

FACES = ["upper", "front", "lower", "right", "left", "back"]
BASIC = "RLUDFB"
ALL   = "RLUDFBXYZMES"

passed = failed = 0
def check(name, ok, detail=""):
    global passed, failed
    if ok: passed += 1
    else:
        failed += 1; print(f"FAIL  {name}  {detail}")

def section(t): print(f"\n=== {t} ===")

# ---------- helpers ----------
def meth(m, prime=False):
    if not prime: return m
    return m + ("_Prime" if m in BASIC else "_prime")

def do(c, moves):
    """moves: 'R U R' U' X2 M' ...' (2 = twice, ' = prime)"""
    for t in moves.split():
        m, rest = t[0], t[1:]
        for _ in range({"": 1, "'": 1, "2": 2}[rest]):
            getattr(c, meth(m, rest == "'"))()
    return c

def run(moves, times=1):
    c = Cube()
    for _ in range(times): do(c, moves)
    return c

def st(c): return {f: getattr(c, f).copy() for f in FACES}
def eq(a, b): return all(np.array_equal(a[f], b[f]) for f in FACES)
SOLVED = st(Cube())
def is_solved(c): return eq(st(c), SOLVED)

def inverse(moves):
    out = []
    for t in reversed(moves.split()):
        if t.endswith("2"): out.append(t)
        elif t.endswith("'"): out.append(t[:-1])
        else: out.append(t + "'")
    return " ".join(out)

def order(moves, limit=2000):
    c = Cube()
    for n in range(1, limit + 1):
        do(c, moves)
        if is_solved(c): return n
    return None

def col(a, b, c_):  return np.array([[a, b, c_]] * 3)           # columns a|b|c
def row(a, b, c_):  return np.array([[a]*3, [b]*3, [c_]*3])      # rows a/b/c
def solid(i): return np.full((3, 3), i)

# ============================================================
section("A. one move from solved (hand-derivable)")
# only changed faces are listed; others stay solid
EXPECT = {
 # face turns
 "R": dict(upper=col(0,0,1), front=col(1,1,2), lower=col(2,2,5), back=col(0,5,5)),
 "L": dict(upper=col(5,0,0), front=col(0,1,1), lower=col(1,2,2), back=col(5,5,2)),
 "U": dict(front=row(3,1,1), right=row(5,3,3), back=row(4,5,5), left=row(1,4,4)),
 "D": dict(front=row(1,1,4), left=row(4,4,5), back=row(5,5,3), right=row(3,3,1)),
 "F": dict(upper=row(0,0,4), right=col(0,3,3), lower=row(3,2,2), left=col(4,4,2)),
 "B": dict(upper=row(3,0,0), right=col(3,3,2), lower=row(2,2,4), left=col(0,4,4)),
 # whole-cube rotations: X like R, Y like U, Z like F
 "X": dict(upper=solid(1), front=solid(2), lower=solid(5), back=solid(0)),
 "Y": dict(front=solid(3), right=solid(5), back=solid(4), left=solid(1)),
 "Z": dict(upper=solid(4), right=solid(0), lower=solid(3), left=solid(2)),
 # slices: M like L, E like D, S like F
 "M": dict(upper=col(0,5,0), front=col(1,0,1), lower=col(2,1,2), back=col(5,2,5)),
 "E": dict(front=row(1,4,1), left=row(4,5,4), back=row(5,3,5), right=row(3,1,3)),
 "S": dict(upper=row(0,4,0), right=col(3,0,3), lower=row(2,3,2), left=col(4,2,4)),
}
for m, changes in EXPECT.items():
    for prime in (False, True):
        c = Cube(); getattr(c, meth(m, prime))()
        for i, f in enumerate(FACES):
            if not prime:
                exp = changes.get(f, solid(i))
                check(f"{m}: {f}", np.array_equal(getattr(c, f), exp), f"\n{getattr(c, f)}\nexpected\n{exp}")
    # prime must undo the move
    c = Cube(); getattr(c, m)(); getattr(c, meth(m, True))()
    check(f"{m} then {m}' = solved", is_solved(c))

# a face turned clockwise (non-uniform face shows rotation direction)
pat = np.arange(9).reshape(3, 3)
cw = np.array([[6,3,0],[7,4,1],[8,5,2]])
for m, f in [("R","right"),("L","left"),("U","upper"),("D","lower"),("F","front"),("B","back")]:
    c = Cube(); setattr(c, f, pat.copy()); getattr(c, m)()
    check(f"{m}: {f} turns clockwise", np.array_equal(getattr(c, f), cw))

# ============================================================
section("B. every move vs independent 3D simulator (scrambled cubes)")
def build():
    S = []
    cols = {(0,1,0):0,(0,0,1):1,(0,-1,0):2,(1,0,0):3,(-1,0,0):4,(0,0,-1):5}
    for n, c in cols.items():
        idx = [i for i in range(3) if n[i] == 0]
        for a in (-1,0,1):
            for b in (-1,0,1):
                p = [0,0,0]; p[idx[0]] = a; p[idx[1]] = b
                for i in range(3):
                    if n[i]: p[i] = n[i]
                S.append((tuple(p), n, c))
    return S
def rot(v, axis, k):               # k quarter turns counterclockwise seen from +axis
    x, y, z = v
    for _ in range(k % 4):
        if axis == 0: y, z = -z, y
        elif axis == 1: x, z = z, -x
        else: x, y = -y, x
    return (x, y, z)
# move -> (axis, layers, k for the unprimed move)
SPEC = {"R":(0,(1,),-1),"L":(0,(-1,),1),"U":(1,(1,),-1),"D":(1,(-1,),1),
        "F":(2,(1,),-1),"B":(2,(-1,),1),
        "X":(0,(-1,0,1),-1),"Y":(1,(-1,0,1),-1),"Z":(2,(-1,0,1),-1),
        "M":(0,(0,),1),"E":(1,(0,),1),"S":(2,(0,),-1)}
def sim(S, m, prime=False, n=1):
    ax, layers, k = SPEC[m]; k = (-k if prime else k) * n
    return [(rot(p,ax,k), rot(nn,ax,k), c) if p[ax] in layers else (p,nn,c) for p,nn,c in S]
def read(S):
    d = {(p, n): c for p, n, c in S}
    def face(n, rd, cd):
        g = np.zeros((3,3), int)
        for r in range(3):
            for c in range(3):
                g[r,c] = d[(tuple(n[i] + rd[i]*(r-1) + cd[i]*(c-1) for i in range(3)), n)]
        return g
    return dict(upper=face((0,1,0),(0,0,1),(1,0,0)), front=face((0,0,1),(0,-1,0),(1,0,0)),
                lower=face((0,-1,0),(0,0,-1),(1,0,0)), right=face((1,0,0),(0,-1,0),(0,0,-1)),
                left=face((-1,0,0),(0,-1,0),(0,0,1)), back=face((0,0,-1),(0,-1,0),(-1,0,0)))

random.seed(12345)
# each move alone, after a scramble made of the six basic moves
for m in ALL:
    for prime in (False, True):
        bad = 0
        for _ in range(40):
            c = Cube(); S = build()
            for _ in range(15):
                b = random.choice(BASIC); p = random.random() < .5
                getattr(c, meth(b, p))(); S = sim(S, b, p)
            getattr(c, meth(m, prime))(); S = sim(S, m, prime)
            bad += not eq(st(c), read(S))
        check(f"{m}{chr(39) if prime else ''} matches 3D simulator", bad == 0, f"{bad}/40 scrambles wrong")
# long random sequences using all 12 moves
bad = 0
for _ in range(300):
    c = Cube(); S = build()
    for _ in range(30):
        m = random.choice(ALL); p = random.random() < .5
        getattr(c, meth(m, p))(); S = sim(S, m, p)
    bad += not eq(st(c), read(S))
check("300 random 30-move sequences (all 12 moves) match simulator", bad == 0, f"{bad}/300 wrong")

# ============================================================
section("C. identities")
for m in ALL:
    check(f"{m} x4 = solved", is_solved(run(m, 4)))
    check(f"{m}' x4 = solved", is_solved(run(m + "'", 4)))
    check(f"{m} x1 != solved", not is_solved(run(m)))
    check(f"{m}2 x2 = solved", is_solved(run(m + "2", 2)))

# whole-cube rotation = outer layers + slice  (standard definitions)
IDENT = {"X = R M' L'": ("X", "R M' L'"), "Y = U E' D'": ("Y", "U E' D'"),
         "Z = F S B'": ("Z", "F S B'")}
for name, (a, b) in IDENT.items():
    ok = True
    for _ in range(30):
        pre = " ".join(random.choice(BASIC) + random.choice(["", "'"]) for _ in range(12))
        ok &= eq(st(do(Cube(), pre + " " + a)), st(do(Cube(), pre + " " + b)))
    check(name + " (on scrambled cubes)", ok)

# opposite faces commute; adjacent faces do not
for a, b in [("R","L"),("U","D"),("F","B"),("M","R"),("E","U"),("S","F"),("X","R"),("Y","U"),("Z","F")]:
    check(f"{a},{b} commute", eq(st(run(f"{a} {b}")), st(run(f"{b} {a}"))))
for a, b in [("R","U"),("U","F"),("F","R"),("L","B"),("D","L"),("M","U"),("X","U"),("Y","R"),("Z","U")]:
    check(f"{a},{b} do not commute", not eq(st(run(f"{a} {b}")), st(run(f"{b} {a}"))))

# two quarter turns about different axes compose to a 120-degree rotation => order 3
for a, b in [("X","Y"),("Y","Z"),("Z","X"),("X","Z'"),("Y'","X")]:
    check(f"({a} {b})^3 = solved", is_solved(run(f"{a} {b}", 3)))
    check(f"({a} {b})^1 != solved", not is_solved(run(f"{a} {b}")))

# ============================================================
section("D. famous algorithms and their known orders")
KNOWN = {
 "R U R' U'": 6,                                  # sexy move
 "R U' R' U": 6,
 "R U R' U R U2 R'": 6,                           # Sune
 "R' U' R U' R' U2 R": 6,                         # anti-Sune
 "F R U R' U' F'": 6,                             # OLL cross-ish
 "R U": 105, "R U'": 63, "R2 U2": 6,
 "R2 L2 U2 D2 F2 B2": 2,                          # checkerboard-ish
 "M2 E2 S2": 2,
 "R U R' U' R' F R2 U' R' U' R U R' F'": 2,       # T-perm
 "F R U' R' U' R U R' F' R U R' U' R' F R F'": 2, # Y-perm
 "X": 4, "Y": 4, "Z": 4, "M": 4, "E": 4, "S": 4,
 "X2": 2, "Y2": 2, "Z2": 2, "M2": 2, "E2": 2, "S2": 2,
 "X Y": 3,
}
for alg, n in KNOWN.items():
    if n is None: continue
    got = order(alg)
    check(f"order of ({alg}) = {n}", got == n, f"got {got}")
    if n > 1:
        check(f"({alg}) x{n-1} != solved", not is_solved(run(alg, n - 1)))

# an algorithm followed by its inverse is always the identity
for alg in ["R U R' U R U2 R'", "F R U R' U' F'", "M' U M U2 M' U M", "X R U R' Y U' Z D E' S2"]:
    check(f"({alg}) then inverse = solved", is_solved(do(do(Cube(), alg), inverse(alg))))

# ============================================================
section("E. random scrambles: inverse, counts, fixed pieces")
for trial in range(200):
    scr = " ".join(random.choice(ALL) + random.choice(["", "'", "2"]) for _ in range(random.randint(1, 40)))
    c = do(Cube(), scr)
    vals = np.concatenate([getattr(c, f).ravel() for f in FACES])
    if not all((vals == i).sum() == 9 for i in range(6)):
        check(f"sticker counts ({scr})", False); break
    if not is_solved(do(c, inverse(scr))):
        check(f"scramble+inverse ({scr})", False); break
else:
    check("200 random scrambles: 9 stickers per color", True)
    check("200 random scrambles: scramble + inverse = solved", True)

# face turns never move centers
for m in BASIC:
    c = do(Cube(), " ".join(random.choice(BASIC) + random.choice(["","'"]) for _ in range(25)))
    check("centers fixed after random face turns", all(getattr(c, f)[1,1] == i for i, f in enumerate(FACES)))
    break
# slices and rotations never move corners relative to each other: corner stickers per face
def corners(c): return [getattr(c, f)[r, k] for f in FACES for r in (0,2) for k in (0,2)]
for m in "MES":
    c = do(Cube(), "R U F' L D B' R U'")
    before = corners(c); getattr(c, m)()
    check(f"{m} leaves all corner stickers in place", before == corners(c))
# X/Y/Z moves every center, but a solved cube stays 'solved' up to rotation
for m in "XYZ":
    c = run(m)
    ok = all(len(set(getattr(c, f).ravel())) == 1 for f in FACES)
    check(f"{m} keeps every face one solid color", ok)
# M E S on a solved cube then 4 turns of each
check("M E S M' E' S' returns a valid cube (9 per color)",
      all((np.concatenate([getattr(run("M E S M' E' S'"), f).ravel() for f in FACES]) == i).sum() == 9 for i in range(6)))

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
