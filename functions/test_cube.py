import numpy as np
from notations import Cube

# Colors: upper=0 front=1 lower=2 right=3 left=4 back=5
# Convention: every face is read from OUTSIDE the cube (back = viewed from behind).
FACES = ["upper", "front", "lower", "right", "left", "back"]

def state(c):
    return {f: getattr(c, f).copy() for f in FACES}

def same(a, b):
    return all(np.array_equal(a[f], b[f]) for f in FACES)

def rows(*r):
    return np.array([list(x) for x in r])

def col(a, b, c):
    """3x3 face whose columns are a|b|c (every row identical)"""
    return rows([a, b, c], [a, b, c], [a, b, c])

def row(a, b, c):
    """3x3 face whose rows are a/b/c (every column identical)"""
    return rows([a, a, a], [b, b, b], [c, c, c])

passed = failed = 0
def check(name, ok, detail=""):
    global passed, failed
    if ok:
        passed += 1; print(f"PASS  {name}")
    else:
        failed += 1; print(f"FAIL  {name}  {detail}")

# ---------------------------------------------------------------
# 1. Solved cube + sticker counts
# ---------------------------------------------------------------
c = Cube()
for i, f in enumerate(FACES):
    check(f"solved: {f} all {i}", (getattr(c, f) == i).all())

# ---------------------------------------------------------------
# 2. Hand-derived result of ONE move from solved.
#    (read these tables and verify them with a real cube in your hands)
# ---------------------------------------------------------------
# Each entry: move -> only the faces that change.  Unlisted faces stay solid.
EXPECT = {
    # R: F's right col goes up to U, U's goes to B (left col from behind), B's to D, D's to F
    "R": dict(upper=col(0,0,1), front=col(1,1,2), lower=col(2,2,5), back=col(0,5,5)),
    "L": dict(upper=col(5,0,0), front=col(0,1,1), lower=col(1,2,2), back=col(5,5,2)),
    # U: front row -> left, left -> back, back -> right, right -> front
    "U": dict(front=row(3,1,1), right=row(5,3,3), back=row(4,5,5), left=row(1,4,4)),
    "D": dict(front=row(1,1,4), left=row(4,4,5), back=row(5,5,3), right=row(3,3,1)),
    # F: U -> R -> D -> L -> U
    "F": dict(upper=row(0,0,4), right=col(0,3,3), lower=row(3,2,2), left=col(4,4,2)),
    # B: U -> L -> D -> R -> U
    "B": dict(upper=row(3,0,0), right=col(3,3,2), lower=row(2,2,4), left=col(0,4,4)),
}
for m, changes in EXPECT.items():
    c = Cube(); getattr(c, m)()
    for i, f in enumerate(FACES):
        exp = changes.get(f, np.full((3, 3), i))
        check(f"{m}: {f}", np.array_equal(getattr(c, f), exp),
              f"\n got\n{getattr(c, f)}\n expected\n{exp}")

# ---------------------------------------------------------------
# 3. Turning the moved face itself (use a non-uniform face so
#    rotation direction is visible).  Clockwise when facing it.
# ---------------------------------------------------------------
pat = np.arange(9).reshape(3, 3)       # [[0,1,2],[3,4,5],[6,7,8]]
cw  = rows([6,3,0],[7,4,1],[8,5,2])    # clockwise quarter turn
for m, f in [("R","right"),("L","left"),("U","upper"),("D","lower"),("F","front"),("B","back")]:
    c = Cube(); setattr(c, f, pat.copy()); getattr(c, m)()
    check(f"{m}: face {f} turns clockwise", np.array_equal(getattr(c, f), cw))

# ---------------------------------------------------------------
# 4. Algebraic identities (any wrong index breaks these)
# ---------------------------------------------------------------
solved = state(Cube())
def run(seq, n=1):
    c = Cube()
    for _ in range(n):
        for m in seq.split():
            getattr(c, m.replace("'", "_Prime"))()
    return c

for m in ["R","L","U","D","F","B"]:
    check(f"{m} x4 = identity", same(state(run(m, 4)), solved))
    check(f"{m} {m}' = identity", same(state(run(f"{m} {m}'")), solved))
    check(f"{m}' x4 = identity", same(state(run(f"{m}'", 4)), solved))

check("(R U R' U') x6 = identity", same(state(run("R U R' U'", 6)), solved))
check("(R U R' U') x3 != identity", not same(state(run("R U R' U'", 3)), solved))
check("(R U' R' U) x6 = identity", same(state(run("R U' R' U", 6)), solved))
check("(R U) x105 = identity", same(state(run("R U", 105)), solved))
check("(R U) x35 != identity", not same(state(run("R U", 35)), solved))
check("(F R U R' U' F') x6 = identity", same(state(run("F R U R' U' F'", 6)), solved))
check("(L' B L B') x6 = identity", same(state(run("L' B L B'", 6)), solved))
check("(D R' D' R) x6 = identity", same(state(run("D R' D' R", 6)), solved))
check("(R2 U2) x6 = identity",    same(state(run("R R U U", 6)), solved))

# Inverse of a long scramble undoes it
scr = "R U F' L D' B R' U' L F D B'"
inv = " ".join((t[:-1] if t.endswith("'") else t + "'") for t in reversed(scr.split()))
check("scramble + inverse = identity", same(state(run(scr + " " + inv)), solved))

# Opposite faces commute, adjacent faces do not
for a, b in [("R","L"),("U","D"),("F","B")]:
    check(f"{a},{b} commute", same(state(run(f"{a} {b}")), state(run(f"{b} {a}"))))
for a, b in [("R","U"),("U","F"),("F","R"),("L","B"),("D","L"),("B","D")]:
    check(f"{a},{b} do not commute", not same(state(run(f"{a} {b}")), state(run(f"{b} {a}"))))

# ---------------------------------------------------------------
# 5. Every color still appears exactly 9 times after a scramble
# ---------------------------------------------------------------
c = run(scr + " " + scr)
allv = np.concatenate([getattr(c, f).ravel() for f in FACES])
check("sticker counts = 9 each", all((allv == i).sum() == 9 for i in range(6)))

# ---------------------------------------------------------------
# 6. Your own run: R U' R' U  once
# ---------------------------------------------------------------
c = run("R U' R' U")
exp_back = rows([0,5,5],[5,5,5],[5,5,5])
check("R U' R' U -> back matches your output", np.array_equal(c.back, exp_back),
      f"\n{c.back}")

print(f"\n{passed} passed, {failed} failed")
