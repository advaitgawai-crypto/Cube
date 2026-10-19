import numpy as np

from notations import Cube
from oll_algorithms import OLLAlgorithms

# A real OLL algorithm acts on the yellow (last) layer and leaves the first layer
# (white) alone. So applied to a solved cube, the white face must stay uniform.
# In notations.py, white is on UPPER and yellow is on LOWER, but the OLL
# algorithms are written for yellow on top. This script checks which layer they
# actually disturb.

oll = OLLAlgorithms()

def white_intact(cube, white_face):
    return bool(np.all(getattr(cube, white_face) == 0))

# Case 1: current setup. Algorithms act on UPPER, which is white.
broken_now = []
for n in range(1, OLLAlgorithms.COUNT + 1):
    c = Cube()
    oll.apply(c, n)
    if not white_intact(c, "upper"):
        broken_now.append(n)

# Case 2: rotate the solved cube with z2 first, so white ends up on LOWER
# and yellow on UPPER. Then the algorithms act on yellow.
broken_flipped = []
for n in range(1, OLLAlgorithms.COUNT + 1):
    c = Cube()
    c.Z()
    c.Z()
    oll.apply(c, n)
    if not white_intact(c, "lower"):
        broken_flipped.append(n)

print(f"Current setup (algorithms act on UPPER): {len(broken_now)} of 57 break white.")
print(f"Flipped setup (algorithms act on yellow): {len(broken_flipped)} of 57 break white.")
print("Expected: many breaks in the current setup and 0 in the flipped setup.")
print("If the flipped setup breaks white, the Z-rotation convention is off. Paste the output.")
