from collections import Counter

from notations import Cube
from oll_algorithms import OLLAlgorithms
from oll_verify import (
    AUF_STEPS,
    AUF_TOKENS,
    FRAME_STEPS,
    FRAME_TOKENS,
    OLL_COUNT,
    valid_triples,
)

# Inverse of each token, used to build scrambles.
AUF_INVERSE = {"none": "none", "U": "U'", "U2": "U2", "U'": "U"}
FRAME_INVERSE = {"none": "none", "y": "y'", "y2": "y2", "y'": "y"}

_oll = OLLAlgorithms()


def build_states():
    """One scrambled state per (frame, auf, case): 4 * 4 * 57 = 912 states.

    Each state is built so that the triple it was generated from solves it:
        solved -> inverse(OLL case) -> inverse(AUF) -> inverse(frame)
    Solving applies frame, then AUF, then the case, which undoes that in reverse.
    """
    states = []
    for n in range(1, OLL_COUNT + 1):
        for frame in FRAME_TOKENS:
            for auf in AUF_TOKENS:
                cube = Cube()
                _oll.apply(cube, n, inverse=True)
                for name in AUF_STEPS[AUF_INVERSE[auf]]:
                    getattr(cube, name)()
                for name in FRAME_STEPS[FRAME_INVERSE[frame]]:
                    getattr(cube, name)()
                states.append((cube, (frame, auf, n)))
    return states


def main():
    states = build_states()
    print(f"Built {len(states)} states.")

    generator_mismatch = []   # generating triple does not solve its own state
    multi_label = 0           # states with more than one valid triple
    conflicts = Counter()     # (case, other case) pairs that both solve one state
    frame_none_aufs = {}      # case -> set of AUFs that solve it with frame "none"

    for cube, gen in states:
        frame, auf, n = gen
        triples = valid_triples(cube)

        if gen not in triples:
            generator_mismatch.append(gen)
            continue

        if len(triples) > 1:
            multi_label += 1

        other_cases = {m for (_, _, m) in triples if m != n}
        for m in other_cases:
            conflicts[(n, m)] += 1

        if frame == "none":
            frame_none_aufs.setdefault(n, set()).update(
                a for (f, a, m) in triples if f == "none" and m == n
            )

    print(f"Generator/verifier mismatches: {len(generator_mismatch)}")
    for gen in generator_mismatch[:10]:
        print("   ", gen)

    print(f"States with more than one valid triple: {multi_label} of {len(states)}")

    print("Cases that share a state with another case (possible wrong sequence):")
    if conflicts:
        for (n, m), count in sorted(conflicts.items()):
            print(f"    OLL {n} and OLL {m}: {count} shared states")
    else:
        print("    none")

    print("AUF values needed per case (frame = none):")
    histogram = Counter()
    for n in range(1, OLL_COUNT + 1):
        aufs = tuple(sorted(frame_none_aufs.get(n, set())))
        histogram[aufs] += 1
    for aufs, count in histogram.most_common():
        print(f"    {count} cases solve with AUF set {list(aufs) or 'none found'}")

    missing = [n for n in range(1, OLL_COUNT + 1) if not frame_none_aufs.get(n)]
    if missing:
        print(f"Cases with no solving AUF when frame = none: {missing}")


if __name__ == "__main__":
    main()