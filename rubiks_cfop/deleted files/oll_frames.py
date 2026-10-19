from collections import Counter

from notations import Cube
from oll_algorithms import OLLAlgorithms
from oll_verify import FRAME_TOKENS, OLL_COUNT, verify

_oll = OLLAlgorithms()


def case_state(n):
    """Scrambled state for case n: the inverse of its algorithm applied to a solved cube."""
    cube = Cube()
    _oll.apply(cube, n, inverse=True)
    return cube


def main():
    frames_by_case = {}
    for n in range(1, OLL_COUNT + 1):
        state = case_state(n)
        frames = tuple(
            f for f in FRAME_TOKENS
            if verify(state, f, "none", n)[0]
        )
        frames_by_case[n] = frames

    kinds = Counter()
    for frames in frames_by_case.values():
        if len(frames) == len(FRAME_TOKENS):
            kinds["all four frames"] += 1
        elif len(frames) == 0:
            kinds["no frame"] += 1
        else:
            kinds["some frames"] += 1

    print("Cases by number of frames that solve them (AUF = none):")
    for label, count in kinds.most_common():
        print(f"    {count} cases: {label}")

    print("Frames per case:")
    for n, frames in frames_by_case.items():
        print(f"    OLL {n}: {list(frames) if frames else 'none'}")


if __name__ == "__main__":
    main()