"""Independent legal/illegal fixtures from kociemba's cubie representation.

This is a differential check against the external solver implementation,
not independent mathematical proof of either implementation.
"""
from copy import deepcopy
import random
from kociemba.pykociemba.cubiecube import CubieCube, moveCube

FACE_ORDER = "URFDLB"
FACE_COLORS = dict(zip(FACE_ORDER, "YGRWBO"))


def as_captured(cube):
    text = cube.toFaceCube().to_String()
    return {face: [FACE_COLORS[c] for c in text[i * 9:(i + 1) * 9]]
            for i, face in enumerate(FACE_ORDER)}


def scrambled(seed=20261008, length=20):
    rng = random.Random(seed)
    cube = CubieCube()
    moves = []
    previous = None
    for _ in range(length):
        face = rng.choice([f for f in FACE_ORDER if f != previous])
        turns = rng.randint(1, 3)
        for _ in range(turns):
            cube.multiply(moveCube[FACE_ORDER.index(face)])
        moves.append(face + {1: "", 2: "2", 3: "'"}[turns])
        previous = face
    return as_captured(cube), moves


def impossible_cases():
    cases = {}
    cube = CubieCube()
    cube.co[0] = 1
    cases["single_corner_twist"] = as_captured(cube)
    cube = CubieCube()
    cube.eo[0] = 1
    cases["single_edge_flip"] = as_captured(cube)
    cube = CubieCube()
    cube.ep[0], cube.ep[1] = cube.ep[1], cube.ep[0]
    cases["edge_permutation_parity"] = as_captured(cube)
    return cases


def rotated_capture(state, seed):
    from cube_assistant import rotate_face_n_times
    rng = random.Random(seed)
    return {face: rotate_face_n_times(labels, rng.randrange(4))
            for face, labels in deepcopy(state).items()}
