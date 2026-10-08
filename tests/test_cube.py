from copy import deepcopy
import contextlib
import io

import numpy as np
import pytest
from kociemba.pykociemba.cubiecube import CubieCube, moveCube
from kociemba.pykociemba.facecube import FaceCube

import cube_assistant as app
import cube_solver as solver
import gl_simulator as viewer
from tools.cube_cases import (FACE_ORDER, as_captured, impossible_cases,
                              rotated_capture, scrambled)


def replay(state, moves):
    result = deepcopy(state)
    for move in viewer._expand_moves_quarter_turns(moves):
        viewer.apply_move_inplace(result, move[0], not move.endswith("'"))
    return result


def test_solved_state():
    state = as_captured(CubieCube())
    assert app._early_filter_impossible_combos(state)[0]
    assert app.is_valid_cube_silent(state)
    assert solver.solve_moves(state) == []


@pytest.mark.parametrize("seed", range(20))
def test_legal_scrambles_and_solution_replay(seed):
    state, moves = scrambled(seed)
    assert app._early_filter_impossible_combos(state)[0]
    assert app.is_valid_cube_silent(state)
    # Independent cubie generator and project's geometric move engine must agree.
    assert replay(as_captured(CubieCube()), moves) == state
    assert replay(state, solver.solve_moves(state)) == as_captured(CubieCube())


@pytest.mark.parametrize("name,state", list(impossible_cases().items()))
def test_reject_physically_impossible_cube(name, state):
    assert viewer.validate_captured(state)[0]  # All colors still occur nine times.
    assert app._early_filter_impossible_combos(state)[0]
    assert not app.is_valid_cube_silent(state)
    assert FaceCube(solver.captured_to_facelets_URFDLB(state)).toCubieCube().verify() != 0


@pytest.mark.parametrize("face", list(FACE_ORDER))
def test_turn_inverse_and_four_turns(face):
    state, _ = scrambled(42)
    assert replay(state, [face, face + "'"]) == state
    assert replay(state, [face] * 4) == state


@pytest.mark.parametrize("seed", range(6))
def test_capture_rotation_recovery(seed):
    original, _ = scrambled(100 + seed)
    capture = rotated_capture(original, 200 + seed)
    before = deepcopy(capture)
    with contextlib.redirect_stdout(io.StringIO()):
        normalized, rotations = app.try_all_orientations(capture)
    assert capture == before
    assert normalized is not None and rotations is not None
    assert FaceCube(solver.captured_to_facelets_URFDLB(normalized)).toCubieCube().verify() == 0
    assert replay(normalized, solver.solve_moves(normalized)) == as_captured(CubieCube())
    # A first valid candidate is not proof of uniquely recovering the original.


def test_counts_labels_and_duplicate_capture():
    state = as_captured(CubieCube())
    state['U'][0] = 'R'
    assert not viewer.validate_captured(state)[0]
    state['U'][0] = '?'
    assert not viewer.validate_captured(state)[0]
    app.CAPTURED.clear()
    app.CAPTURED_ORDER.clear()
    try:
        assert app.capture(['?'] * 9)[1] is not None
        assert app.capture(['Y'] * 9) == ('U', None)
        assert app.capture(['Y'] * 9)[1] is not None
        assert app.undo_capture() == 'U'
        assert not app.CAPTURED
    finally:
        app.CAPTURED.clear()
        app.CAPTURED_ORDER.clear()


@pytest.mark.parametrize('label,hsv', [
    ('W',(0,0,240)), ('Y',(28,200,220)), ('O',(12,200,220)),
    ('R',(0,200,220)), ('R',(175,200,220)),
    ('G',(60,200,220)), ('B',(110,200,220)), ('?',(0,0,0))])
def test_synthetic_color_patches(label, hsv):
    patch = np.full((16,16,3), hsv, dtype=np.uint8)
    assert app.label_by_mode(patch, app.COLOR_RULES, coverage_min=0.25) == label


def test_incomplete_solver_input():
    with pytest.raises(ValueError):
        solver.captured_to_facelets_URFDLB({'U': ['Y'] * 9})
