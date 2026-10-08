"""Synthetic state replay, not an application or webcam recording."""
from copy import deepcopy
import contextlib
import io
from pathlib import Path
import math

from PIL import Image, ImageDraw, ImageFont
import cube_assistant as app
import gl_simulator as viewer
from cube_solver import solve_moves
from tools.cube_cases import impossible_cases, rotated_capture, scrambled


COLORS = {'W': '#ecf0f7', 'Y': '#ffdc48', 'O': '#ff983a',
          'R': '#f45b69', 'G': '#36ce9e', 'B': '#5897ff'}


def font(size):
    for name in ['DejaVuSans.ttf', 'Arial.ttf', 'arial.ttf']:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def project(p):
    x, y, z = p
    return (400 + 78 * (x-z) * math.sqrt(3)/2,
            310 + 78 * ((x+z)/2-y))


def frame(state, title, detail, status='STATE REPLAY'):
    im = Image.new('RGB', (800, 590), '#111827')
    draw = ImageDraw.Draw(im)
    draw.text((30, 24), '3D RUBIK\'S CUBE ASSISTANT', font=font(27), fill='#f3f4f6')
    draw.text((30, 66), 'SYNTHETIC DEMO  /  NO CAMERA RECORDING', font=font(17), fill='#fbbf24')
    draw.text((30, 108), title, font=font(22), fill='#93c5fd')
    for face in ('F','R','U'):
        n, u, v = viewer.FACE_FRAMES[face]
        for i, label in enumerate(state[face]):
            center = viewer._sticker_center(face, i, 1.012)
            points = [project(tuple(center[k] + a * u[k] + b * v[k] for k in range(3)))
                      for a,b in [(-.30,-.30),(.30,-.30),(.30,.30),(-.30,.30)]]
            draw.polygon(points, fill=COLORS[label], outline='#111827', width=3)
    draw.rounded_rectangle((25, 475, 775, 568), radius=12, fill='#1f2937')
    draw.text((42, 490), status, font=font(16), fill='#6ee7b7')
    draw.text((42, 522), detail, font=font(18), fill='#e5e7eb')
    return im


def main():
    state, _ = scrambled(20261008, length=8)
    capture = rotated_capture(state, 53)
    with contextlib.redirect_stdout(io.StringIO()):
        normalized, _ = app.try_all_orientations(capture)
    assert normalized is not None
    frames = [frame(state, '1 / A legal scrambled cube', 'Input is generated from a cubie model.')]
    durations = [1500]
    frames.append(frame(normalized, '2 / Capture orientation normalized',
                        'A valid candidate found among up to 4,096 rotations.'))
    durations.append(1700)
    moves = viewer._expand_moves_quarter_turns(solve_moves(normalized))
    current = deepcopy(normalized)
    for i, move in enumerate(moves):
        viewer.apply_move_inplace(current, move[0], not move.endswith("'"))
        frames.append(frame(current, f'3 / Solution step {i+1} of {len(moves)}',
                            f'Applied move: {move}   |   Solver: kociemba'))
        durations.append(400)
    frames.append(frame(current, '4 / Solved state', 'All solution steps replayed using the project move engine.'))
    durations.append(1700)
    bad = impossible_cases()['single_edge_flip']
    assert not app.is_valid_cube_silent(bad)
    frames.append(frame(bad, '5 / Invalid input rejected',
                        'One flipped edge: color counts alone are insufficient.',
                        'PHYSICAL VALIDATION: REJECTED'))
    durations.append(2400)
    Path('assets').mkdir(exist_ok=True)
    frames[0].save('assets/synthetic_demo.gif', save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, optimize=True)
    frames[0].save('assets/synthetic_preview.png')
    print(f'Created synthetic demo: {len(frames)} frames, {sum(durations)/1000:.1f}s')


if __name__ == '__main__':
    main()
