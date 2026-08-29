import math
from typing import Dict, List, Tuple, Optional
import glfw
from OpenGL.GL import *
from OpenGL.GLU import *
from cube_solver import solve_moves  # 해법 시퀀스 가져오기

# ===== 스케일 & 외형 =====
BOX_SIZE     = 2.0
BOX_HALF     = BOX_SIZE / 2.0

# 면 패널/스티커 오프셋 (본체 표면에서 얼마나 띄우는지)
PANEL_EPS    = 0.0025   # 검정 패널(면 전체를 덮는 통판)
STICKER_EPS  = 0.0120   # 스티커(패널보다 확실히 더 앞으로)

# 3x3 그리드 파라미터
CELL_GAP     = 0.66     # 스티커 중심 간 거리
MARGIN       = 0.08     # 스티커의 테두리 여백(검정 테두리 두께 느낌)
EDGE_WIDTH   = 6.0      # 큐브 외곽 12변 라인 두께

# ===== 카메라 =====
_yaw, _pitch, _zoom = 45.0, 30.0, 10.0

# ===== 상태: 캡처 색 / 애니메이션 컨트롤 =====
_state: Dict[str, List[str]] = {}  # 각 면 9칸 색 라벨(위→아래, 좌→우)
_anim = None                      # 현재 애니메이션 정보 dict 또는 None

# ▶▶ 해법 재생 상태
_moves_expanded: List[str] = []    # ¼턴 단위로 확장된 시퀀스(예: "U2" -> "U","U")
_step_idx: int = 0                 # 현재까지 적용한 ¼턴 수
_advance_on_finish: int = 0        # 애니 끝날 때 인덱스 증감(+1:앞으로, -1:뒤로)

# ===== 유효성 검사 =====
VALID_COLORS = {"W","Y","O","R","G","B"}
# 그리기 순서(임의). 내부 로직은 face 키로 직접 접근.
FACES = ("U","F","R","L","B","D")
# 네 GUI의 COLOR_TO_FACE(Y→U, R→F, G→R, B→L, O→B, W→D)와 일치
EXPECTED_CENTERS = {"U":"Y","F":"R","R":"G","L":"B","B":"O","D":"W"}

def validate_captured(captured: Dict[str, List[str]]) -> Tuple[bool, Optional[str]]:
    missing = [f for f in FACES if f not in captured]
    if missing:
        return False, f"faces missing: {missing}"
    for face in FACES:
        labs = captured.get(face)
        if not isinstance(labs, list) or len(labs) != 9:
            return False, f"face '{face}' must have 9 labels"
        if any(l not in VALID_COLORS for l in labs):
            return False, f"face '{face}' has invalid label(s)"
        if labs[4] != EXPECTED_CENTERS[face]:
            return False, f"center mismatch on '{face}' (got {labs[4]}, expect {EXPECTED_CENTERS[face]})"
    counts = {c:0 for c in VALID_COLORS}
    for face in FACES:
        for l in captured[face]:
            counts[l]+=1
    bad = {c:n for c,n in counts.items() if n!=9}
    if bad:
        return False, f"color counts must be 9 each: {bad}"
    return True, None

# ===== 고정 좌표 프레임 (법선 n, 오른쪽 u, 위쪽 v) =====
FACE_FRAMES: Dict[str, Tuple[Tuple[float,float,float],Tuple[float,float,float],Tuple[float,float,float]]] = {
    'F': ((0,0,+1), (+1,0,0), (0,+1,0)),  # 앞(+Z)
    'B': ((0,0,-1), (-1,0,0), (0,+1,0)),  # 뒤(-Z) ← u 반전(좌→우 정렬 유지)
    'U': ((0,+1,0), (+1,0,0), (0,0,-1)),  # 위(+Y)   v가 -Z
    'D': ((0,-1,0), (+1,0,0), (0,0,+1)),  # 아래(-Y) v가 +Z
    'R': ((+1,0,0), (0,0,-1), (0,+1,0)),  # 오른(+X) u가 -Z
    'L': ((-1,0,0), (0,0,+1), (0,+1,0)),  # 왼(-X)   u가 +Z
}

COLOR_RGB = {
    'W': (1.0, 1.0, 1.0),
    'Y': (1.0, 1.0, 0.0),
    'O': (1.0, 0.65, 0.0),
    'R': (1.0, 0.0, 0.0),
    'G': (0.0, 0.8, 0.0),
    'B': (0.0, 0.0, 1.0),
}

# ===== 행/열 인덱스 (편의) =====
TOP    = [0,1,2]
MID    = [3,4,5]
BOT    = [6,7,8]
LEFT   = [0,3,6]
CENTER = [1,4,7]
RIGHT  = [2,5,8]

# ===== 행렬/좌표 유틸 =====
def _rot_matrix(axis: Tuple[int,int,int], theta_deg: float):
    th = math.radians(theta_deg)
    c, s = math.cos(th), math.sin(th)
    x,y,z = axis
    if x:
        if x < 0: s = -s
        return ((1,0,0),(0,c,-s),(0,s,c))
    if y:
        if y < 0: s = -s
        return ((c,0,s),(0,1,0),(-s,0,c))
    if z < 0: s = -s
    return ((c,-s,0),(s,c,0),(0,0,1))

def _matmul3(M, v):
    return (
        M[0][0]*v[0] + M[0][1]*v[1] + M[0][2]*v[2],
        M[1][0]*v[0] + M[1][1]*v[1] + M[1][2]*v[2],
        M[2][0]*v[0] + M[2][1]*v[1] + M[2][2]*v[2],
    )

def _dot(a,b): return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]
def _sub(a,b): return (a[0]-b[0], a[1]-b[1], a[2]-b[2])
def _scale(a,s): return (a[0]*s, a[1]*s, a[2]*s)

def _sticker_center(face: str, idx: int, plane_stk: float) -> Tuple[float,float,float]:
    n,u,v = FACE_FRAMES[face]
    i, j = divmod(idx, 3)
    off_u = (j - 1) * CELL_GAP
    off_v = (1 - i) * CELL_GAP
    return (
        n[0]*plane_stk + u[0]*off_u + v[0]*off_v,
        n[1]*plane_stk + u[1]*off_u + v[1]*off_v,
        n[2]*plane_stk + u[2]*off_u + v[2]*off_v,
    )

def _face_index_from_pos(pos: Tuple[float,float,float], plane_stk: float) -> Tuple[str,int]:
    best_face = None
    best_abs = 1e9
    for f,(n,u,v) in FACE_FRAMES.items():
        d = abs(_dot(n,pos) - plane_stk)
        if d < best_abs:
            best_abs = d
            best_face = f
    f = best_face
    n,u,v = FACE_FRAMES[f]
    t = _dot(n,pos) - plane_stk
    pos_on = _sub(pos, _scale(n, t))
    plane_origin = (n[0]*plane_stk, n[1]*plane_stk, n[2]*plane_stk)
    du = _dot(u, _sub(pos_on, plane_origin)) / CELL_GAP
    dv = _dot(v, _sub(pos_on, plane_origin)) / CELL_GAP
    j = int(round(du + 1.0))
    i = int(round(1.0 - dv))
    j = max(0, min(2, j))
    i = max(0, min(2, i))
    idx = i*3 + j
    return f, idx

# ===== 회전 대상(stickers) 집합 =====
def _rotating_indices_for(move: str) -> List[Tuple[str,int]]:
    L: List[Tuple[str,int]] = []
    def add(face, idxs):
        for i in idxs: L.append((face, i))
    if move == 'U':
        add('U', range(9)); add('F', TOP); add('R', TOP); add('B', TOP); add('L', TOP)
    elif move == 'D':
        add('D', range(9)); add('F', BOT); add('R', BOT); add('B', BOT); add('L', BOT)
    elif move == 'R':
        add('R', range(9)); add('U', RIGHT); add('F', RIGHT); add('D', RIGHT); add('B', LEFT)
    elif move == 'L':
        add('L', range(9)); add('U', LEFT); add('F', LEFT); add('D', LEFT); add('B', RIGHT)
    elif move == 'F':
        add('F', range(9)); add('U', BOT); add('R', LEFT); add('D', TOP); add('L', RIGHT)
    elif move == 'B':
        add('B', range(9)); add('U', TOP); add('L', LEFT); add('D', BOT); add('R', RIGHT)
    return L

# ===== 퍼뮤테이션(색 재배치) — 기하 기반 =====
def apply_move_inplace(state: Dict[str, List[str]], move: str, cw: bool):
    plane_stk = BOX_HALF + STICKER_EPS
    axis = FACE_FRAMES[move][0]
    theta = -90.0 if cw else +90.0
    R = _rot_matrix(axis, theta)
    S = {f: state[f][:] for f in FACES}
    band = _rotating_indices_for(move)
    for (face, idx) in band:
        c0 = _sticker_center(face, idx, plane_stk)
        c1 = _matmul3(R, c0)
        f_dst, i_dst = _face_index_from_pos(c1, plane_stk)
        state[f_dst][i_dst] = S[face][idx]

# ====== 내부 유틸: 솔버 입력 키 재정렬 & 시퀀스 확장 ======
def _captured_for_solver(captured: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """solver.solve_moves가 기대하는 URFDLB 키 배열로 재정렬"""
    return {
        'U': captured['U'],
        'R': captured['R'],
        'F': captured['F'],
        'D': captured['D'],
        'L': captured['L'],
        'B': captured['B'],
    }

def _expand_moves_quarter_turns(moves: List[str]) -> List[str]:
    """
    ['U', "R'", 'U2', ...] -> ¼턴 단위로 확장
      - 'X'   -> ['X']
      - "X'"  -> ["X'"]
      - 'X2'  -> ['X','X']
    """
    out: List[str] = []
    for m in moves:
        if len(m) == 2 and m[1] == '2':
            out.extend([m[0], m[0]])
        else:
            out.append(m)  # 'X' 또는 "X'"
    return out

def _inverse_elementary(m: str) -> str:
    """¼턴 1개를 역회전으로 변환: 'U' <-> "U'" """
    if len(m) == 2 and m[1] == "'":
        return m[0]
    return m[0] + "'" if len(m) == 1 else m  # 'X' -> "X'"

# ===== 공개 API =====
def simulator(captured: Dict[str, List[str]]) -> None:
    ok, err = validate_captured(captured)
    if not ok:
        print(f"[Simulator] Invalid cube: {err}")
        return

    # 내부 렌더 상태
    global _state, _moves_expanded, _step_idx
    _state = {f: captured[f][:] for f in FACES}

    # 해법 시퀀스 받아 ¼턴 단위로 확장
    try:
        moves = solve_moves(_captured_for_solver(captured))  # ["R","U","R'",...,"F2",...]
    except Exception as e:
        print("[Simulator] Solver error:", e)
        moves = []  # 솔버 실패 시에도 뷰어는 띄워서 수동 확인 가능

    _moves_expanded = _expand_moves_quarter_turns(moves)  # ¼턴 리스트
    _step_idx = 0

    if moves:
        print("[Simulator] Solution length (tokens):", len(moves))
        print("[Simulator] Expanded (quarter turns):", len(_moves_expanded))
    else:
        print("[Simulator] No solution sequence (already solved or solver error).")

    run_gl_viewer()

# ====== GLFW 메인 ======
def run_gl_viewer() -> None:
    # GLFW 초기화 전 상태 저장
    print("[Simulator] Initializing GLFW...")
    
    if not glfw.init():
        print("[Simulator] glfw.init() failed")
        return

    # 윈도우 힌트 설정 (tkinter 간섭 최소화)
    glfw.window_hint(glfw.SAMPLES, 4)  # MSAA
    glfw.window_hint(glfw.FOCUSED, glfw.TRUE)
    glfw.window_hint(glfw.AUTO_ICONIFY, glfw.FALSE)  # 자동 최소화 방지
    glfw.window_hint(glfw.FOCUS_ON_SHOW, glfw.TRUE)
    glfw.window_hint(glfw.RESIZABLE, glfw.TRUE)
    glfw.window_hint(glfw.MAXIMIZED, glfw.TRUE)
    
    width, height = 1100, 780
    window = glfw.create_window(width, height, "Rubik's Cube Simulator", None, None)
    if not window:
        glfw.terminate()
        print("[Simulator] create_window failed")
        return

    glfw.make_context_current(window)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_MULTISAMPLE)
    glFrontFace(GL_CCW)
    glEnable(GL_CULL_FACE)
    glCullFace(GL_BACK)
    glClearColor(1.0, 1.0, 1.0, 1.0)
    glDisable(GL_LIGHTING)

    # 입력 콜백
    glfw.set_key_callback(window, _on_key)
    glfw.set_scroll_callback(window, _on_scroll)

    drag = {'on': False, 'x': 0.0, 'y': 0.0}
    def mouse_button_cb(win, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT:
            if action == glfw.PRESS:
                drag['on'] = True
                drag['x'], drag['y'] = glfw.get_cursor_pos(win)
            elif action == glfw.RELEASE:
                drag['on'] = False
    def cursor_pos_cb(win, x, y):
        if drag['on']:
            _on_drag(x - drag['x'], y - drag['y'])
            drag['x'], drag['y'] = x, y
    glfw.set_mouse_button_callback(window, mouse_button_cb)
    glfw.set_cursor_pos_callback(window, cursor_pos_cb)

    # 메인 루프
    prev_time = glfw.get_time()
    while not glfw.window_should_close(window):
        now = glfw.get_time()
        dt = now - prev_time
        prev_time = now

        # 애니메이션 진행
        if _anim is not None:
            _update_anim(now)

        fbw, fbh = glfw.get_framebuffer_size(window)
        glViewport(0, 0, fbw, fbh)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Projection
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        aspect = max(fbw / max(fbh, 1), 1e-6)
        gluPerspective(45.0, aspect, 1.0, 100.0)  # near=1.0

        # View
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        cx, cy, cz = _camera_pos(_yaw, _pitch, _zoom)
        gluLookAt(cx, cy, cz,  0, 0, 0,  0, 1, 0)

        # 본체 + 외곽
        _draw_cube_body()

        # 패널 & 스티커 (애니메이션 레이어는 transform을 따로 줌)
        _draw_layers_with_animation()

        # 진행상태를 윈도 타이틀에 표시
        if _moves_expanded:
            glfw.set_window_title(
                window,
                f"Rubik's Cube Simulator  |  step {_step_idx}/{len(_moves_expanded)}"
            )

        glfw.swap_buffers(window)
        glfw.poll_events()

    # 종료 처리
    print("[Simulator] Closing GLFW window...")
    glfw.destroy_window(window)
    glfw.terminate()
    print("[Simulator] GLFW terminated successfully")

# ===== 입력 =====
def _on_key(window, key, scancode, action, mods):
    global _yaw, _pitch
    if action not in (glfw.PRESS, glfw.REPEAT):
        return

    # 방향키: ¼턴 단위 재생/되감기
    if key in (glfw.KEY_RIGHT, glfw.KEY_UP):
        _step_forward()
        return
    if key in (glfw.KEY_LEFT, glfw.KEY_DOWN):
        _step_backward()
        return

    # ESC: 종료
    if key == glfw.KEY_ESCAPE:
        glfw.set_window_should_close(window, True)
        return

def _step_forward():
    """해법 ¼턴 한 수 재생"""
    global _anim, _advance_on_finish, _step_idx
    if _anim is not None:
        return
    if _step_idx >= len(_moves_expanded):
        return
    m = _moves_expanded[_step_idx]  # 'U' 또는 "U'"
    face = m[0]
    cw = (len(m) == 1)  # 'X'면 CW, "X'"면 CCW
    _advance_on_finish = +1
    _try_start_anim(face, cw)

def _step_backward():
    """직전 ¼턴 한 수 되감기"""
    global _anim, _advance_on_finish, _step_idx
    if _anim is not None:
        return
    if _step_idx <= 0:
        return
    m_prev = _moves_expanded[_step_idx - 1]
    inv = _inverse_elementary(m_prev)  # 역회전
    face = inv[0]
    cw = (len(inv) == 1)
    _advance_on_finish = -1
    _try_start_anim(face, cw)

def _on_scroll(window, xoff, yoff):
    global _zoom
    _zoom = max(1.8, min(10.0, _zoom - yoff * 0.2))

def _on_drag(dx, dy):
    global _yaw, _pitch
    _yaw   += dx * 0.25
    _pitch -= dy * 0.25
    _pitch = max(-89.0, min(89.0, _pitch))

def _camera_pos(yaw_deg: float, pitch_deg: float, radius: float) -> Tuple[float, float, float]:
    yr = math.radians(yaw_deg); pr = math.radians(pitch_deg)
    x = radius * math.cos(pr) * math.sin(yr)
    y = radius * math.sin(pr)
    z = radius * math.cos(pr) * math.cos(yr)
    return (x, y, z)

# ===== 애니메이션 =====
def _try_start_anim(move: str, cw: bool):
    global _anim
    if _anim is not None:
        return  # 재생 중엔 무시
    axis = FACE_FRAMES[move][0]         # 면 법선 축
    rotating_stickers = _rotating_indices_for(move)
    _anim = dict(
        move=move, cw=cw,
        t0=glfw.get_time(), dur=0.18,   # 180ms
        axis=axis,
        layer=rotating_stickers
    )

def _update_anim(now: float):
    global _anim, _state, _step_idx, _advance_on_finish
    t0 = _anim['t0']; dur = _anim['dur']
    alpha = (now - t0) / max(dur, 1e-6)
    if alpha >= 1.0:
        # 종료: 퍼뮤테이션을 실제 적용
        apply_move_inplace(_state, _anim['move'], _anim['cw'])
        _anim = None
        # 재생 인덱스 갱신 (앞으로/뒤로)
        if _advance_on_finish != 0:
            _step_idx = max(0, min(len(_moves_expanded), _step_idx + _advance_on_finish))
            _advance_on_finish = 0

def _current_angle_deg() -> float:
    """진행 중 각도(도): CW=-, CCW=+ (면 정면 기준)"""
    if _anim is None:
        return 0.0
    t0 = _anim['t0']; dur = _anim['dur']
    alpha = (glfw.get_time() - t0) / max(dur, 1e-6)
    alpha = max(0.0, min(1.0, alpha))
    s = 0.5 - 0.5 * math.cos(math.pi * alpha)  # sine in/out
    ang = 90.0 * s
    return -ang if _anim['cw'] else +ang

# ===== 렌더 =====
def _draw_cube_body():
    glColor3f(0.06, 0.06, 0.06)
    _draw_solid_box(BOX_SIZE)
    _draw_box_edges(BOX_SIZE, width=EDGE_WIDTH)

def _draw_layers_with_animation():
    plane_panel = BOX_HALF + PANEL_EPS
    plane_stk   = BOX_HALF + STICKER_EPS
    side = CELL_GAP * 3.0
    size = CELL_GAP - 2.0 * MARGIN

    rotating = set(_anim['layer']) if _anim else set()
    angle = _current_angle_deg() if _anim else 0.0
    axis = _anim['axis'] if _anim else (0,0,1)

    # 1) 패널
    for face in FACES:
        n,u,v = FACE_FRAMES[face]
        if _anim and face == _anim['move']:
            glPushMatrix()
            glRotatef(angle, *axis)
            _draw_quad_centered(n,u,v, plane_panel, side, color=(0.0,0.0,0.0))
            glPopMatrix()
        else:
            _draw_quad_centered(n,u,v, plane_panel, side, color=(0.0,0.0,0.0))

    # 2) 스티커
    for face in FACES:
        n,u,v = FACE_FRAMES[face]
        labs = _state[face]
        idx = 0
        for i in range(3):
            for j in range(3):
                off_u = (j - 1) * CELL_GAP
                off_v = (1 - i) * CELL_GAP
                center = (n[0]*plane_stk + u[0]*off_u + v[0]*off_v,
                          n[1]*plane_stk + u[1]*off_u + v[1]*off_v,
                          n[2]*plane_stk + u[2]*off_u + v[2]*off_v)
                col = COLOR_RGB.get(labs[idx], (0.6,0.6,0.6))
                if (face, idx) in rotating:
                    glPushMatrix()
                    glRotatef(angle, *axis)
                    _draw_square(center, n,u,v, size, col, outline=True)
                    glPopMatrix()
                else:
                    _draw_square(center, n,u,v, size, col, outline=True)
                idx += 1

def _draw_solid_box(size: float):
    s = size / 2.0
    glBegin(GL_QUADS)
    # +X
    glVertex3f(+s,-s,-s); glVertex3f(+s,+s,-s); glVertex3f(+s,+s,+s); glVertex3f(+s,-s,+s)
    # -X
    glVertex3f(-s,-s,+s); glVertex3f(-s,+s,+s); glVertex3f(-s,+s,-s); glVertex3f(-s,-s,-s)
    # +Y
    glVertex3f(-s,+s,-s); glVertex3f(+s,+s,-s); glVertex3f(+s,+s,+s); glVertex3f(-s,+s,+s)
    # -Y
    glVertex3f(-s,-s,+s); glVertex3f(+s,-s,+s); glVertex3f(+s,-s,-s); glVertex3f(-s,-s,-s)
    # +Z
    glVertex3f(-s,-s,+s); glVertex3f(-s,+s,+s); glVertex3f(+s,+s,+s); glVertex3f(+s,-s,+s)
    # -Z
    glVertex3f(+s,-s,-s); glVertex3f(+s,+s,-s); glVertex3f(-s,+s,-s); glVertex3f(-s,-s,-s)
    glEnd()

def _draw_box_edges(size: float, width: float = 5.0):
    s = size / 2.0
    glLineWidth(width)
    glColor3f(0.0, 0.0, 0.0)
    glBegin(GL_LINES)
    # 아래 사각
    glVertex3f(-s,-s,-s); glVertex3f(+s,-s,-s)
    glVertex3f(+s,-s,-s); glVertex3f(+s,-s,+s)
    glVertex3f(+s,-s,+s); glVertex3f(-s,-s,+s)
    glVertex3f(-s,-s,+s); glVertex3f(-s,-s,-s)
    # 위 사각
    glVertex3f(-s,+s,-s); glVertex3f(+s,+s,-s)
    glVertex3f(+s,+s,-s); glVertex3f(+s,+s,+s)
    glVertex3f(+s,+s,+s); glVertex3f(-s,+s,+s)
    glVertex3f(-s,+s,+s); glVertex3f(-s,+s,-s)
    # 수직 4개
    glVertex3f(-s,-s,-s); glVertex3f(-s,+s,-s)
    glVertex3f(+s,-s,-s); glVertex3f(+s,+s,-s)
    glVertex3f(+s,-s,+s); glVertex3f(+s,+s,+s)
    glVertex3f(-s,-s,+s); glVertex3f(-s,+s,+s)
    glEnd()
    glLineWidth(1.0)

def _draw_quad_centered(n, u, v, plane, side, color=(0,0,0)):
    glColor3f(*color)
    c = (n[0]*plane, n[1]*plane, n[2]*plane)
    s = side / 2.0
    p1 = (c[0] + u[0]*(-s) + v[0]*(-s), c[1] + u[1]*(-s) + v[1]*(-s), c[2] + u[2]*(-s) + v[2]*(-s))
    p2 = (c[0] + u[0]*(+s) + v[0]*(-s), c[1] + u[1]*(+s) + v[1]*(-s), c[2] + u[2]*(+s) + v[2]*(-s))
    p3 = (c[0] + u[0]*(+s) + v[0]*(+s), c[1] + u[1]*(+s) + v[1]*(+s), c[2] + u[2]*(+s) + v[2]*(+s))
    p4 = (c[0] + u[0]*(-s) + v[0]*(+s), c[1] + u[1]*(-s) + v[1]*(+s), c[2] + u[2]*(-s) + v[2]*(+s))
    glBegin(GL_QUADS)
    glNormal3f(*n)
    glVertex3f(*p1); glVertex3f(*p2); glVertex3f(*p3); glVertex3f(*p4)
    glEnd()

def _draw_square(center, n, u, v, side, color, outline=False):
    glColor3f(*color)
    s = side / 2.0
    p1 = (center[0]+u[0]*(-s)+v[0]*(-s), center[1]+u[1]*(-s)+v[1]*(-s), center[2]+u[2]*(-s)+v[2]*(-s))
    p2 = (center[0]+u[0]*(+s)+v[0]*(-s), center[1]+u[1]*(+s)+v[1]*(-s), center[2]+u[2]*(+s)+v[2]*(-s))
    p3 = (center[0]+u[0]*(+s)+v[0]*(+s), center[1]+u[1]*(+s)+v[1]*(+s), center[2]+u[2]*(+s)+v[2]*(+s))
    p4 = (center[0]+u[0]*(-s)+v[0]*(+s), center[1]+u[1]*(-s)+v[1]*(+s), center[2]+u[2]*(-s)+v[2]*(+s))
    glBegin(GL_QUADS)
    glNormal3f(*n)
    glVertex3f(*p1); glVertex3f(*p2); glVertex3f(*p3); glVertex3f(*p4)
    glEnd()
    if outline:
        glLineWidth(2.0)
        glColor3f(0.0, 0.0, 0.0)
        glBegin(GL_LINE_LOOP)
        glVertex3f(*p1); glVertex3f(*p2); glVertex3f(*p3); glVertex3f(*p4)
        glEnd()
        glLineWidth(1.0)