import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import sys
from gl_simulator import simulator
from cube_solver import solve_moves
from itertools import product
import time

MIRROR_PREVIEW = True  # 화면만 거울처럼 보이게 할지 여부
DEBUG_HSV = False

# 센터 색-면 고정 매핑
COLOR_TO_FACE = {
    "Y": "U",
    "R": "F",
    "G": "R",
    "B": "L",
    "O": "B",
    "W": "D"
}

# 색 규칙
COLOR_RULES = {
    "W": dict(s_max=40,  v_min=180),                        # White
    "Y": dict(h_min=22,  h_max=34,  s_min=70,  v_min=150),  # Yellow
    "O": dict(h_min=5,   h_max=22,  s_min=90, v_min=100),  # Orange
    "R": dict(h_wrap=[(0, 6), (170, 179)], s_min=80, v_min=80),  # Red
    "G": dict(h_min=40,  h_max=85,  s_min=60,  v_min=80),   # Green
    "B": dict(h_min=90,  h_max=130, s_min=60,  v_min=80)    # Blue
}

LETTER_BGR = {
    "W": (255, 255, 255),
    "Y": (0, 255, 255),
    "O": (0, 165, 255),
    "R": (0, 0, 255),
    "G": (0, 200, 0),
    "B": (255, 0, 0)
}

# 디버그 스위치
BLUR_KERNEL = 3         # 0이면 블러 끔, 3/5 추천

# 캡처 상태
CAPTURED = {}           # {'U':[9 labels], ...}
CAPTURED_ORDER = []     # ['U','F',...]  오른쪽 미리보기 표시/되돌리기용

# ROI
def calc_grid(frame_shape, box_ratio=0.6):
    """
    중앙 정사각형 의도 크기 S에서 3의 배수 S3로 줄여 모든 셀 동일 크기.
    반환: [(x,y,w,h)*9] (정수)
    """
    H, W = frame_shape[:2]
    S  = int(min(W, H) * box_ratio)  # 중앙 정사각형 한 변
    x0 = (W - S) // 2                # 좌상단 x
    y0 = (H - S) // 2                # 좌상단 y

    cs = S // 3                      # 기본 셀 한 변(정수)
    rects = []
    for r in range(3):
        for c in range(3):
            w = cs; h = cs
            x = x0 + c*cs
            y = y0 + r*cs
            rects.append((int(x), int(y), int(w), int(h)))
    return rects

def center_rect(rect, frame_shape, center_ratio=0.6):
    """
    (x,y,w,h)에서 중앙 부분만 잘라 (cx,cy,cw,ch) 반환.
    """
    x, y, w, h = rect
    cx = x + (1 - center_ratio) * 0.5 * w
    cy = y + (1 - center_ratio) * 0.5 * h
    cw = center_ratio * w
    ch = center_ratio * h
    return int(cx), int(cy), int(cw), int(ch)

# BGR2HSV
def to_hsv(bgr_img):
    """
    BGR→HSV 변환. OpenCV HSV: H[0,179], S/V[0,255]
    """
    return cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)

# Gaussian Blur
def apply_blur(img, k=BLUR_KERNEL):
    """
    가우시안 블러. k=0이면 원본 반환.
    """
    if k is None or k <= 1:
        return img
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(img, (k, k), 0)

# 마스크 & 모드 라벨
def build_mask(hsv_roi, rule):
    """
    규칙에 맞는 0/255 마스크 반환.
    rule 스키마:
      - h_min/h_max (없으면 0~180), 또는 h_wrap=[(l1,u1),(l2,u2)]
      - s_min/s_max (없으면 0~255)
      - v_min/v_max (없으면 0~255)
    """
    H = hsv_roi[:, :, 0]
    S = hsv_roi[:, :, 1]
    V = hsv_roi[:, :, 2]

    h_min = int(rule.get("h_min", 0))
    h_max = int(rule.get("h_max", 179))
    s_min = int(rule.get("s_min", 0))
    s_max = int(rule.get("s_max", 255))
    v_min = int(rule.get("v_min", 0))
    v_max = int(rule.get("v_max", 255))

    if "h_wrap" in rule:
        red = rule.get("h_wrap")
        r1 = cv2.inRange(H, red[0][0], red[0][1])
        r2 = cv2.inRange(H, red[1][0], red[1][1])
        mask_h = cv2.bitwise_or(r1, r2)
    else:
        mask_h = cv2.inRange(H, h_min, h_max)

    mask_s = cv2.inRange(S, s_min, s_max)
    mask_v = cv2.inRange(V, v_min, v_max)
    mask = cv2.bitwise_and(mask_h, cv2.bitwise_and(mask_s, mask_v))
    return mask

def label_by_mode(hsv_roi, rules, coverage_min=0.3):
    """
    여러 색 중 픽셀수가 가장 많은 색(모드) 반환.
    """
    h, w = hsv_roi.shape[:2]
    if h == 0 or w == 0:
        return "?"
    area = h * w

    best_label = "?"
    best_count = 0
    for label, rule in rules.items():
        m = build_mask(hsv_roi, rule)
        cnt = cv2.countNonZero(m)
        if cnt > best_count:
            best_count = cnt
            best_label = label

    coverage = best_count / float(area)
    if coverage < coverage_min:
        return "?"
    return best_label

def mirror_rects_horiz(rects, frame_width):
    """좌우 반전된 프레임에서의 (x,y,w,h)로 변환"""
    mirrored = []
    for (x, y, w, h) in rects:
        mx = frame_width - (x + w)
        mirrored.append((mx, y, w, h))
    return mirrored

# 오버레이(디버그)
def draw_grid(frame, rects, labels=None):
    """
    3x3 격자 테두리를 그리고, 각 칸의 '정중앙'에 색 라벨 표시
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2

    for i, (x, y, w, h) in enumerate(rects):
        # 큰 칸 테두리
        cv2.rectangle(frame, (x, y), (x + w, y + h), (210, 210, 210), 2)

        if labels is None or i >= len(labels):
            continue

        lab = str(labels[i])
        bgr = LETTER_BGR.get(lab, (180, 180, 180))  # 라벨 배경색

        # 칸 중심 좌표
        px = x + w // 2
        py = y + h // 2

        # 중앙 배경 사각형 크기 (칸 크기의 약 32%)
        box_w = int(min(w, h) * 0.32)
        box_h = int(min(w, h) * 0.32)
        bx1 = px - box_w // 2
        by1 = py - box_h // 2
        bx2 = bx1 + box_w
        by2 = by1 + box_h

        # 배경 박스(채움) + 테두리
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), bgr, -1)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (30, 30, 30), 2)

        # 라벨 텍스트 중앙 정렬
        (tw, th), base = cv2.getTextSize(lab, font, scale, thickness)
        tx = int(px - tw / 2)
        ty = int(py + th / 2)

        # 가독성: 외곽(검정) + 본문(흰색)
        cv2.putText(frame, lab, (tx, ty), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(frame, lab, (tx, ty), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return frame

def capture(labels9):
    """센터색으로 face 결정 후 CAPTURED/ORDER에 반영"""
    if len(labels9) != 9:
        return None, "인식할 면의 9개의 블록을 3X3 사각형 안에 인식시켜 주세요."
    if any(lab == "?" for lab in labels9):
        return None, "'?' 색을 인식하기 어려운 블록이 존재합니다.\n조명/각도 조정 후 다시 시도해 주세요."

    center = labels9[4]
    face = COLOR_TO_FACE.get(center)
    if face in CAPTURED:
        return None, f"'{center}' 면은 이미 캡처되었습니다."
    CAPTURED[face] = labels9[:]
    CAPTURED_ORDER.append(face)
    return face, None

def undo_capture():
    """최근 캡처 1개 제거"""
    if not CAPTURED_ORDER:
        return None
    last = CAPTURED_ORDER.pop()
    CAPTURED.pop(last, None)
    return last

# ==================== 큐브 유효성 검증 함수들 ====================

def rotate_face_90(face_labels):
    """면을 시계방향 90도 회전"""
    # 0 1 2         6 3 0
    # 3 4 5  ->     7 4 1
    # 6 7 8         8 5 2
    return [
        face_labels[6], face_labels[3], face_labels[0],
        face_labels[7], face_labels[4], face_labels[1],
        face_labels[8], face_labels[5], face_labels[2]
    ]

def rotate_face_n_times(face_labels, n):
    """면을 시계방향 90도씩 n번 회전 (n=0,1,2,3)"""
    result = face_labels[:]
    for _ in range(n % 4):
        result = rotate_face_90(result)
    return result

def extract_corners(captured):
    """
    표준 큐브 넷 기준으로 코너 8개를 정확히 추출.
    각 항목의 colors는 [U/D 스티커, 인접1, 인접2] 순서로 둔다.
    face 인덱스:
        0 1 2
        3 4 5
        6 7 8
    """
    U = captured['U']; D = captured['D']
    F = captured['F']; B = captured['B']
    L = captured['L']; R = captured['R']

    corners = [
        {'colors': [U[0], L[0], B[2]], 'face_names': ['U','L','B'], 'position': 0},  # ULB
        {'colors': [U[2], B[0], R[2]], 'face_names': ['U','B','R'], 'position': 1},  # URB
        {'colors': [U[8], R[0], F[2]], 'face_names': ['U','R','F'], 'position': 2},  # URF
        {'colors': [U[6], F[0], L[2]], 'face_names': ['U','F','L'], 'position': 3},  # ULF

        {'colors': [D[6], B[8], L[6]], 'face_names': ['D','B','L'], 'position': 4},  # DLB
        {'colors': [D[8], R[8], B[6]], 'face_names': ['D','R','B'], 'position': 5},  # DRB
        {'colors': [D[2], F[8], R[6]], 'face_names': ['D','F','R'], 'position': 6},  # DRF
        {'colors': [D[0], L[8], F[6]], 'face_names': ['D','L','F'], 'position': 7},  # DLF
    ]
    return corners

def extract_edges(captured):
    """
    표준 큐브 넷 기준으로 엣지 12개를 정확히 추출.
    colors는 [기준면 스티커(U/D/F/B), 인접면 스티커] 순서로 둔다.
    
    *** 중요: 각 엣지의 face_names 첫 번째가 기준면이 되도록 일관성 유지 ***
    """
    U = captured['U']; D = captured['D']
    F = captured['F']; B = captured['B']
    L = captured['L']; R = captured['R']

    edges = [
        # U면 기준 (4개)
        {'colors': [U[1], B[1]], 'face_names': ['U','B'], 'position': 0},   # UB
        {'colors': [U[5], R[1]], 'face_names': ['U','R'], 'position': 1},   # UR
        {'colors': [U[7], F[1]], 'face_names': ['U','F'], 'position': 2},   # UF
        {'colors': [U[3], L[1]], 'face_names': ['U','L'], 'position': 3},   # UL

        # D면 기준 (4개)
        {'colors': [D[7], B[7]], 'face_names': ['D','B'], 'position': 4},   # DB
        {'colors': [D[5], R[7]], 'face_names': ['D','R'], 'position': 5},   # DR
        {'colors': [D[1], F[7]], 'face_names': ['D','F'], 'position': 6},   # DF
        {'colors': [D[3], L[7]], 'face_names': ['D','L'], 'position': 7},   # DL

        # 중간층 엣지 (4개) - B/F를 기준면으로
        {'colors': [B[3], R[5]], 'face_names': ['B','R'], 'position': 8},   # BR
        {'colors': [F[5], R[3]], 'face_names': ['F','R'], 'position': 9},   # FR
        {'colors': [F[3], L[5]], 'face_names': ['F','L'], 'position': 10},  # FL
        {'colors': [B[5], L[3]], 'face_names': ['B','L'], 'position': 11},  # BL
    ]
    return edges


def face_to_center_color(captured):
    """각 얼굴의 중앙(인덱스 4) 색 → FACE:COLOR 매핑 생성"""
    return {face: labels[4] for face, labels in captured.items()}

def get_solved_corner_position(colors, face2col):
    """색 세트로 코너의 정답 위치 인덱스 반환 (없으면 -1)"""
    s = set(colors)
    U, D, F, B, L, R = (face2col['U'], face2col['D'], face2col['F'],
                        face2col['B'], face2col['L'], face2col['R'])
    solved_sets = [
        {U,L,B}, {U,B,R}, {U,R,F}, {U,F,L},
        {D,B,L}, {D,R,B}, {D,F,R}, {D,L,F},
    ]
    for i, tgt in enumerate(solved_sets):
        if s == tgt:
            return i
    return -1

def get_solved_edge_position(colors, face2col):
    """색 세트로 엣지의 정답 위치 인덱스 반환 (없으면 -1)"""
    s = set(colors)
    U, D, F, B, L, R = (face2col['U'], face2col['D'], face2col['F'],
                        face2col['B'], face2col['L'], face2col['R'])
    solved_sets = [
        {U,B}, {U,R}, {U,F}, {U,L},
        {D,B}, {D,R}, {D,F}, {D,L},
        {B,R}, {F,R}, {F,L}, {B,L},
    ]
    for i, tgt in enumerate(solved_sets):
        if s == tgt:
            return i
    return -1

def count_swaps(permutation):
    """순열을 정렬하는 데 필요한 최소 swap 횟수"""
    n = len(permutation)
    perm = list(permutation)
    swaps = 0
    visited = [False] * n
    for i in range(n):
        if visited[i] or perm[i] == i:
            continue
        cycle_size = 0
        j = i
        while not visited[j]:
            visited[j] = True
            j = perm[j]
            cycle_size += 1
        if cycle_size > 0:
            swaps += cycle_size - 1
    return swaps

def calculate_permutation_parity(pieces, face2col, is_corner=True):
    """순열의 짝/홀수성(0 짝수, 1 홀수)"""
    permutation = []
    for piece in pieces:
        colors = piece['colors']
        if is_corner:
            target_pos = get_solved_corner_position(colors, face2col)
        else:
            target_pos = get_solved_edge_position(colors, face2col)
        if target_pos == -1:
            raise ValueError(f"알 수 없는 조각 (Parity 계산 실패): {colors}")
        permutation.append(target_pos)
    return count_swaps(permutation) % 2

def calculate_corner_orientation_sum(corners, face2col):
    """
    코너 twist 합: U/D 스티커가 조각 내에서 몇 번째 위치에 있는지의 합 (mod 3 == 0이어야 함)
    corners[i]['colors']는 [U/D, Adj1, Adj2] 순서를 가정한다.
    """
    U_center = face2col['U']
    D_center = face2col['D']
    ud_colors = {U_center, D_center}
    total_twist = 0
    for corner in corners:
        colors = corner['colors']  # [ud?, adj1, adj2]
        ud_index = None
        for idx, c in enumerate(colors):
            if c in ud_colors:
                ud_index = idx
                break
        if ud_index is None:
            raise ValueError(f"코너 {colors}에 U/D 색({U_center},{D_center})이 없습니다.")
        total_twist += ud_index
    return total_twist

def calculate_edge_flip_count(edges, face2col):
    """
    뒤집힌 엣지 개수: 각 엣지의 실제 위치를 찾아서 해당 위치의 기준면 색과 비교
    """
    # 각 위치(0~11)별 기준면 정의
    reference_faces = ['U','U','U','U',  # 0:UB, 1:UR, 2:UF, 3:UL
                       'D','D','D','D',  # 4:DB, 5:DR, 6:DF, 7:DL
                       'B','F','F','B']  # 8:BR, 9:FR, 10:FL, 11:BL
    
    flipped = 0
    for edge in edges:
        colors = edge['colors']
        
        # 이 엣지 조각이 실제로 있어야 할 위치(solved position) 찾기
        solved_pos = get_solved_edge_position(colors, face2col)
        if solved_pos == -1:
            raise ValueError(f"알 수 없는 엣지 조각: {colors}")
        
        # 해당 위치의 기준면
        ref_face = reference_faces[solved_pos]
        ref_color = face2col[ref_face]
        
        # colors[0]이 기준면 색이 아니면 flipped
        # (엣지는 2개 색만 가지므로, 기준면 색이 두 번째에 있으면 뒤집힌 것)
        if colors[0] != ref_color:
            flipped += 1
    
    return flipped


def _early_filter_impossible_combos(captured):
    """
    1) 코너는 서로 다른 3색이어야 함
    2) 엣지는 서로 다른 2색이어야 함
    3) 코너/엣지의 색 조합(순서 무관)은 센터색 기반 합법 세트에 포함되어야 함
    4) 동일 세트 중복 금지
    """
    face2col = face_to_center_color(captured)

    def solved_corner_sets():
        U, D, F, B, L, R = (face2col['U'], face2col['D'], face2col['F'],
                            face2col['B'], face2col['L'], face2col['R'])
        return [
            {U,L,B}, {U,B,R}, {U,R,F}, {U,F,L},
            {D,B,L}, {D,R,B}, {D,F,R}, {D,L,F}
        ]

    def solved_edge_sets():
        U, D, F, B, L, R = (face2col['U'], face2col['D'], face2col['F'],
                            face2col['B'], face2col['L'], face2col['R'])
        return [
            {U,B}, {U,R}, {U,F}, {U,L},
            {D,B}, {D,R}, {D,F}, {D,L},
            {B,R}, {F,R}, {F,L}, {B,L}
        ]

    allowed_corner = {frozenset(s) for s in solved_corner_sets()}
    allowed_edge   = {frozenset(s) for s in solved_edge_sets()}

    corners = extract_corners(captured)
    edges   = extract_edges(captured)

    csets, esets = [], []

    for c in corners:
        s = set(c['colors'])
        if len(s) != 3:
            return False, f"불가능한 코너 조합(중복색 포함): {c['colors']}"
        fs = frozenset(s)
        if fs not in allowed_corner:
            return False, f"인접할 수 없는 코너 색 조합: {sorted(list(s))}"
        csets.append(fs)

    for e in edges:
        s = set(e['colors'])
        if len(s) != 2:
            return False, f"불가능한 엣지 조합(중복색 포함): {e['colors']}"
        fs = frozenset(s)
        if fs not in allowed_edge:
            return False, f"인접할 수 없는 엣지 색 조합: {sorted(list(s))}"
        esets.append(fs)

    if len(csets) != len(set(csets)):
        return False, "중복된 코너 세트가 있습니다."
    if len(esets) != len(set(esets)):
        return False, "중복된 엣지 세트가 있습니다."

    return True, None

def is_valid_cube_silent(captured):
    """유효성 검사 (에러 발생 시 False) — 프리필터는 이미 통과했다고 가정"""
    try:
        face2col = face_to_center_color(captured)
        corners = extract_corners(captured)
        edges   = extract_edges(captured)

        # 코너 꼬임
        corner_twist = calculate_corner_orientation_sum(corners, face2col)
        if corner_twist % 3 != 0:
            return False

        # 엣지 뒤집힘
        edge_flip = calculate_edge_flip_count(edges, face2col)
        if edge_flip % 2 != 0:
            return False

        # 순열 parity
        corner_parity = calculate_permutation_parity(corners, face2col, is_corner=True)
        edge_parity   = calculate_permutation_parity(edges,   face2col, is_corner=False)
        if corner_parity != edge_parity:
            return False

        return True
    except Exception as e:
        # 디버깅용 출력
        print(f"[is_valid_cube_silent] 예외 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

def is_valid_cube_debug(captured):
    """디버깅용: 어디서 실패하는지 상세 출력"""
    try:
        face2col = face_to_center_color(captured)
        corners = extract_corners(captured)
        edges   = extract_edges(captured)

        print("\n=== 디버깅 정보 ===")
        print("FACE→CENTER:", face2col)
        print("코너:", [c['colors'] for c in corners])
        print("엣지:", [e['colors'] for e in edges])

        ok, reason = _early_filter_impossible_combos(captured)
        if not ok:
            print(f"❌ 프리필터 실패: {reason}")
            return False

        corner_twist = calculate_corner_orientation_sum(corners, face2col)
        print(f"코너 twist 합: {corner_twist} (mod 3={corner_twist % 3})")
        if corner_twist % 3 != 0:
            print("❌ 코너 twist 실패")
            return False

        edge_flip = calculate_edge_flip_count(edges, face2col)
        print(f"엣지 flip 개수: {edge_flip} ({'짝수' if edge_flip % 2 == 0 else '홀수'})")
        if edge_flip % 2 != 0:
            print("❌ 엣지 flip 실패")
            # 상세 정보 출력
            print("\n엣지별 상세 정보:")
            reference_faces = ['U','U','U','U','D','D','D','D','B','F','F','B']
            for edge in edges:
                solved_pos = get_solved_edge_position(edge['colors'], face2col)
                ref_face = reference_faces[solved_pos]
                ref_color = face2col[ref_face]
                is_flipped = edge['colors'][0] != ref_color
                print(f"  {edge['colors']} → 위치{solved_pos}({ref_face}면 기준), "
                      f"기준색={ref_color}, {'뒤집힌' if is_flipped else '정상'}")
            return False

        corner_parity = calculate_permutation_parity(corners, face2col, is_corner=True)
        edge_parity   = calculate_permutation_parity(edges,   face2col, is_corner=False)
        print(f"코너 parity: {corner_parity} ({'짝수' if corner_parity == 0 else '홀수'})")
        print(f"엣지 parity: {edge_parity} ({'짝수' if edge_parity == 0 else '홀수'})")
        if corner_parity != edge_parity:
            print("❌ Parity 불일치")
            return False

        print("✅ 모든 검증 통과!")
        return True

    except Exception as e:
        print(f"❌ 예외 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

def try_all_orientations(captured):
    """
    모든 면(U,D,F,B,L,R)을 0~3회 회전시키며 유효한 조합 탐색 (총 4096가지)
    개선: 프리필터와 전체 검증을 분리하지 않고 바로 체크
    """
    print("브루트포스 시작: 4096가지 회전 조합 시도 중...")
    t0 = time.time()

    originals = {f: captured[f][:] for f in ['U','D','F','B','L','R']}
    face_order = ['U','D','F','B','L','R']
    count = 0
    prefilter_passed = 0
    full_check_attempted = 0

    for rotations in product(range(4), repeat=6):
        count += 1
        test = {}
        for i, f in enumerate(face_order):
            test[f] = rotate_face_n_times(originals[f], rotations[i])

        # 프리필터 체크
        ok, reason = _early_filter_impossible_combos(test)
        if not ok:
            if count % 512 == 0:
                print(f"진행중... {count}/4096 ({time.time()-t0:.2f}s) - "
                      f"프리필터 통과: {prefilter_passed}개")
            continue
        
        prefilter_passed += 1
        full_check_attempted += 1
        
        # 프리필터 통과한 경우 바로 전체 검증
        if is_valid_cube_silent(test):
            print(f"\n✓ 유효한 조합 발견! (조합 #{count}, 경과 {time.time()-t0:.2f}s)")
            print(f"회전 정보: {dict(zip(face_order, rotations))}")
            print(f"프리필터 통과: {prefilter_passed}개, 전체 검증 시도: {full_check_attempted}개")
            # 상세 확인
            is_valid_cube_debug(test)
            return test, {f: rotations[i] for i, f in enumerate(face_order)}
        
        if count % 512 == 0:
            print(f"진행중... {count}/4096 ({time.time()-t0:.2f}s) - "
                  f"프리필터 통과: {prefilter_passed}개, 전체 검증 시도: {full_check_attempted}개")

    print(f"\n4096가지 모두 시도 완료 (총 {time.time()-t0:.2f}s)")
    print(f"프리필터 통과: {prefilter_passed}개, 전체 검증 시도: {full_check_attempted}개")
    print("→ 유효한 조합 없음")
    
    return None, None


def validate_captured_or_msg(captured, parent=None):
    """
    큐브 물리적 유효성 최종 검증 (색 개수 체크 + 브루트포스 포함)
    통과 시 CAPTURED에 정규화 상태를 반영하고 True 리턴.
    """
    if len(captured) != 6:
        messagebox.showerror("검증 실패", "6개 면을 모두 캡처해야 합니다.", parent=parent)
        return False

    # 색상 개수 9개씩 확인
    color_count = {}
    for labels in captured.values():
        for lab in labels:
            color_count[lab] = color_count.get(lab, 0) + 1
    expected = {'W','Y','R','O','G','B'}
    for col in expected:
        if color_count.get(col, 0) != 9:
            messagebox.showerror("검증 실패",
                f"'{col}' 색상이 {color_count.get(col,0)}개 감지되었습니다. (정확히 9개 필요)",
                parent=parent)
            return False

    normalized, rotations = try_all_orientations(captured)
    if normalized is None:
        messagebox.showerror(
            "검증 실패",
            "모든 회전 조합(4096가지)을 시도했지만 유효한 큐브 상태를 찾지 못했습니다.\n"
            "→ 큐브 조립을 확인하거나 색 인식을 다시 시도해 주세요.",
            parent=parent
        )
        return False

    # 통과 상태 반영
    for f in normalized:
        CAPTURED[f] = normalized[f]
    return True

# ==================== App 클래스 ====================
        
class App:
    def __init__(self, cam_index=0, width=1280, height=720):
        self.window = tk.Tk()
        self.window.title("Color Classificator")
        
        
        # HSV 디버그 모드 스위치 (전역값으로 초기화)
        self.debug_hsv = DEBUG_HSV
        # 키보드 H를 눌러서 디버그 온/오프
        self.window.bind("<Key-h>", self.toggle_debug_hsv)
        
        # 화면 크기 가져오기
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()

        window_width = screen_width
        window_height = screen_height
        
        # 창을 화면 중앙에 배치
        x = - 10 #(screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        
        self.window.geometry(f"{window_width}x{window_height}+{x}+{y}")
        self.window.resizable(True, True)  # 크기 조정 가능하게 변경
        self.window.minsize(800, 600)      # 최소 크기 설정
    
        # 웹캠 초기화
        self.cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        
        # === Webcam controls (Logitech C270 / DirectShow) ============================
        # MJPG로 바꿔 해상도/프레임 유지 안정화
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        # 자동 노출 끄고 수동 노출값으로 고정 (OpenCV on Windows: 0.25=manual, 0.75=auto)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)   # manual
        # C270 노출 스케일은 보통 -13(어둠) ~ -1(밝음) 범위. 흰색 날림 방지용으로 약간 어둡게.
        self.cap.set(cv2.CAP_PROP_EXPOSURE, -6)          # 필요 시 -7 ~ -4 사이에서 조정

        # 자동 화이트밸런스 끄고 색온도 고정 (4000~5000K 권장)
        self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        self.cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 4500)

        # (선택) 이 값들이 지원되지 않는 경우가 있어서, 실패하더라도 무시해도 됨
        # gain을 낮추면 노이즈가 줄고 색이 안정됨 (장면이 너무 어두우면 조명으로 보완)
        self.cap.set(cv2.CAP_PROP_GAIN, 0)

        # (C270은 자동초점/수동초점 자체가 없음. 아래는 다른 모델 호환용으로 무시돼도 OK)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        # self.cap.set(cv2.CAP_PROP_FOCUS, 0)  # C270에선 적용 안 됨
        # ============================================================================

        # 프레임 버퍼 딜레이 줄이기 (가능하면)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        ok, frame = self.cap.read()
        if not ok:
            messagebox.showerror("Camera Error","웹캠을 열 수 없습니다.")
            self.window.destroy()
            sys.exit(1)

        cam_h, cam_w = frame.shape[:2]
        
        # 원본 크기 저장 (ROI 계산용)
        self.cam_w = cam_w
        self.cam_h = cam_h
        
        # 카메라 비율 유지하면서 고정 크기 설정
        cam_ratio = cam_w / cam_h
        target_height = 480  # 고정 높이
        self.display_h = target_height
        self.display_w = int(target_height * cam_ratio)  # 비율에 맞춰 너비 계산
        
        # 캡처된 면 캔버스 고정 크기
        self.captured_w = 400
        self.captured_h = 600
        
        # UI 요소들을 중앙에 배치하기 위한 계산
        content_width = self.display_w + 50 + self.captured_w  # 캠 + 간격 + 캔버스
        content_height = max(self.display_h + 60, self.captured_h)  # 버튼 공간 포함
        
        start_x = (window_width - content_width) // 2
        start_y = (window_height - content_height) // 2
        
        # 비디오 캔버스
        self.video_canvas = tk.Canvas(
            self.window, 
            width=self.display_w, 
            height=self.display_h, 
            bg="black", 
            highlightthickness=0
        )
        self.video_canvas.place(x=start_x, y=start_y)
        
        self.tk_img = None
        self.video_img_handle = None

        # 캡처된 면 표시 캔버스 (비디오 오른쪽에 배치)
        captured_x = start_x + self.display_w + 50
        captured_y = start_y
        self.captured_canvas = tk.Canvas(
            self.window, 
            width=self.captured_w, 
            height=self.captured_h, 
            bg="black", 
            highlightthickness=0
        )
        self.captured_canvas.place(x=captured_x, y=captured_y)

        # 버튼 프레임 (비디오 화면 아래 중앙에 배치)
        btn = tk.Frame(self.window)
        btn_x = start_x + self.display_w - 305  # 3개 버튼 중앙 정렬
        btn_y = start_y + self.display_h + 20  # 비디오 바로 아래
        btn.place(x=btn_x, y=btn_y)

        # 버튼 크기
        btn_width = 12
        font_size = 10
        
        self.undo_btn = tk.Button(
            btn, text="되돌리기", width=btn_width, 
            font=("맑은 고딕", font_size), command=self.on_undo
        )
        self.undo_btn.grid(row=0, column=0, padx=5)

        self.capture_btn = tk.Button(
            btn, text="캡처하기", width=btn_width,
            font=("맑은 고딕", font_size), command=self.on_capture
        )
        self.capture_btn.grid(row=0, column=1, padx=5)

        self.exit_btn = tk.Button(
            btn, text="종료", width=btn_width,
            font=("맑은 고딕", font_size), command=self.on_close
        )
        self.exit_btn.grid(row=0, column=2, padx=5)

        self.update_btn()

        # ROI는 원본 카메라 크기 기준으로 계산
        self.rects = calc_grid((self.cam_h, self.cam_w), box_ratio=0.6)

        self.draw_captured()
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)
        self.loop()
        self.window.mainloop()

    def loop(self):
        ok, frame = self.cap.read()
        if ok:
            frame_proc = frame

            # --- 라벨 계산은 원본 기준 (그대로 유지) ---
            blurred = apply_blur(frame_proc, BLUR_KERNEL)
            blurred_hsv = to_hsv(blurred)

            # ★ 중앙 패치 HSV 평균값 계산 (디버그용)
            H_all, W_all = blurred_hsv.shape[:2]
            cx_all, cy_all = W_all // 2, H_all // 2
            patch_size = 20  # 20x20 패치
            px1 = max(cx_all - patch_size, 0)
            py1 = max(cy_all - patch_size, 0)
            px2 = min(cx_all + patch_size, W_all)
            py2 = min(cy_all + patch_size, H_all)
            center_patch = blurred_hsv[py1:py2, px1:px2]
            if center_patch.size > 0:
                h_mean, s_mean, v_mean = np.mean(center_patch.reshape(-1, 3), axis=0)
            else:
                h_mean, s_mean, v_mean = 0, 0, 0

            labels = []
            for (x, y, w, h) in self.rects:
                cx, cy, cw, ch = center_rect((x, y, w, h), frame_proc.shape, center_ratio=0.6)
                roi = blurred_hsv[cy:cy+ch, cx:cx+cw]
                labels.append(label_by_mode(roi, COLOR_RULES, coverage_min=0.25))

            # --- 표시용 프레임을 먼저 준비 (미러/비미러) ---
            if MIRROR_PREVIEW:
                disp = cv2.flip(frame_proc, 1)
                rects_to_draw = mirror_rects_horiz(self.rects, frame_proc.shape[1])
            else:
                disp = frame_proc.copy()
                rects_to_draw = self.rects

            # ★ 디버그 모드일 때만 HSV 텍스트를 화면에 표시
            if self.debug_hsv:
                debug_text = f"H:{h_mean:.0f} S:{s_mean:.0f} V:{v_mean:.0f}"
                cv2.putText(disp, debug_text, (18, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(disp, debug_text, (18, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 2, cv2.LINE_AA)
                
            # 캡처 인덱스(카메라 좌측 상단) 표시
            if len(CAPTURED) < 6:
                idx_now = len(CAPTURED_ORDER) + 1
                idx_text = f"[{idx_now}]"
                cv2.putText(disp, idx_text, (18, 58), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 0), 6, cv2.LINE_AA)
                cv2.putText(disp, idx_text, (18, 58), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 3, cv2.LINE_AA)

            out = draw_grid(disp, rects_to_draw, labels)

            # 고정 크기 리사이즈 및 Tk 표시 (기존과 동일)
            out_resized = cv2.resize(out, (self.display_w, self.display_h))
            rgb = cv2.cvtColor(out_resized, cv2.COLOR_BGR2RGB)
            self.tk_img = ImageTk.PhotoImage(Image.fromarray(rgb))
            
            if self.video_img_handle is None:
                self.video_img_handle = self.video_canvas.create_image(
                    0, 0, image=self.tk_img, anchor="nw"
                )
            else:
                self.video_canvas.itemconfig(self.video_img_handle, image=self.tk_img)

            self.current_labels = labels[:]

        self.window.after(15, self.loop)

    def draw_captured(self):
        c = self.captured_canvas
        c.delete("all")

        cols, rows = 2, 3
        gap = 14  # 고정 간격
        margin_min = 14  # 고정 여백
        title_height = 24  # 고정 타이틀 높이

        bs_w = (self.captured_w - margin_min*2 - gap*(cols-1)) // cols
        bs_h = (self.captured_h - margin_min*2 - gap*(rows-1) - title_height) // rows
        bs = min(bs_w, bs_h)

        total_w = cols*bs + (cols-1)*gap
        total_h = rows*bs + (rows-1)*gap
        margin_x = max((self.captured_w - total_w) // 2, 0)
        margin_y = max((self.captured_h - total_h - title_height) // 2, 0) + title_height

        # 고정 폰트 크기
        title_font_size = 12
        label_font_size = 10
        
        c.create_text(
            self.captured_w//2, title_height//2,
            text=f"캡처한 면: {len(CAPTURED)}/6",
            fill="white", font=("맑은 고딕", title_font_size, "bold")
        )

        for idx, face in enumerate(CAPTURED_ORDER[:6]):
            labs = CAPTURED.get(face)
            row = idx // cols
            col = idx % cols
            ox = margin_x + col * (bs + gap)
            oy = margin_y + row * (bs + gap)

            c.create_rectangle(ox, oy, ox + bs, oy + bs, outline="white", width=1)

            c.create_text(
                ox + bs // 2, oy - 8, text=f"[{idx+1}]",
                fill="white", font=("맑은 고딕", label_font_size, "bold")
            )

            xs = [ox + round(k * bs / 3) for k in range(4)]
            ys = [oy + round(k * bs / 3) for k in range(4)]

            for i in range(3):
                for j in range(3):
                    lab = labs[i*3 + j]
                    b, g, r = LETTER_BGR.get(lab, (128, 128, 128))
                    rgb = f"#{r:02x}{g:02x}{b:02x}"
                    c.create_rectangle(
                        xs[j], ys[i], xs[j+1], ys[i+1],
                        fill=rgb, outline="black", width=2
                    )

    def on_capture(self):
        if not hasattr(self, "current_labels") or len(self.current_labels) != 9:
            messagebox.showwarning("캡처 실패", "라벨이 준비되지 않았습니다.")
            return

        _, err = capture(self.current_labels)

        if err:
            messagebox.showwarning("캡처 실패", err)

        self.draw_captured()
        self.update_btn()

    def on_undo(self):
        last = undo_capture()
        if not last:
            messagebox.showinfo("Undo Error", "되돌릴 수 있는 캡처된 면이 없습니다.")

        self.draw_captured()
        self.update_btn()

    def on_done(self):
        # 유효성 검사
        if not validate_captured_or_msg(CAPTURED, parent=self.window):
            return

        try:
            captured_for_solver = {
                'U': CAPTURED['U'],
                'R': CAPTURED['R'],
                'F': CAPTURED['F'],
                'D': CAPTURED['D'],
                'L': CAPTURED['L'],
                'B': CAPTURED['B'],
            }
            moves = solve_moves(captured_for_solver)
        except Exception as e:
            messagebox.showerror("솔버 오류", str(e), parent=self.window)
            return
        
        # 최소화
        self.window.iconify()
        self.window.update()
        
        try:
            simulator(CAPTURED)
        finally:
            import time
            time.sleep(0.3)
            self.window.deiconify()
            self.window.lift()
            self.window.focus_force()

    def on_close(self):
        try:
            if self.cap and self.cap.isOpened():
                self.cap.release()
        finally:
            self.window.destroy()

    def update_btn(self):
        if len(CAPTURED) == 6:
            self.capture_btn.config(text="완료", command=self.on_done)
        else:
            self.capture_btn.config(text="캡처하기", command=self.on_capture)
    
    def toggle_debug_hsv(self, event=None):
        # True <-> False 토글
        self.debug_hsv = not self.debug_hsv
        state = "ON" if self.debug_hsv else "OFF"
        print(f"[DEBUG_HSV] HSV 디버그 모드: {state}")



if __name__ == "__main__":
    App(cam_index=0, width=960, height=540)