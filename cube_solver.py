from typing import Dict, List

# 네 프로젝트의 고정 매핑(센터색 -> 면라벨)
COLOR_TO_FACE = {
    "Y": "U",
    "R": "F",
    "G": "R",
    "B": "L",
    "O": "B",
    "W": "D"
}

# kociemba가 요구하는 직렬화 순서
FACE_ORDER_KOCIEMBA = ["U", "R", "F", "D", "L", "B"]

def _captured_rekey_URFDLB(captured: Dict[str, List[str]]) -> Dict[str, List[str]]:
    needed = set(['U','R','F','D','L','B'])
    if not needed.issubset(set(captured.keys())):
        missing = sorted(list(needed - set(captured.keys())))
        raise ValueError(f"CAPTURED에 키가 없습니다: {missing}")
    return {
        'U': captured['U'],
        'R': captured['R'],
        'F': captured['F'],
        'D': captured['D'],
        'L': captured['L'],
        'B': captured['B'],
    }

def _col2face(c: str) -> str:
    if c not in COLOR_TO_FACE:
        raise ValueError(f"알 수 없는 색 라벨: {c}")
    return COLOR_TO_FACE[c]

def captured_to_facelets_URFDLB(captured: Dict[str, List[str]]) -> str:
    """
    CAPTURED(U,R,F,D,L,B 키 각각 9개 색) -> 54자 facelets(URFDLB)
    각 스티커의 '색'을 COLOR_TO_FACE 매핑을 통해 '면문자'로 변환해서 직렬화.
    """
    cap = _captured_rekey_URFDLB(captured)
    for face, lst in cap.items():
        if len(lst) != 9:
            raise ValueError(f"{face} 면 스티커 수가 9가 아닙니다: {len(lst)}")
    out = []
    for face in FACE_ORDER_KOCIEMBA:
        out.extend(_col2face(c) for c in cap[face])
    return "".join(out)

def _is_solved_facelets(facelets: str) -> bool:
    """
    직렬화된 54자 facelets가 해결 상태인지 검사.
    URFDLB 순서로 각 9자 블록이 동일 문자이면 true.
    """
    if len(facelets) != 54:
        return False
    for i, face in enumerate(FACE_ORDER_KOCIEMBA):
        block = facelets[i*9:(i+1)*9]
        if any(ch != face for ch in block):
            return False
    return True

def parse_moves(seq: str) -> List[str]:
    """'R U R' U' F2' 같은 문자열을 토큰 리스트로."""
    tokens = []
    for tok in seq.replace("\n"," ").split():
        tok = tok.strip()
        if not tok:
            continue
        if tok[-1] in ["'", "2"]:
            base, suf = tok[:-1], tok[-1]
        else:
            base, suf = tok, ""
        if base not in "UDLRFB":
            continue
        tokens.append(base + suf)
    return tokens

def invert_move(m: str) -> str:
    """'R' -> 'R'' , 'R'' -> 'R', 'R2' -> 'R2'"""
    if m.endswith("2"):
        return m
    if m.endswith("'"):
        return m[:-1]
    return m + "'"

def invert_moves(moves: List[str]) -> List[str]:
    """시퀀스 역순/반대회전으로 되감기용"""
    return [invert_move(m) for m in reversed(moves)]

def solve_moves(captured: Dict[str, List[str]]) -> List[str]:
    """
    외부에서 호출하는 주 함수:
    - CAPTURED(검증/정규화 완료 상태)를 받아
    - 해법 시퀀스(리스트)를 반환
    해결 상태면 kociemba를 사용하지 않고 빈 리스트를 반환.
    """
    facelets = captured_to_facelets_URFDLB(captured)

    if _is_solved_facelets(facelets):
        return []

    try:
        import kociemba  # pip install kociemba
    except Exception as e:
        raise RuntimeError(
            "큐브 솔버(kociemba)가 설치되어 있지 않습니다.\n"
            "터미널에서 'pip install kociemba' 실행 후 다시 시도하세요."
        ) from e

    sol_str = kociemba.solve(facelets)   # 예: "R U R' U' F2 ..."
    return parse_moves(sol_str)
