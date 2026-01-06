# /main.py
"""
Gomoku (5 taş yan yana) - Android GUI (Kivy) + Çok Güçlü YZ (TR)
- Tahta: 13x13 veya 15x15
- Mod: İnsan(Siyah), İnsan(Beyaz), YZ vs YZ
- YZ: iterative deepening + alpha-beta + TT + killer + agresif move ordering
- Ek: VCF/VCT (zorunlu kazanma tehdidi araması)
  - Sürekli "bir sonraki hamlede kazanırım" tehditleri üzerinden hızlı forced-win bulur.
  - Double-threat (iki farklı winning-move) algılar.

Not:
- Renju kısıtları YOK (klasik Gomoku).
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.metrics import dp
from kivy.properties import NumericProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.slider import Slider
from kivy.uix.spinner import Spinner
from kivy.uix.widget import Widget
from kivy.graphics import Color, Ellipse, Line, Rectangle

WIN_LEN = 5
EMPTY = 0
BLACK = 1
WHITE = 2

DIRS_4 = ((1, 0), (0, 1), (1, 1), (1, -1))

INF = 10**15
WIN_SCORE = 10**14


def opp(p: int) -> int:
    return WHITE if p == BLACK else BLACK


def now_perf() -> float:
    return time.perf_counter()


@dataclass(frozen=True)
class State:
    grid: Tuple[int, ...]  # len = n*n
    turn: int              # BLACK/WHITE
    last_move: int         # -1 if none
    winner: int            # 0 none, BLACK/WHITE

    def is_over(self) -> bool:
        return self.winner != 0 or all(v != EMPTY for v in self.grid)


class Rules:
    def __init__(self, n: int):
        self.n = n

    def idx(self, x: int, y: int) -> int:
        return y * self.n + x

    def xy(self, i: int) -> Tuple[int, int]:
        return i % self.n, i // self.n

    def inside(self, x: int, y: int) -> bool:
        return 0 <= x < self.n and 0 <= y < self.n

    def center_index(self) -> int:
        c = self.n // 2
        return self.idx(c, c)

    def initial_state(self) -> State:
        return State(grid=tuple([EMPTY] * (self.n * self.n)), turn=BLACK, last_move=-1, winner=0)

    def _count_dir(self, grid: Tuple[int, ...], x: int, y: int, dx: int, dy: int, player: int) -> int:
        c = 0
        x += dx
        y += dy
        while self.inside(x, y) and grid[self.idx(x, y)] == player:
            c += 1
            x += dx
            y += dy
        return c

    def is_win_after_move(self, grid: Tuple[int, ...], move: int, player: int) -> bool:
        x, y = self.xy(move)
        for dx, dy in DIRS_4:
            c = 1
            c += self._count_dir(grid, x, y, dx, dy, player)
            c += self._count_dir(grid, x, y, -dx, -dy, player)
            if c >= WIN_LEN:
                return True
        return False

    def apply_move(self, s: State, move: int) -> State:
        if s.winner != 0:
            return s
        if s.grid[move] != EMPTY:
            raise ValueError("Dolu kareye hamle.")
        g = list(s.grid)
        g[move] = s.turn
        g2 = tuple(g)
        win = s.turn if self.is_win_after_move(g2, move, s.turn) else 0
        return State(grid=g2, turn=opp(s.turn), last_move=move, winner=win)

    def gen_candidates_radius(self, grid: Tuple[int, ...], radius: int = 2) -> List[int]:
        stones = [i for i, v in enumerate(grid) if v != EMPTY]
        if not stones:
            return [self.center_index()]

        mark = [False] * (self.n * self.n)
        for i in stones:
            x, y = self.xy(i)
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    nx, ny = x + dx, y + dy
                    if self.inside(nx, ny):
                        j = self.idx(nx, ny)
                        if grid[j] == EMPTY:
                            mark[j] = True

        out = [i for i, m in enumerate(mark) if m]
        if not out:
            out = [i for i, v in enumerate(grid) if v == EMPTY]
        return out

    def local_run_info(self, grid: Tuple[int, ...], mv: int, player: int, dx: int, dy: int) -> Tuple[int, int]:
        x, y = self.xy(mv)

        left = 0
        lx, ly = x - dx, y - dy
        while self.inside(lx, ly) and grid[self.idx(lx, ly)] == player:
            left += 1
            lx, ly = lx - dx, ly - dy

        right = 0
        rx, ry = x + dx, y + dy
        while self.inside(rx, ry) and grid[self.idx(rx, ry)] == player:
            right += 1
            rx, ry = rx + dx, ry + dy

        run_len = 1 + left + right

        open_ends = 0
        if self.inside(lx, ly) and grid[self.idx(lx, ly)] == EMPTY:
            open_ends += 1
        if self.inside(rx, ry) and grid[self.idx(rx, ry)] == EMPTY:
            open_ends += 1

        return run_len, open_ends

    def tactical_signature(self, grid: Tuple[int, ...], mv: int, player: int) -> Tuple[int, int, int, int]:
        open4 = closed4 = open3 = closed3 = 0
        for dx, dy in DIRS_4:
            run_len, open_ends = self.local_run_info(grid, mv, player, dx, dy)
            if run_len >= 5:
                continue
            if run_len == 4:
                if open_ends == 2:
                    open4 += 1
                elif open_ends == 1:
                    closed4 += 1
            elif run_len == 3:
                if open_ends == 2:
                    open3 += 1
                elif open_ends == 1:
                    closed3 += 1
        return open4, closed4, open3, closed3

    def immediate_wins(self, grid: Tuple[int, ...], player: int, moves: List[int]) -> List[int]:
        wins: List[int] = []
        for mv in moves:
            if grid[mv] != EMPTY:
                continue
            g = list(grid)
            g[mv] = player
            g2 = tuple(g)
            if self.is_win_after_move(g2, mv, player):
                wins.append(mv)
        return wins

    def eval_segment(self, seg: List[int]) -> int:
        b = seg.count(BLACK)
        w = seg.count(WHITE)
        if b > 0 and w > 0:
            return 0
        if b == 0 and w == 0:
            return 0
        weights = {1: 10, 2: 200, 3: 5000, 4: 250000, 5: WIN_SCORE}
        if w == 0:
            return weights[b]
        return -weights[w]

    def evaluate(self, grid: Tuple[int, ...], turn: int) -> int:
        n = self.n
        g = grid
        total = 0

        for y in range(n):
            base = y * n
            for x in range(n - 4):
                total += self.eval_segment([g[base + x + i] for i in range(5)])

        for x in range(n):
            for y in range(n - 4):
                total += self.eval_segment([g[(y + i) * n + x] for i in range(5)])

        for y in range(n - 4):
            for x in range(n - 4):
                total += self.eval_segment([g[(y + i) * n + (x + i)] for i in range(5)])

        for y in range(4, n):
            for x in range(n - 4):
                total += self.eval_segment([g[(y - i) * n + (x + i)] for i in range(5)])

        c = n // 2
        center_bonus = 0
        for i, v in enumerate(g):
            if v == EMPTY:
                continue
            x, y = self.xy(i)
            dist = abs(x - c) + abs(y - c)
            bonus = max(0, 10 - dist)
            center_bonus += bonus if v == BLACK else -bonus

        eval_black = total + center_bonus
        return eval_black if turn == BLACK else -eval_black


class Engine:
    EXACT = 0
    LOWER = 1
    UPPER = 2

    @dataclass
    class TTEntry:
        depth: int
        value: int
        flag: int
        best: int

    @dataclass
    class ThreatEntry:
        depth: int
        ok: bool
        best: int

    def __init__(self, rules: Rules) -> None:
        self.rules = rules
        self.deadline = 0.0
        self.tt: Dict[int, Engine.TTEntry] = {}
        self.threat_tt: Dict[int, Engine.ThreatEntry] = {}
        self._zob = self._make_zobrist(rules.n)
        self.killers: List[List[int]] = [[-1, -1] for _ in range(96)]  # max ply bound

    def _make_zobrist(self, n: int) -> List[List[int]]:
        z = [[0, 0] for _ in range(n * n)]
        seed = 0xC0FFEE1234ABCDE1
        for i in range(n * n):
            for p in range(2):
                seed ^= (seed << 13) & 0xFFFFFFFFFFFFFFFF
                seed ^= (seed >> 7) & 0xFFFFFFFFFFFFFFFF
                seed ^= (seed << 17) & 0xFFFFFFFFFFFFFFFF
                z[i][p] = seed & 0xFFFFFFFFFFFFFFFF
        return z

    def hash(self, grid: Tuple[int, ...], turn: int) -> int:
        h = 0
        for i, v in enumerate(grid):
            if v == BLACK:
                h ^= self._zob[i][0]
            elif v == WHITE:
                h ^= self._zob[i][1]
        h ^= 0x9E3779B97F4A7C15 if turn == BLACK else 0xD1B54A32D192ED03
        return h & 0xFFFFFFFFFFFFFFFF

    def _check(self) -> None:
        if now_perf() >= self.deadline:
            raise TimeoutError

    def _difficulty_to_limits(self, d: int) -> Tuple[float, int, int, int]:
        """
        Returns:
        - time_limit_sec
        - max_depth (alpha-beta)
        - move_cap (candidate cap)
        - threat_depth (VCF/VCT ply depth for attacker moves)
        """
        d = max(1, min(10, int(d)))
        time_limit = 0.12 + d * 0.28        # ~0.4s..2.92s
        max_depth = 2 + (d // 2)            # 2..7
        move_cap = 14 + d * 4               # 18..54
        threat_depth = 3 + (d // 3)         # 3..6
        return time_limit, max_depth, move_cap, threat_depth

    # ---------- VCF/VCT (Threat Search) ----------

    def _threat_hash(self, grid: Tuple[int, ...], turn: int, depth: int, attacker: int) -> int:
        # separate hash space for threat search
        h = self.hash(grid, turn)
        h ^= (attacker * 0xA5A5A5A5A5A5A5A5) & 0xFFFFFFFFFFFFFFFF
        h ^= (depth * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        return h & 0xFFFFFFFFFFFFFFFF

    def find_forced_win(self, grid: Tuple[int, ...], turn: int, last_move: int, threat_depth: int, cap: int) -> Optional[int]:
        """
        Returns best move if forced win is found by VCF/VCT-like threat chain search.
        """
        attacker = turn
        h = self._threat_hash(grid, turn, threat_depth, attacker)
        ent = self.threat_tt.get(h)
        if ent and ent.depth >= threat_depth:
            return ent.best if ent.ok and ent.best >= 0 else None

        best = self._vcf_vct(grid, turn, last_move, attacker, threat_depth, cap)
        self.threat_tt[h] = Engine.ThreatEntry(depth=threat_depth, ok=(best is not None), best=(best if best is not None else -1))
        return best

    def _vcf_vct(
        self,
        grid: Tuple[int, ...],
        turn: int,
        last_move: int,
        attacker: int,
        depth: int,
        cap: int,
    ) -> Optional[int]:
        """
        Attack/defense threat recursion.
        - Attacker tries to force win by moves that lead to "immediate win threat next".
        - Defender is restricted to forced blocks of attacker's immediate winning moves.
        - If attacker creates >=2 immediate winning moves next, treat as win (double-threat).
        """
        self._check()
        r = self.rules

        # If attacker to play and can win immediately => success
        moves0 = r.gen_candidates_radius(grid, radius=2)
        wins_now = r.immediate_wins(grid, turn, moves0)
        if wins_now:
            return wins_now[0] if turn == attacker else None

        if depth <= 0:
            return None

        # Generate candidate threats for current side-to-move
        moves = self._generate_moves(grid, turn, last_move, cap)

        # Attacker's turn: pick forcing move that makes defender forced
        if turn == attacker:
            for mv in moves:
                self._check()
                if grid[mv] != EMPTY:
                    continue
                g = list(grid)
                g[mv] = turn
                g2 = tuple(g)

                if r.is_win_after_move(g2, mv, turn):
                    return mv

                # After attacker move, compute attacker's immediate wins next
                cand2 = r.gen_candidates_radius(g2, radius=2)
                wins_next = r.immediate_wins(g2, attacker, cand2)

                # Double-threat: defender can't block two distinct winning cells at once
                if len(wins_next) >= 2:
                    return mv

                # Forcing condition: must at least create 1 immediate winning move next (VCF step)
                if len(wins_next) == 0:
                    continue

                # Defender replies: must block that winning cell (only)
                block_moves = wins_next[:]
                # If multiple, defender can choose among them; handle list
                ok = True
                for dm in block_moves:
                    self._check()
                    if g2[dm] != EMPTY:
                        continue
                    gd = list(g2)
                    gd[dm] = opp(attacker)
                    gd2 = tuple(gd)
                    # attacker continues
                    cont = self._vcf_vct(gd2, attacker, dm, attacker, depth - 1, cap)
                    if cont is None:
                        ok = False
                        break
                if ok:
                    return mv

            return None

        # Defender's turn in threat search: if attacker has forced immediate win threats, we already
        # constrain defender replies at caller side. If we reach here, treat as "no forced line".
        return None

    # ---------- Normal Alpha-Beta ----------

    def best_move(self, s: State, difficulty: int) -> Optional[int]:
        if s.is_over():
            return None

        time_limit, max_depth, move_cap, threat_depth = self._difficulty_to_limits(difficulty)
        self.deadline = now_perf() + max(0.05, time_limit)

        r = self.rules
        grid = s.grid
        turn = s.turn

        # root candidate list
        root_moves = self._generate_moves(grid, turn, s.last_move, move_cap)
        if not root_moves:
            return None

        # immediate win now
        wins = r.immediate_wins(grid, turn, root_moves)
        if wins:
            return wins[0]

        # immediate block (opponent win next)
        op = opp(turn)
        opp_wins = r.immediate_wins(grid, op, root_moves)
        if opp_wins:
            return opp_wins[0]

        # VCF/VCT forced win first (very strong)
        try:
            forced = self.find_forced_win(grid, turn, s.last_move, threat_depth=threat_depth, cap=min(move_cap, 36))
        except TimeoutError:
            forced = None
        if forced is not None and grid[forced] == EMPTY:
            return forced

        # iterative deepening alpha-beta
        best: Optional[int] = None
        for depth in range(1, max_depth + 1):
            try:
                _, mv = self._root(grid, turn, depth, s.last_move, move_cap)
            except TimeoutError:
                break
            if mv is not None:
                best = mv
        return best

    def _generate_moves(self, grid: Tuple[int, ...], turn: int, last_move: int, cap: int) -> List[int]:
        """
        Threat-based generator + aggressive ordering.
        Also: if opponent has immediate win, only blocking moves.
        """
        r = self.rules
        candidates = r.gen_candidates_radius(grid, radius=2)
        op = opp(turn)

        opp_wins = r.immediate_wins(grid, op, candidates)
        if opp_wins:
            # any of those squares must be occupied
            blocks = [mv for mv in opp_wins if grid[mv] == EMPTY]
            return blocks[:cap]

        cur_wins = r.immediate_wins(grid, turn, candidates)
        if cur_wins:
            return cur_wins[:cap]

        def score(mv: int) -> int:
            if grid[mv] != EMPTY:
                return -INF

            o4, c4, o3, c3 = r.tactical_signature(grid, mv, turn)
            o4_op, c4_op, o3_op, c3_op = r.tactical_signature(grid, mv, op)

            sc = 0
            # attack
            sc += o4 * 50_000_000
            sc += c4 * 10_000_000
            sc += o3 * 2_000_000
            sc += c3 * 200_000
            # defense
            sc += o4_op * 55_000_000
            sc += c4_op * 12_000_000
            sc += o3_op * 2_500_000
            sc += c3_op * 250_000

            x, y = r.xy(mv)
            c = r.n // 2
            sc += max(0, 30 - (abs(x - c) + abs(y - c))) * 2000
            if last_move >= 0:
                lx, ly = r.xy(last_move)
                sc += max(0, 12 - (abs(x - lx) + abs(y - ly))) * 1200
            return sc

        candidates.sort(key=score, reverse=True)
        return candidates[:cap]

    def _root(self, grid: Tuple[int, ...], turn: int, depth: int, last_move: int, cap: int) -> Tuple[int, Optional[int]]:
        alpha = -INF
        beta = INF
        best_val = -INF
        best_mv: Optional[int] = None

        moves = self._generate_moves(grid, turn, last_move, cap)
        if not moves:
            return self.rules.evaluate(grid, turn), None

        for mv in moves:
            self._check()
            g = list(grid)
            g[mv] = turn
            g2 = tuple(g)

            if self.rules.is_win_after_move(g2, mv, turn):
                return WIN_SCORE, mv

            val = -self._negamax(g2, opp(turn), depth - 1, -beta, -alpha, mv, ply=1, cap=cap)
            if val > best_val:
                best_val, best_mv = val, mv
            if val > alpha:
                alpha = val
            if alpha >= beta:
                break

        return best_val, best_mv

    def _negamax(
        self,
        grid: Tuple[int, ...],
        turn: int,
        depth: int,
        alpha: int,
        beta: int,
        last_move: int,
        ply: int,
        cap: int,
    ) -> int:
        self._check()

        h = self.hash(grid, turn)
        entry = self.tt.get(h)
        alpha0 = alpha

        if entry and entry.depth >= depth:
            if entry.flag == self.EXACT:
                return entry.value
            if entry.flag == self.LOWER:
                alpha = max(alpha, entry.value)
            elif entry.flag == self.UPPER:
                beta = min(beta, entry.value)
            if alpha >= beta:
                return entry.value

        if depth <= 0:
            return self.rules.evaluate(grid, turn)

        if all(v != EMPTY for v in grid):
            return 0

        moves = self._generate_moves(grid, turn, last_move, cap)
        if not moves:
            return self.rules.evaluate(grid, turn)

        # TT best + killer first
        prefer = entry.best if entry else -1
        k1, k2 = (-1, -1)
        if ply < len(self.killers):
            k1, k2 = self.killers[ply]

        def order_bonus(mv: int) -> int:
            b = 0
            if mv == prefer:
                b += 3_000_000_000
            if mv == k1:
                b += 2_000_000_000
            if mv == k2:
                b += 1_000_000_000
            return b

        moves.sort(key=order_bonus, reverse=True)

        best_val = -INF
        best_mv = -1

        for mv in moves:
            g = list(grid)
            g[mv] = turn
            g2 = tuple(g)

            if self.rules.is_win_after_move(g2, mv, turn):
                val = WIN_SCORE - ply * 1000
            else:
                val = -self._negamax(g2, opp(turn), depth - 1, -beta, -alpha, mv, ply + 1, cap)

            if val > best_val:
                best_val, best_mv = val, mv
            if val > alpha:
                alpha = val
            if alpha >= beta:
                if ply < len(self.killers) and mv not in self.killers[ply]:
                    self.killers[ply] = [mv, self.killers[ply][0]]
                break

        flag = self.EXACT
        if best_val <= alpha0:
            flag = self.UPPER
        elif best_val >= beta:
            flag = self.LOWER

        self.tt[h] = Engine.TTEntry(depth=depth, value=best_val, flag=flag, best=best_mv)
        return best_val


class GomokuBoard(Widget):
    last_move = NumericProperty(-1)

    def __init__(self, controller=None, **kwargs):
        super().__init__(**kwargs)
        self.controller = controller
        self.bind(pos=self._redraw, size=self._redraw)

    def _cell_from_touch(self, x: float, y: float) -> Optional[Tuple[int, int]]:
        ctrl = self.controller
        if ctrl is None:
            return None
        n = ctrl.rules.n

        bx, by = self.pos
        w, h = self.size
        side = min(w, h)
        ox = bx + (w - side) / 2
        oy = by + (h - side) / 2
        if not (ox <= x <= ox + side and oy <= y <= oy + side):
            return None

        cell = side / (n - 1)
        cx = int(round((x - ox) / cell))
        cy = int(round((y - oy) / cell))
        if 0 <= cx < n and 0 <= cy < n:
            return cx, cy
        return None

    def on_touch_down(self, touch):
        if not self.collide_point(*touch.pos):
            return super().on_touch_down(touch)
        if self.controller is None:
            return True
        cell = self._cell_from_touch(*touch.pos)
        if cell is None:
            return True
        x, y = cell
        mv = self.controller.rules.idx(x, y)
        self.controller.on_human_move(mv)
        return True

    def _redraw(self, *_):
        self.canvas.clear()

        ctrl = self.controller
        if ctrl is None or not hasattr(ctrl, "state"):
            return

        rls = ctrl.rules
        n = rls.n
        s: State = ctrl.state
        grid = s.grid

        bx, by = self.pos
        w, h = self.size
        side = min(w, h)
        ox = bx + (w - side) / 2
        oy = by + (h - side) / 2
        cell = side / (n - 1)

        with self.canvas:
            Color(0.55, 0.40, 0.22, 1)
            Rectangle(pos=(ox, oy), size=(side, side))

            Color(0.08, 0.05, 0.02, 1)
            for i in range(n):
                y = oy + i * cell
                Line(points=[ox, y, ox + side, y], width=1.1)
                x = ox + i * cell
                Line(points=[x, oy, x, oy + side], width=1.1)

            if self.last_move >= 0:
                lx, ly = rls.xy(self.last_move)
                Color(1.0, 0.85, 0.2, 0.35)
                Rectangle(
                    pos=(ox + lx * cell - cell * 0.35, oy + ly * cell - cell * 0.35),
                    size=(cell * 0.7, cell * 0.7),
                )

            rr = cell * 0.42
            for i, v in enumerate(grid):
                if v == EMPTY:
                    continue
                x, y = rls.xy(i)
                cx = ox + x * cell
                cy = oy + y * cell
                if v == BLACK:
                    Color(0.05, 0.05, 0.05, 1)
                else:
                    Color(0.96, 0.96, 0.96, 1)
                Ellipse(pos=(cx - rr, cy - rr), size=(2 * rr, 2 * rr))
                if v == WHITE:
                    Color(0.15, 0.15, 0.15, 0.55)
                    Line(circle=(cx, cy, rr), width=1.0)


class Controller:
    def __init__(self, info_label: Label, board: GomokuBoard):
        self.info_label = info_label
        self.board = board

        self.rules = Rules(15)
        self.engine = Engine(self.rules)

        self.state = self.rules.initial_state()
        self.history: List[State] = [self.state]

        self.mod = "İnsan: Siyah"
        self.difficulty = 7

        self._ai_token = 0
        self._ai_running = False

    def start(self) -> None:
        self._sync_ui()
        self._maybe_start_ai()

    def set_mode(self, mode_text: str) -> None:
        self.mod = mode_text
        self._sync_ui()
        self._maybe_start_ai()

    def set_difficulty(self, val: int) -> None:
        self.difficulty = int(val)
        self._sync_ui()

    def set_board_size(self, size_text: str) -> None:
        n = 15 if "15" in size_text else 13
        if n == self.rules.n:
            return

        self._ai_token += 1
        self._ai_running = False

        self.rules = Rules(n)
        self.engine = Engine(self.rules)

        self.state = self.rules.initial_state()
        self.history = [self.state]
        self.board.last_move = -1

        self._sync_ui()
        self._maybe_start_ai()

    def yeni_oyun(self) -> None:
        self._ai_token += 1
        self._ai_running = False
        self.state = self.rules.initial_state()
        self.history = [self.state]
        self.board.last_move = -1
        self._sync_ui()
        self._maybe_start_ai()

    def geri_al(self) -> None:
        if len(self.history) <= 1:
            return

        self._ai_token += 1
        self._ai_running = False

        steps = 2 if self.mod in ("İnsan: Siyah", "İnsan: Beyaz") else 1
        for _ in range(steps):
            if len(self.history) > 1:
                self.history.pop()

        self.state = self.history[-1]
        self.board.last_move = self.state.last_move
        self._sync_ui()
        self._maybe_start_ai()

    def is_human_turn(self) -> bool:
        if self.mod == "YZ vs YZ":
            return False
        if self.mod == "İnsan: Siyah":
            return self.state.turn == BLACK
        if self.mod == "İnsan: Beyaz":
            return self.state.turn == WHITE
        return False

    def on_human_move(self, mv: int) -> None:
        if self.state.is_over() or not self.is_human_turn():
            return
        if self.state.grid[mv] != EMPTY:
            return

        try:
            self.state = self.rules.apply_move(self.state, mv)
            self.history.append(self.state)
            self.board.last_move = mv
        except Exception:
            return

        self._sync_ui()
        self._maybe_start_ai()

    def _maybe_start_ai(self) -> None:
        if self.state.is_over() or self.is_human_turn() or self._ai_running:
            return

        self._ai_running = True
        token = self._ai_token + 1
        self._ai_token = token

        self.info_label.text = self._status_text(thinking=True)

        def worker():
            try:
                mv = self.engine.best_move(self.state, self.difficulty)
            except Exception:
                mv = None

            def apply_on_ui(_dt):
                if token != self._ai_token:
                    return
                self._ai_running = False

                if mv is None or self.state.is_over():
                    self._sync_ui()
                    self._maybe_start_ai()
                    return
                if self.state.grid[mv] != EMPTY:
                    self._sync_ui()
                    self._maybe_start_ai()
                    return

                try:
                    self.state = self.rules.apply_move(self.state, mv)
                    self.history.append(self.state)
                    self.board.last_move = mv
                except Exception:
                    pass

                self._sync_ui()
                self._maybe_start_ai()

            Clock.schedule_once(apply_on_ui, 0)

        threading.Thread(target=worker, daemon=True).start()

    def _sync_ui(self) -> None:
        self.board._redraw()
        self.info_label.text = self._status_text(thinking=False)

    def _status_text(self, thinking: bool) -> str:
        s = self.state
        b = sum(1 for v in s.grid if v == BLACK)
        w = sum(1 for v in s.grid if v == WHITE)
        turn = "Siyah" if s.turn == BLACK else "Beyaz"

        if s.winner == BLACK:
            return f"BİTTİ: Siyah kazandı | Taş: {b}-{w}"
        if s.winner == WHITE:
            return f"BİTTİ: Beyaz kazandı | Taş: {b}-{w}"
        if s.is_over():
            return f"BİTTİ: Berabere | Taş: {b}-{w}"

        who = "İnsan" if self.is_human_turn() else "YZ"
        think = " (düşünüyor...)" if thinking and not self.is_human_turn() else ""
        return f"Tahta: {self.rules.n}x{self.rules.n} | Sıra: {turn} | Mod: {self.mod} | {who}{think} | Taş: {b}-{w}"


class GomokuApp(App):
    title = "Gomoku - Türkçe + Çok Güçlü YZ"

    def build(self):
        Window.clearcolor = (0.12, 0.12, 0.12, 1)

        root = BoxLayout(orientation="vertical", padding=dp(8), spacing=dp(8))

        info = Label(text="", size_hint_y=None, height=dp(48), halign="left", valign="middle")
        info.bind(size=lambda *_: setattr(info, "text_size", info.size))
        root.add_widget(info)

        board_box = BoxLayout(size_hint_y=1.0)
        root.add_widget(board_box)

        controls = BoxLayout(size_hint_y=None, height=dp(54), spacing=dp(8))
        root.add_widget(controls)

        bottom = BoxLayout(size_hint_y=None, height=dp(54), spacing=dp(8))
        root.add_widget(bottom)

        board = GomokuBoard(controller=None)
        board_box.add_widget(board)

        controller = Controller(info_label=info, board=board)
        board.controller = controller
        controller.start()

        btn_new = Button(text="Yeni Oyun")
        btn_undo = Button(text="Geri Al")
        mode = Spinner(text="İnsan: Siyah", values=("İnsan: Siyah", "İnsan: Beyaz", "YZ vs YZ"))

        size_label = Label(text="Tahta", size_hint_x=None, width=dp(60))
        size_spin = Spinner(text="15x15", values=("13x13", "15x15"))

        diff_label = Label(text="Zorluk", size_hint_x=None, width=dp(70))
        diff = Slider(min=1, max=10, value=controller.difficulty, step=1)

        btn_new.bind(on_release=lambda *_: controller.yeni_oyun())
        btn_undo.bind(on_release=lambda *_: controller.geri_al())
        mode.bind(text=lambda _s, t: controller.set_mode(t))
        size_spin.bind(text=lambda _s, t: controller.set_board_size(t))
        diff.bind(value=lambda _i, v: controller.set_difficulty(int(v)))

        controls.add_widget(btn_new)
        controls.add_widget(btn_undo)
        controls.add_widget(mode)

        bottom.add_widget(size_label)
        bottom.add_widget(size_spin)
        bottom.add_widget(diff_label)
        bottom.add_widget(diff)

        return root


if __name__ == "__main__":
    GomokuApp().run()
