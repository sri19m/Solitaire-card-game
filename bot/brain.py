"""Decision engine: card counting + scored lookahead over the visible cards.

Ranks: 1=A ... 13=K. Ace and King are adjacent (wrap-around).
A "free" card is a face-up card that can be tapped right now.
"""
from collections import Counter
from dataclasses import dataclass

# Tuned with tools/simulate.py (see README). Re-tune after layout changes.
DEFAULT_W = {"len": 100.0, "unlock": 40.0, "cont": 12.0, "hid": 30.0}


@dataclass(frozen=True)
class Free:
    key: object          # identity (board index in the simulator, position live)
    rank: int
    unlock: float = 0.0  # how much face-down material removing this card should free up
    x: int = 0           # tap position (live only)
    y: int = 0


def adjacent(a: int, b: int, wrap: bool = True) -> bool:
    d = abs(a - b)
    return d == 1 or (wrap and d == 12)


class LevelMemory:
    """Card counting for one level. Every card is seen once when it first becomes face-up."""

    def __init__(self, copies: int = 4):
        self.copies = copies
        self.board: list[tuple[int, int, int]] = []  # (x, y, rank) of every face-up card ever seen
        self.drawn: list[int] = []                   # ranks of cards that appeared from the draw pile

    def see_board(self, x: int, y: int, rank: int) -> None:
        for bx, by, _ in self.board:
            if abs(bx - x) < 25 and abs(by - y) < 25:
                return
        self.board.append((x, y, rank))

    def see_drawn(self, rank: int) -> None:
        self.drawn.append(rank)

    def seen(self) -> Counter:
        c = Counter(r for _, _, r in self.board)
        c.update(self.drawn)
        return c

    def probs(self) -> dict[int, float]:
        return rank_probs(self.seen(), self.copies)


def rank_probs(seen: Counter, copies: int = 4) -> dict[int, float]:
    """Probability that a still-unseen card has each rank."""
    copies = max(copies, max(seen.values(), default=0))  # tolerate multi-deck games
    left = {r: max(copies - seen.get(r, 0), 0) for r in range(1, 14)}
    total = sum(left.values()) or 1
    return {r: n / total for r, n in left.items()}


def next_prob(p: dict[int, float], rank: int, wrap: bool = True) -> float:
    return sum(pr for r, pr in p.items() if adjacent(r, rank, wrap))


def first_failed_tap(chain, board, tol: int = 12):
    """The first tapped card that is still on the board afterwards (same place, same rank), or None.

    A successful tap removes the card, so a card that is still there means the game rejected
    the tap: the rank was misread, or the card is covered by another card.
    """
    for c in chain:
        if any(abs(b.x - c.x) < tol and abs(b.y - c.y) < tol and b.rank == c.rank for b in board):
            return c
    return None


def plan(active: int, free: list[Free], p: dict[int, float], w: dict | None = None,
         wrap: bool = True, node_cap: int = 40000) -> list[Free]:
    """Pick the best sequence of taps from the current board. [] means: draw a card.

    Every ordered chain of legal taps is scored. A non-empty chain is always
    preferred over drawing, because drawing burns a card from the limited pile.
    """
    w = w or DEFAULT_W
    best_score = float("-inf")
    best: list[int] = []
    nodes = 0
    remaining = set(range(len(free)))
    chain: list[int] = []

    def dfs(cur: int, unlock_sum: float) -> None:
        nonlocal best_score, best, nodes
        nodes += 1
        if chain:
            cont = sum(1 for i in remaining if adjacent(free[i].rank, cur, wrap))
            s = (w["len"] * len(chain) + w["unlock"] * unlock_sum + w["cont"] * cont
                 + w["hid"] * next_prob(p, cur, wrap) * (1 + unlock_sum))
            if s > best_score:
                best_score, best = s, list(chain)
        if nodes >= node_cap:
            return
        for i in sorted(remaining):
            if adjacent(free[i].rank, cur, wrap):
                chain.append(i)
                remaining.discard(i)
                dfs(free[i].rank, unlock_sum + free[i].unlock)
                remaining.add(i)
                chain.pop()

    dfs(active, 0.0)
    return [free[i] for i in best]
