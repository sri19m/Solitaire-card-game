"""Offline simulator of a TriPeaks-style game, used to compare and tune strategies for free.

The real game's layout differs from this classic 3-peak layout, so absolute win rates
are only indicative. Relative differences between strategies are what matter.
"""
import random
from collections import Counter

from .brain import DEFAULT_W, Free, adjacent, plan, rank_probs


def tripeaks_cover():
    """covered_by[i] = indices of cards that cover card i (28 cards: 3+6+9+10)."""
    cov = [[] for _ in range(28)]
    r1, r2, r3 = 3, 9, 18
    for p in range(3):
        for k in (0, 1):
            cov[p].append(r1 + 2 * p + k)
    for j in range(6):
        p, k = divmod(j, 2)
        cov[r1 + j] += [r2 + 3 * p + k, r2 + 3 * p + k + 1]
    for m in range(9):
        cov[r2 + m] += [r3 + m, r3 + m + 1]
    return cov


COVER = tripeaks_cover()


class Game:
    def __init__(self, rng: random.Random, stock_size: int = 24):
        deck = [r for r in range(1, 14) for _ in range(4)]
        rng.shuffle(deck)
        self.rank = deck[:28]
        self.stock = deck[28:28 + stock_size]
        self.active = self.stock.pop()
        self.removed: set[int] = set()
        self.drawn = [self.active]

    def free_idx(self):
        return [i for i in range(28) if i not in self.removed
                and all(c in self.removed for c in COVER[i])]

    def unlock_value(self, i):
        v = 0.0
        for j in range(28):
            if j in self.removed or i not in COVER[j]:
                continue
            left = [c for c in COVER[j] if c not in self.removed]
            v += 1.0 / len(left)
        return v

    def seen(self) -> Counter:
        c = Counter(self.drawn)
        known = set(self.removed) | set(self.free_idx())
        c.update(self.rank[i] for i in known)
        return c

    def play(self, i):
        self.removed.add(i)
        self.active = self.rank[i]

    def draw(self) -> bool:
        if not self.stock:
            return False
        self.active = self.stock.pop()
        self.drawn.append(self.active)
        return True

    @property
    def won(self):
        return len(self.removed) == 28


def policy_first(g: Game, w):
    for i in g.free_idx():
        if adjacent(g.rank[i], g.active):
            return [i]
    return []


def policy_longest(g: Game, w):
    zero = {"len": 1.0, "unlock": 0.0, "cont": 0.0, "hid": 0.0}
    free = [Free(i, g.rank[i]) for i in g.free_idx()]
    return [f.key for f in plan(g.active, free, {}, zero)]


def policy_brain(g: Game, w):
    free = [Free(i, g.rank[i], g.unlock_value(i)) for i in g.free_idx()]
    p = rank_probs(g.seen())
    return [f.key for f in plan(g.active, free, p, w)]


POLICIES = {"first-match": policy_first, "longest-chain": policy_longest, "brain": policy_brain}


def play_game(seed: int, policy, w=None) -> tuple[bool, int]:
    g = Game(random.Random(seed))
    w = w or DEFAULT_W
    while not g.won:
        chain = policy(g, w)
        if chain:
            for i in chain:
                g.play(i)
        elif not g.draw():
            break
    return g.won, len(g.removed)


def win_rate(policy, games: int, w=None, seed0: int = 0) -> tuple[float, float]:
    wins = cleared = 0
    for s in range(games):
        won, n = play_game(seed0 + s, policy, w)
        wins += won
        cleared += n
    return wins / games, cleared / games
