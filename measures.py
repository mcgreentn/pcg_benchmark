"""
Behavior measures for the MAP-Elites archive.

A measure is a *descriptor* of how a run played, not a reward -- the objective
(completion percentage) is the only thing maximized. Measures decide which cell a
run is filed under, and the archive keeps the best-completing run per cell.

Each entry pairs a label with the MarioResult accessor that produces it, plus a
default bin count and range. Ranges must span what real play produces: pyribs
silently CLIPS out-of-range values into the edge cell, so a range that is too
narrow quietly merges distinct behaviors instead of raising.

Coin and enemy ceilings are properties of the LEVEL, not of the agents, so those two
axes support `max: auto` and are resolved from the level file at startup. Calibrating
them from observed play would have been wrong: A* and the heuristic agent beeline for
the exit and score 0 coins on every level, which says nothing about what is reachable.

    lvl-1  ->  11 coins, 15 enemies
    lvl-7  ->  35 coins, 12 enemies
    lvl-15 ->  17 coins, 50 enemies

Time and jumps are bounded by the clock rather than the level, so they stay fixed.
Observed on lvl-1 over a 20s run:

    agent       compl  kills(net)  jumps  time_left
    astar       1.000           3     18       6.9s
    heuristic   0.782           2     22       9.5s
    random      0.099           0      4      18.6s
"""
import os
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Measure:
    key: str
    label: str
    extract: Callable   # MarioResult -> float
    bins: int
    low: float
    high: float

    @property
    def range(self):
        return (float(self.low), float(self.high))


# Kills excluding falls, matching how the benchmark's own MarioProblem counts
# enemies (see probs/smb/problem.py). An enemy that walks off a ledge is not a
# thing the player did, so it should not move the player's behavior coordinate.
def _net_kills(result):
    return max(0, result.getKillsTotal() - result.getKillsByFall())


# Seconds left on the clock. High either because the run finished fast or because
# it died fast -- which is fine here: completion is the objective, so the elite of
# a high-time-left cell is whoever got furthest on the least time. That is a
# speedrunner.
def _seconds_left(result):
    return result.getRemainingTime() / 1000.0


# Characters the level parser turns into an enemy sprite (see MarioLevel in
# engine/core.py). Uppercase variants are the winged forms.
ENEMY_CHARS = "yYEgGkKrR"

# Bullet-bill spawners emit enemies indefinitely, so a level containing them has no
# fixed enemy ceiling and `max: auto` will undercount.
SPAWNER_CHARS = "Bb"


def level_limits(level):
    """
    Ceilings a level imposes on the count-based measures.

    Coins come from MarioLevel.totalCoins, which is the engine's own tally of every
    coin tile plus every coin-bearing block -- more reliable than counting characters,
    since '?' and '@' are power-up blocks that hold no coin while 'C', 'Q', '!', '2'
    and 'o' all do.
    """
    from pcg_benchmark.probs.smb.engine.core import MarioLevel

    tiles = level.replace("\n", "")
    return {
        "coins": MarioLevel(level).totalCoins,
        "kills": sum(tiles.count(c) for c in ENEMY_CHARS),
        "spawners": sum(tiles.count(c) for c in SPAWNER_CHARS),
    }


MEASURE_LIBRARY = {
    # auto_source "level" reads the ceiling from the level file, "config" from a config
    # key. auto_pad is added on top: integer counts get +1 so the maximum attainable
    # value lands in its own bin instead of clipping; continuous axes get +0.
    "coins": dict(
        label="Tile coins collected",
        extract=lambda r: r.getNumCollectedTileCoins(),
        bins=4, low=0.0, high=12.0,
        auto_source="level", auto_key="coins", auto_pad=1,
    ),
    "kills": dict(
        label="Enemies defeated (excl. falls)",
        extract=_net_kills,
        bins=4, low=0.0, high=16.0,
        auto_source="level", auto_key="kills", auto_pad=1,
    ),
    "time": dict(
        label="Seconds left on the clock",
        extract=_seconds_left,
        bins=4, low=0.0, high=20.0,
        auto_source="config", auto_key="game_time", auto_pad=0,
    ),
    "jumps": dict(
        label="Jumps taken",
        extract=lambda r: r.getNumJumps(),
        bins=4, low=0.0, high=24.0,
    ),
}

DEFAULT_MEASURES = ["coins", "kills", "time", "jumps"]

# Must match runner.runLevelWithNet's gameTime default; the config's game_time key
# overrides both.
DEFAULT_GAME_TIME = 20


def load_measures(config=None, level=None):
    """
    Resolve the active measures from a config dict, in the order written.

        measures:
          coins: {bins: 4, max: auto}      # ceiling read from the level
          time:  {bins: 4, min: 0, max: 20}

    Any of bins/min/max may be omitted to take the library default. `max: auto` reads
    the ceiling from the level named by config["level_path"], loaded lazily so callers
    that use fixed ranges never touch the filesystem. Falls back to the legacy
    coins_dim/kills_dim keys when no `measures` block is present.
    """
    config = config or {}
    spec = config.get("measures")

    if not spec:
        return [
            _build("coins", {"bins": config.get("coins_dim", 2), "max": config.get("coins_dim", 2)}, config, level),
            _build("kills", {"bins": config.get("kills_dim", 2), "max": config.get("kills_dim", 2)}, config, level),
        ]

    return [_build(key, spec[key] or {}, config, level) for key in spec]


_LIMIT_CACHE = {}


def _resolve_auto(key, config, level):
    """Ceiling for `max: auto`, read from the level or from another config key."""
    base = MEASURE_LIBRARY[key]
    source = base.get("auto_source")
    if source is None:
        raise ValueError(
            f"measure '{key}' has no automatic ceiling -- nothing in the level or the "
            f"config bounds it. Give it an explicit max."
        )

    pad = base.get("auto_pad", 0)

    if source == "config":
        ceiling = config.get(base["auto_key"], DEFAULT_GAME_TIME)
        return float(ceiling + pad)

    if level is None:
        level_path = config.get("level_path", "./data/smb/original/lvl-1.txt")
        if level_path not in _LIMIT_CACHE:
            if not os.path.exists(level_path):
                raise FileNotFoundError(
                    f"measure '{key}' uses `max: auto` but the level '{level_path}' "
                    f"does not exist. Fix level_path or set an explicit max."
                )
            with open(level_path, "r") as f:
                _LIMIT_CACHE[level_path] = level_limits(f.read())
        limits = _LIMIT_CACHE[level_path]
    else:
        limits = level_limits(level)

    return float(limits[base["auto_key"]] + pad)


def _build(key, overrides, config=None, level=None):
    if key not in MEASURE_LIBRARY:
        known = ", ".join(sorted(MEASURE_LIBRARY))
        raise KeyError(f"Unknown measure '{key}'. Available measures: {known}")

    base = MEASURE_LIBRARY[key]
    bins = int(overrides.get("bins", base["bins"]))
    low = float(overrides.get("min", base["low"]))

    raw_high = overrides.get("max", base["high"])
    if isinstance(raw_high, str) and raw_high.lower() == "auto":
        high = _resolve_auto(key, config or {}, level)
    else:
        high = float(raw_high)

    if bins < 1:
        raise ValueError(f"measure '{key}': bins must be >= 1, got {bins}")
    if high <= low:
        raise ValueError(f"measure '{key}': max ({high}) must exceed min ({low})")

    return Measure(key=key, label=base["label"], extract=base["extract"],
                   bins=bins, low=low, high=high)


def describe(measures):
    """One-line summary naming each axis with its resolved range."""
    shape = "x".join(str(m.bins) for m in measures)
    cells = 1
    for m in measures:
        cells *= m.bins
    axes = ", ".join(f"{m.key}[{m.low:g}-{m.high:g}]" for m in measures)
    return f"{shape} = {cells} cells over {axes}"
