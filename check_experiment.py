"""
Sanity checks for the Mario neuroevolution experiment.

There is no test suite in this repo, and the failures that matter here are silent
ones -- an evaluation that scores a random network instead of the evolved one still
produces a perfectly plausible archive. These checks pin the invariants that make
the experiment mean anything.

    ./venv/Scripts/python.exe check_experiment.py           # fast checks (~30s)
    ./venv/Scripts/python.exe check_experiment.py --full    # also runs the driver end to end

Exits non-zero if anything fails.
"""
import hashlib
import os
import shutil
import sys
import tempfile

import numpy as np
import yaml

LEVEL_PATH = "./data/smb/original/lvl-1.txt"
WEIGHTS_PATH = "./data/smb/weights.json"

_results = []


def check(name):
    """Register a check. The wrapped function returns a detail string, or raises."""
    def decorator(fn):
        _results.append((name, fn))
        return fn
    return decorator


def _level():
    with open(LEVEL_PATH, "r") as f:
        return f.read()


def _fresh_agent(weights=None, seed=None, weights_path=None):
    """Build an agent whose brain is initialized against a real forward model."""
    from pcg_benchmark.probs.smb.engine.agents import nn
    from pcg_benchmark.probs.smb.engine.core import MarioGame

    agent = nn.Agent(seed=seed, weights=weights, weights_path=weights_path)
    game = MarioGame()
    game.setAgent(agent)
    game.setup(_level(), 20, 0)
    return agent, game


# --------------------------------------------------------------------------
# The regression that motivated all of this: runGame() re-runs setup(), which
# calls agent.initialize(). If that rebuilds the network, the evolved genome is
# silently replaced by a random one and every fitness score becomes noise.
# --------------------------------------------------------------------------

@check("genome survives runGame")
def _genome_round_trip():
    agent, game = _fresh_agent()
    size = agent.brain.get_param_size()

    weights = np.linspace(-1.0, 1.0, size)
    agent, game = _fresh_agent(weights=weights)
    game.runGame(agent, _level(), 3, 0)

    after = agent.brain.get_param()
    if not np.allclose(after, weights, atol=1e-6):
        raise AssertionError(
            f"genome was modified by runGame: first values {weights[:3]} -> {after[:3]}"
        )
    return f"{size} parameters preserved exactly"


@check("weights assigned to agent.brain survive runGame")
def _legacy_assignment_round_trip():
    """
    The constructor route (nn.Agent(weights=...)) is re-applied on every initialize(),
    so it survives even a network rebuild. Weights poked straight onto agent.brain --
    the pattern the original runner used -- only survive if initialize() is idempotent.
    This is the check that actually pins that guard.
    """
    agent, game = _fresh_agent()
    weights = np.full(agent.brain.get_param_size(), 0.5)
    agent.brain.load_weights(weights)

    game.runGame(agent, _level(), 3, 0)

    after = agent.brain.get_param()
    if not np.allclose(after, weights, atol=1e-6):
        raise AssertionError(
            "runGame() rebuilt the network and discarded weights assigned to agent.brain "
            f"({weights[:3]} -> {after[:3]}); initialize() must be idempotent"
        )
    return "initialize() is idempotent; assigned weights kept"


@check("evaluation is deterministic")
def _deterministic():
    from runner import runLevelWithNet

    weights = np.full(2869, 0.5)
    a = runLevelWithNet(_level(), weights, gameTime=5).getCompletionPercentage()
    b = runLevelWithNet(_level(), weights, gameTime=5).getCompletionPercentage()
    if a != b:
        raise AssertionError(f"same genome gave different results: {a} vs {b}")
    return f"same genome twice -> {a:.4f}"


@check("different genomes behave differently")
def _genome_sensitivity():
    from runner import runLevelWithNet

    rng = np.random.default_rng(0)
    size = 2869
    scores = [
        runLevelWithNet(_level(), rng.normal(0, 3, size), gameTime=8).getCompletionPercentage()
        for _ in range(6)
    ]
    if len(set(scores)) == 1:
        raise AssertionError(
            f"all {len(scores)} random genomes scored identically ({scores[0]}) -- "
            "weights are probably being ignored"
        )
    return f"{len(set(scores))} distinct outcomes across {len(scores)} genomes"


# --------------------------------------------------------------------------
# smb/ and smbtile/ carry duplicate copies of the engine. smb's core.py was
# refactored to extract setup(); this pins that it stayed behavior-preserving.
# --------------------------------------------------------------------------

@check("smb and smbtile engines agree")
def _engine_parity():
    from pcg_benchmark.probs.smb.engine import runLevel as smb_run
    from pcg_benchmark.probs.smbtile.engine import runLevel as tile_run

    level = _level()
    for agent_name in ["astar", "heuristic", "random", "donothing"]:
        a = smb_run(level, agent_name, 20, 50, seed=0).getCompletionPercentage()
        b = tile_run(level, agent_name, 20, 50, seed=0).getCompletionPercentage()
        if abs(a - b) > 1e-12:
            raise AssertionError(f"{agent_name}: smb={a} smbtile={b}")

    astar = smb_run(level, "astar", 20, 50, seed=0).getCompletionPercentage()
    if astar < 1.0:
        raise AssertionError(f"astar no longer completes lvl-1 (got {astar})")
    return "4 agents identical; astar completes lvl-1"


# --------------------------------------------------------------------------
# The committed checkpoint has to match the current SimpleNN architecture --
# it silently didn't, because a stray auto-save kept rewriting it.
# --------------------------------------------------------------------------

@check("committed weights.json matches the network")
def _checkpoint_loads():
    agent, _ = _fresh_agent(weights_path=WEIGHTS_PATH)
    size = agent.brain.get_param_size()
    return f"loaded {os.path.basename(WEIGHTS_PATH)} into {size}-parameter network"


@check("bad weights fail loudly")
def _clear_errors():
    agent, _ = _fresh_agent()
    try:
        agent.brain.load_weights(np.ones(5))
    except ValueError as e:
        if "2869" not in str(e):
            raise AssertionError(f"error message should name the expected size: {e}")
        return "wrong-length vector raises a readable ValueError"
    raise AssertionError("a 5-element vector was accepted into a 2869-parameter network")


# --------------------------------------------------------------------------
# Upstream benchmark still works (the base Problem gained a parameters() hook).
# --------------------------------------------------------------------------

@check("benchmark problems still evaluate")
def _benchmark_regression():
    import pcg_benchmark

    for name in ["smb-scene-v0", "smbtile-scene-v0"]:
        env = pcg_benchmark.make(name)
        env.seed(0)
        env._problem.parameters(diversity=0.4)   # used to raise AttributeError
        contents = [env.content_space.sample() for _ in range(3)]
        env.evaluate(contents, [env.control_space.sample()])
    return "smb + smbtile evaluate(); parameters() hook present"


# --------------------------------------------------------------------------
# Runs must not write into the tracked data/ tree.
# --------------------------------------------------------------------------

def _hash_tree(root):
    digest = {}
    for dirpath, _, filenames in os.walk(root):
        for filename in sorted(filenames):
            path = os.path.join(dirpath, filename)
            with open(path, "rb") as f:
                digest[path] = hashlib.md5(f.read()).hexdigest()
    return digest


@check("measures extract from a real run")
def _measures_extract():
    from measures import load_measures, describe
    from config import load_config
    from runner import runLevelWithNet

    measures = load_measures(load_config())
    result = runLevelWithNet(_level(), np.full(2869, 0.5), gameTime=6)

    values = []
    for m in measures:
        v = m.extract(result)
        if not isinstance(v, (int, float, np.integer, np.floating)):
            raise AssertionError(f"measure '{m.key}' returned {type(v).__name__}, not a number")
        values.append(f"{m.key}={v:g}")
    return f"{describe(measures)}; {', '.join(values)}"


@check("kills measure excludes falls")
def _kills_excludes_falls():
    from measures import MEASURE_LIBRARY

    extract = MEASURE_LIBRARY["kills"]["extract"]

    class FakeResult:
        def __init__(self, total, fall):
            self._t, self._f = total, fall
        def getKillsTotal(self):
            return self._t
        def getKillsByFall(self):
            return self._f

    if extract(FakeResult(5, 2)) != 3:
        raise AssertionError("fall kills are still being counted")
    if extract(FakeResult(2, 5)) != 0:
        raise AssertionError("more falls than kills should clamp to 0, not go negative")
    return "5 kills - 2 falls = 3; clamps at 0"


@check("measure ranges cover what the level allows")
def _ranges_cover_level():
    """
    Out-of-range values are clipped by pyribs, silently merging behaviors. Coin and
    enemy ceilings come from the level, not from agent play -- A* and the heuristic
    beeline for the exit and collect zero coins on every level, so calibrating these
    axes against observed play would leave them far too narrow.
    """
    from measures import load_measures, level_limits
    from config import load_config

    config = load_config()
    measures = load_measures(config)
    limits = level_limits(_level())
    by_key = {m.key: m for m in measures}

    problems = []
    for key in ("coins", "kills"):
        if key not in by_key:
            continue
        m, ceiling = by_key[key], limits[key]
        if ceiling >= m.high:
            problems.append(f"{key}: level allows {ceiling} but range tops out at {m.high:g}")
    if problems:
        raise AssertionError("; ".join(problems))

    detail = ", ".join(f"{k}<={limits[k]}" for k in ("coins", "kills") if k in by_key)
    if limits["spawners"]:
        detail += f" (warning: {limits['spawners']} enemy spawners, kills unbounded)"
    return f"level ceilings inside every axis: {detail}"


@check("measure ranges cover competent play")
def _ranges_cover_play():
    """Time and jumps are clock-bounded, so these are calibrated against real agents."""
    from measures import load_measures
    from config import load_config
    from pcg_benchmark.probs.smb.engine import runLevel

    measures = {m.key: m for m in load_measures(load_config())}
    clipped = []
    for agent in ("astar", "heuristic"):
        result = runLevel(_level(), agent, 20, 50, seed=0)
        for key in ("time", "jumps"):
            if key not in measures:
                continue
            m = measures[key]
            v = float(m.extract(result))
            if v < m.low or v >= m.high:
                clipped.append(f"{agent} {key}={v:g} outside [{m.low:g}, {m.high:g})")
    if clipped:
        raise AssertionError("clipped into edge cells: " + "; ".join(clipped))
    return "astar and heuristic land inside the time and jump axes"


@check("checkpoint columns order numerically")
def _checkpoint_column_order():
    """
    solution_0..solution_2868 sorted lexicographically puts solution_1000 before
    solution_2, permuting every genome on resume -- correct objectives attached to
    scrambled weights.
    """
    import pandas as pd
    from driver import MarioEvolutionDriver

    columns = [f"solution_{i}" for i in range(2869)] + ["objective"]
    df = pd.DataFrame(columns=columns)
    ordered = MarioEvolutionDriver._indexed_columns(df, "solution_")

    expected = [f"solution_{i}" for i in range(2869)]
    if ordered != expected:
        first = next(i for i, (a, b) in enumerate(zip(ordered, expected)) if a != b)
        raise AssertionError(f"column order diverges at index {first}: {ordered[first]} != {expected[first]}")
    return "2869 solution columns recovered in genome order"


@check("config knobs reach the search")
def _config_knobs_wired():
    """batch_size, sigma0, ranker and n_emitters are config keys, not constants."""
    import copy, yaml, tempfile
    from driver import MarioEvolutionDriver

    base = yaml.safe_load(open("exp_config.yaml"))
    tmp = tempfile.mkdtemp(prefix="pcg_knobs_")
    cfg = copy.deepcopy(base)
    cfg.update(n_emitters=2, batch_size=7, sigma0=0.25, ranker="2rd",
               output_dir=os.path.join(tmp, "out"))
    path = os.path.join(tmp, "exp_config.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f)

    try:
        driver = MarioEvolutionDriver(config_path=path)
        emitter = driver.emitters[0]
        problems = []
        if len(driver.emitters) != 2:
            problems.append(f"n_emitters ignored ({len(driver.emitters)})")
        if emitter.batch_size != 7:
            problems.append(f"batch_size ignored ({emitter.batch_size})")
        if abs(emitter._sigma0 - 0.25) > 1e-9:
            problems.append(f"sigma0 ignored ({emitter._sigma0})")
        if "RandomDirection" not in type(emitter._ranker).__name__:
            problems.append(f"ranker ignored ({type(emitter._ranker).__name__})")
        asked = driver.scheduler.ask().shape[0]
        if asked != 14:
            problems.append(f"ask() returned {asked}, expected 2 x 7 = 14")
        if problems:
            raise AssertionError("; ".join(problems))
        return "n_emitters, batch_size, sigma0, ranker all take effect"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@check("game_time drives the time axis")
def _game_time_drives_axis():
    """
    The time measure is seconds left on the clock, so its ceiling IS the game length.
    Hardcoding it would silently mis-scale the axis whenever the clock changed.
    """
    import copy
    from measures import load_measures
    from config import load_config

    base = load_config()
    for game_time in (20, 45, 90):
        cfg = copy.deepcopy(base)
        cfg["game_time"] = game_time
        axes = {m.key: m for m in load_measures(cfg)}
        if "time" not in axes:
            return "time axis not in use"
        if abs(axes["time"].high - game_time) > 1e-9:
            raise AssertionError(
                f"game_time={game_time} but time axis tops out at {axes['time'].high:g}"
            )
    return "time axis follows game_time (checked at 20, 45, 90)"


@check("seeded runs are reproducible")
def _seeded_reproducible():
    """Two drivers with the same seed must propose byte-identical first batches."""
    import copy, yaml, tempfile
    from driver import MarioEvolutionDriver

    base = yaml.safe_load(open("exp_config.yaml"))
    tmp = tempfile.mkdtemp(prefix="pcg_seed_")

    def build(seed, tag):
        cfg = copy.deepcopy(base)
        cfg["seed"] = seed
        cfg["output_dir"] = os.path.join(tmp, "out")
        path = os.path.join(tmp, f"cfg_{tag}.yaml")
        with open(path, "w") as f:
            yaml.safe_dump(cfg, f)
        return MarioEvolutionDriver(config_path=path)

    try:
        a, b = build(0, "a").scheduler.ask(), build(0, "b").scheduler.ask()
        if not np.allclose(a, b):
            raise AssertionError("same seed produced different genomes")

        u, v = build(None, "u").scheduler.ask(), build(None, "v").scheduler.ask()
        if np.allclose(u, v):
            raise AssertionError("seed: null produced identical genomes -- not actually random")
        return "seed=0 reproduces exactly; seed=null varies"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@check("output dir resolution honors overrides")
def _output_resolution():
    from config import get_output_dir

    previous = os.environ.pop("PCG_OUTPUT_DIR", None)
    try:
        from_config = get_output_dir()
        os.environ["PCG_OUTPUT_DIR"] = os.path.join("output", "override-probe")
        from_env = get_output_dir()
    finally:
        os.environ.pop("PCG_OUTPUT_DIR", None)
        if previous is not None:
            os.environ["PCG_OUTPUT_DIR"] = previous

    if "data" in from_config.replace("\\", "/").split("/"):
        raise AssertionError(f"output dir points into tracked data/: {from_config}")
    if "override-probe" not in from_env:
        raise AssertionError(f"PCG_OUTPUT_DIR was ignored (got {from_env})")
    return f"config -> {from_config}; env override respected"


def _full_driver_check():
    """Run the driver end to end into a temp dir and assert data/ is untouched."""
    from driver import MarioEvolutionDriver

    tmp_root = tempfile.mkdtemp(prefix="pcg_check_")
    config_path = os.path.join(tmp_root, "exp_config.yaml")
    out_dir = os.path.join(tmp_root, "out")

    with open("exp_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    config.update({"n_iterations": 2, "checkpoint_interval": 1, "n_emitters": 1,
                   "workers": 1, "output_dir": out_dir})
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f)

    before = _hash_tree("data")
    try:
        MarioEvolutionDriver(config_path=config_path).run()
        after = _hash_tree("data")
        if before != after:
            changed = sorted(set(before) ^ set(after)) or [
                p for p in before if before[p] != after.get(p)
            ]
            raise AssertionError(f"the run modified tracked data/: {changed}")

        final = os.path.join(out_dir, "archive_final.csv")
        if not os.path.exists(final):
            raise AssertionError(f"driver did not write {final}")

        from analyzer import analyze
        import matplotlib
        matplotlib.use("Agg")
        png = os.path.join(out_dir, "heatmap.png")
        analyze(final, png, config_path, show=False)
        if not os.path.exists(png):
            raise AssertionError(f"analyzer did not write {png}")

        return f"driver + analyzer wrote to {out_dir}; data/ unchanged"
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def main():
    full = "--full" in sys.argv
    if full:
        _results.append(("driver end to end, data/ untouched", _full_driver_check))

    print(f"Running {len(_results)} checks...\n")
    failures = []
    for name, fn in _results:
        try:
            detail = fn()
            print(f"  PASS  {name}\n          {detail}")
        except Exception as e:
            failures.append(name)
            print(f"  FAIL  {name}\n          {type(e).__name__}: {e}")

    print()
    if failures:
        print(f"{len(failures)} of {len(_results)} checks FAILED: {', '.join(failures)}")
        return 1
    print(f"All {len(_results)} checks passed.")
    if not full:
        print("Re-run with --full to also exercise the driver and analyzer.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
