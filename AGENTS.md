# AGENTS.md

Orientation for coding agents working in this repo. Read this before exploring — it will save you a lot of file reads.

## What this repo is

Two things live here, layered:

1. **`pcg_benchmark/`** — an upstream fork of [amidos2006/pcg_benchmark](https://github.com/amidos2006/pcg_benchmark), a Gym-style framework for evaluating procedural content generators across 15 game problems. Mostly untouched library code.
2. **Repo-root scripts** (`driver.py`, `runner.py`, `analyzer.py`, `listener.py`, `exp_config.yaml`) — the *actual current work*: a quality-diversity (MAP-Elites via `pyribs`) experiment that evolves a small CNN to **play** Super Mario levels. This is the `smb-neuro` branch and is **not** upstream.

The distinction matters: the benchmark generates *levels*; the branch work evolves *agents that play a fixed level*. Requests about "the experiment" almost always mean the root scripts + `pcg_benchmark/probs/smb/engine/agents/`.

**Fork remotes:** `origin` = this fork, `upstream` = amidos2006. Main branch `main`; active work on `smb-neuro`.

## Environment

- Windows 11, Python 3.12, venv committed in-tree at `./venv/`.
- **Always run Python as `./venv/Scripts/python.exe`.** The system `python` on PATH has no numpy/torch and will fail immediately.
- Package installed editable (`pip install -e .`); `pcg_benchmark.egg-info/` is the artifact.
- Deps (`setup.py`): numpy, pillow, torch, ribs, dask, distributed, PyYAML, tqdm, matplotlib, shapely.
  The venv can lag `setup.py` — `tqdm`, `matplotlib`, and `shapely` were all declared but
  missing at one point. On `ModuleNotFoundError`, install into the venv rather than
  assuming the code is wrong.
- Shell is PowerShell; a Bash tool is also available. Note `driver.shutdown_listeners()` shells out to `pkill`, which does not exist here (the call site is commented out).

## Layout

```
driver.py            MarioEvolutionDriver — the QD loop (pyribs + Dask). Entry point.
runner.py            runLevelWithNet(level, net) — evaluates one genome. Runs on Dask workers.
analyzer.py          Plots an archive CSV as a heatmap. --archive/--output/--config/--no-show.
listener.py          Legacy file-watcher evaluation path. Superseded by Dask; not wired in.
config.py            Config loading + run-output path resolution, shared by the three above.
exp_config.yaml      Experiment hyperparameters and output_dir.
test.py              Upstream smoke test: iterates every registered problem.
data/smb/original/   15 original SMB levels as text (lvl-1.txt .. lvl-15.txt).
data/smb/weights.json  Initial torch checkpoint for the current network (~15 KB, committed).
output/smb/          Run products — archive CSVs, heatmaps, scratch. Gitignored.
.vscode/launch.json  Debug config, targets ./driver.py.

pcg_benchmark/
  __init__.py        make(name) / list() / register(name, cls, args)
  pcg_env.py         PCGEnv — the user-facing wrapper (info/quality/diversity/controlability/evaluate/render)
  spaces/            IntegerSpace, FloatSpace, ArraySpace, DictionarySpace, GenericSpace
  probs/
    problem.py       Problem ABC: info / quality / diversity / controlability / render
    utils.py         Shared level metrics (dijkstra, flood fill, get_range_reward, histograms, ...)
    __init__.py      Auto-discovers every subfolder and merges its PROBLEMS dict
    <15 problems>/   Each: __init__.py (PROBLEMS registry), problem.py, README.md, images/
    smb/             Slice-based Mario. Has engine/ (full game sim) + agents/
    smbtile/         Tile-based Mario. engine/ is a near-copy of smb's — see gotchas.
```

## The benchmark framework (upstream, stable)

Contract for every problem — subclass `pcg_benchmark.probs.Problem`, set `_content_space` and `_control_space` in `__init__(**kwargs)`, implement:

| Method | Returns |
| --- | --- |
| `info(content)` | dict of everything the metrics need (this is where simulation happens) |
| `quality(info)` | float in [0,1]; 1 = passes |
| `diversity(info1, info2)` | float in [0,1]; 1 = sufficiently different |
| `controlability(info, control)` | float in [0,1]; 1 = matches the control params |
| `render(content)` | `PIL.Image` |

Register by adding a `PROBLEMS = {"name-v0": (Class, {kwargs})}` dict to a new folder's `__init__.py` under `probs/` — discovery is automatic via directory scan (`probs/__init__.py`). Naming convention: `{problem}-v0` for defaults, `{problem}-{variant}-v0` for variants.

`PCGEnv` methods accept a single content *or* a list, and also accept pre-computed `info` dicts in place of content (to avoid re-simulating). `evaluate()` does all three metrics in one pass. Use `env.seed(n)` for reproducibility.

`probs/utils.py` is the shared toolbox — check it before writing any new level-analysis helper. `get_range_reward(value, min, plat_low, plat_high, max)` is the workhorse used by nearly every `quality`/`controlability`.

Per-problem READMEs exist under `pcg_benchmark/probs/<name>/README.md` and are genuinely useful.

## The Mario engine

`probs/smb/engine/` is a Python port of the Mario AI Framework. Key classes in `engine/core.py` (~1200 lines):

- **`MarioGame`** — `setAgent(agent)`, `runGame(agent, level, timer, marioState) -> MarioResult`.
- **`MarioAgent`** — the agent ABC: `initialize(model)`, `getActions(model)`, `getAgentName()`. `getActions` returns a length-5 boolean/int array indexed by `MarioActions` = `[LEFT, RIGHT, DOWN, SPEED, JUMP]`.
- **`MarioForwardModel`** — what agents see. `getScreenCompleteObservation()` → 16×16 int grid; also `getMarioCompleteObservation()`, velocities, kill counts, `advance(actions)` for lookahead planners.
- **`MarioResult`** — the fitness/measure source: `getCompletionPercentage()`, `getKillsTotal()`, `getNumCollectedTileCoins()`, `getNumJumps()`, `getCurrentCoins()`, `getAgentEvents()`, etc.

Built-in agents in `engine/agents/`: `astar` (used by the benchmark's playability check), `heuristic`, `greedy`, `random`, `donothing`, and — new on this branch — `nn`.

`engine/__init__.py` exposes `runLevel(levelString, agentName, gameTime, iterations, stickyActions, marioState, seed)`, which imports the agent module by name. `MarioProblem.info()` uses it: tries `heuristic` first, falls back to `astar` if the level isn't completed.

## The neuroevolution experiment (branch work)

**Flow:** `driver.py` → pyribs `Scheduler.ask()` → Dask `client.map(runLevelWithNet, solutions)` → `runner.runLevelWithNet` builds a `MarioGame` + `nn.Agent`, loads the genome into the CNN, plays → `MarioResult` → `scheduler.tell(objectives, measures)`.

- **Genome** = flat float vector of all CNN params, `2869` values for the current architecture.
- **Objective** = `getCompletionPercentage()` (0–1).
- **Measures** = four behavior axes defined in `measures.py` and selected/sized in `exp_config.yaml`: `coins`, `kills` (falls excluded), `time` (seconds left — speedrunning), `jumps` (acrobatics). Default grid is 4×4×4×4 = 256 cells.
- **Network** — `probs/smb/engine/agents/networks/simplenn.py`: 3× (Conv2d k=2 → MaxPool2d k=2) on a 1×16×16 observation, then `fc1(→32)` → `fc2(→5)` → sigmoid → rounded to a 5-bit action. No backprop; evolution only.
- **Genome ↔ network** — `get_param()` / `set_param()` / `get_param_size()` walk `self._modules` and flatten `weight` then `bias` per module. `load_weights()` dispatches on type: `str` → torch checkpoint, `np.ndarray` → `set_param`. `get_weights()` returns a *torch tensor* over `self.parameters()`; `get_param()` returns *numpy*. Use **`get_param()`** on the genome path — it is the exact inverse of the `set_param()` that `load_weights(ndarray)` calls, so the emitter seed → evolve → load round-trip stays numpy end to end.

**`exp_config.yaml` knobs:** search budget (`n_emitters`, `batch_size`, `n_iterations`, `checkpoint_interval`, `workers`), search behavior (`sigma0`, `ranker`), evaluation (`game_time`, `seed`), the `measures` block, and paths (`level_path`, `output_dir`). Evaluations per iteration = `n_emitters × batch_size`; the driver prints the resolved budget and seed at startup.

**Reproducibility:** `seed` feeds the initial network (and therefore `x0`), every emitter (offset per emitter so they don't search identically), and the archive. With a seed, two runs of the same config produce byte-identical archives including all 2869 weights — verified by a check. Set `seed: null` for a fresh search each run. Individual *evaluations* are deterministic either way.

**Checkpointing:** archive dumped to `<output_dir>/archive_{iteration}.csv` every `checkpoint_interval`, plus `archive_final.csv`. Resume with `./venv/Scripts/python.exe driver.py --resume-from output/smb/archive_500.csv` — the iteration number is parsed back out of the filename.

## Behavior measures

All four axes live in `measures.py` as a library of `(label, extractor, default bins,
default range)`; `exp_config.yaml` picks which are active, in what order, and how they
are binned. Adding a fifth means one entry in `MEASURE_LIBRARY` plus one config line —
the driver, the analyzer, and the checks all read the same spec.

Two things to get right when changing them:

- **Ranges must cover what is attainable.** pyribs *clips* out-of-range measures into
  the edge cell rather than raising, so a too-narrow range silently merges distinct
  behaviors.
- **Calibrate count axes from the level, never from agent play.** `coins` and `kills`
  use `max: auto`, resolved from `level_path` at startup (`lvl-1` → 11 coins, 15
  enemies; `lvl-7` → 35 coins; `lvl-15` → 50 enemies). This matters: A* and the
  heuristic agent beeline for the exit and score **zero coins on every level**, so
  calibrating from observed play would have pinned the coin axis near zero and made it
  permanently inert. `time` also takes `max: auto`, but resolved from `game_time`
  rather than the level — that axis *is* the clock, so a hardcoded ceiling mis-scales
  silently whenever the game length changes. `jumps` has no ceiling to read and needs
  an explicit max; `max: auto` on it raises.
- **Kills exclude falls** — `max(0, getKillsTotal() - getKillsByFall())`, matching how
  `probs/smb/problem.py` counts enemies. An enemy that walks off a ledge isn't the
  player's doing.
- **Levels with bullet-bill spawners have no enemy ceiling.** `lvl-14` has 24 of them;
  `level_limits()` reports the count so `auto` can be treated with suspicion there.

Archives are not comparable across a measure change, and resume refuses a checkpoint
whose measure count doesn't match the config. Dimensionality also grows fast: 256 cells
against 9 evaluations per iteration means coverage stays near zero until runs get much
longer. Low coverage on a short run is expected, not a bug.

The analyzer projects an n-dimensional archive onto any two axes, taking the best
objective over the collapsed ones:

```bash
./venv/Scripts/python.exe analyzer.py --x time --y jumps --no-show
```

## Inputs vs. outputs

Keep this split — it is what stops runs from dirtying the repo:

- **`data/`** — tracked *inputs* only: the original levels and `weights.json`. Nothing
  written at runtime should land here.
- **`output/`** — everything a run produces: archive CSVs, heatmaps, and the `run/`,
  `gen/`, `listeners/` scratch dirs. Gitignored in full.

Resolve paths through `config.py` rather than hardcoding, so all three scripts agree:

```python
from config import load_config, get_output_dir
output_dir = get_output_dir(create=True)      # honors PCG_OUTPUT_DIR, then config, then output/smb
```

Precedence is `PCG_OUTPUT_DIR` env var → `output_dir` in `exp_config.yaml` → `output/smb`.
Use the env var to keep concurrent experiments from overwriting each other:

```bash
PCG_OUTPUT_DIR=./output/experiment-b ./venv/Scripts/python.exe driver.py
```

`analyzer.py` defaults both its input CSV and output PNG to the same directory, so with a
matching config it needs no arguments.

## Traps and history — read before debugging

A batch of bugs was fixed on this branch. The invariants below are what keep them
fixed; breaking one reintroduces a failure that is silent rather than loud.

**Weight loading must happen at agent construction.** `MarioGame.runGame()` calls
`setup()`, which calls `agent.initialize()` — so anything assigned to `agent.brain`
*between* an explicit `setup()` and `runGame()` used to be thrown away by the
re-initialize, and every QD evaluation silently scored a **fresh random network**.
Two guards now hold this together, and both matter:

- `nn.Agent.initialize()` only builds `SimpleNN` when `self.brain is None`, and
  re-applies `self.weights` / `self.weights_path` on every call.
- `runner.runLevelWithNet` passes the genome via `nn.Agent(seed, weights=net)`
  instead of poking `agent.brain` after setup.

Prefer the constructor route (`nn.Agent(weights=...)`) in new code. If you touch
`initialize()`, keep it idempotent — a regression here produces a plausible-looking
archive full of noise, not a crash.

**`smb` and `smbtile` engines are parallel copies.** `core.py`, `helper.py`, and
`sprites.py` are duplicated between the two packages. `smb`'s `MarioGame` previously
lacked the `setup()` method that `smbtile`'s had, which is why `driver.py` used to
import `MarioGame` from `smbtile` and `MarioForwardModel` from `smb`. `setup()` has
been extracted in `smb` too, so both are now equivalent and **the experiment imports
only from `smb`**. Don't reintroduce cross-package imports — the classes are distinct
types that happen to be structurally identical. If you fix a bug in one engine,
check whether the other needs the same fix.

**`runLevel` means two different things.** `runner.runLevel(levelString, gameTime, ...)`
always uses the NN agent, while `probs/smb/engine/__init__.py`'s
`runLevel(levelString, agentName, ...)` takes an agent name second. Passing an agent
name to the former collides with `gameTime`.

**Nothing should auto-write `data/smb/weights.json`.** `Agent.initialize` used to save
the freshly built network there whenever `weights_path` was `None` — from every Dask
worker, every evaluation, concurrently, clobbering the committed checkpoint. It no
longer writes anything implicitly; call `agent.save_weights(path)` explicitly.
`Agent.save_weights` delegates to `SimpleNN.save_weights` so the `model_state_dict`
layout stays in sync with the loader.

**`data/smb/weights.json` must track the current architecture.** The auto-save above was
masking a stale checkpoint: the committed file was a 3.8 MB save of an older, larger
network (32/64 channels, 3×3 kernels) that no current code could load, and it only ever
"worked" because a run would silently overwrite it first. It has been regenerated for the
current 2869-parameter network and is reproducible with `nn.Agent(seed=0)`:

```python
a = nn.Agent(seed=0); g = MarioGame(); g.setAgent(a); g.setup(level, 20, 0)
a.save_weights("./data/smb/weights.json")
```

**Regenerate it whenever you change `SimpleNN`'s architecture**, otherwise
`runner.runLevel()` breaks. `load_weights` now raises an explicit message naming the
expected parameter count rather than a wall of per-tensor size mismatches.

**`analyzer.py` needs a real archive.** `grid_archive_heatmap` takes elite *values*
from the DataFrame but grid *geometry* from the archive object, so the archive must
match the run's `coins_dim`/`kills_dim`. It is rebuilt from `exp_config.yaml`, with
`solution_dim` inferred from the CSV's `solution_*` columns. A mismatched stub
archive raises `ValueError: index N is out of bounds` from `int_to_grid_index`.

**Still-live oddities (not bugs, just surprising):**

- `listener.py` and `<output_dir>/{run,gen,listeners}` are a dead file-polling evaluation
  path from before Dask. `driver.clean_dirs()` still creates those dirs;
  `shutdown_listeners()` shells out to `pkill` and is commented out at the call site.
- `qd_score_offset=-600` in the archive is odd given the objective is in [0,1].

## Working in this repo

**Run things:**
```bash
./venv/Scripts/python.exe test.py                    # smoke-test all 15 benchmark problems (slow)
./venv/Scripts/python.exe runner.py                  # play one level with the checkpointed net
./venv/Scripts/python.exe driver.py                  # the QD experiment
./venv/Scripts/python.exe driver.py --resume-from output/smb/archive_5.csv
./venv/Scripts/python.exe analyzer.py --no-show      # heatmap -> output/smb/archive_final.png
./venv/Scripts/python.exe analyzer.py --archive output/smb/archive_5.csv --output /tmp/a.png
```

**Checks:**
```bash
./venv/Scripts/python.exe check_experiment.py           # ~30s, no Dask
./venv/Scripts/python.exe check_experiment.py --full    # also runs driver + analyzer
```

`check_experiment.py` pins the invariants below; run it after touching the engine, the
agent, or the path handling. It exits non-zero on failure.

| Check | Guards against |
| --- | --- |
| genome survives runGame | evolved weights replaced by a fresh network |
| weights assigned to `agent.brain` survive runGame | `initialize()` losing idempotence |
| evaluation is deterministic | hidden per-run randomness in fitness |
| different genomes behave differently | weights being ignored entirely |
| smb and smbtile engines agree | a refactor changing one engine copy's behavior |
| committed weights.json matches the network | a stale checkpoint after an architecture change |
| bad weights fail loudly | silent shape mismatches |
| measures extract from a real run | a measure accessor that no longer exists |
| kills measure excludes falls | falls creeping back into the behavior coordinate |
| measure ranges cover competent play | too-narrow ranges silently clipping behaviors |
| checkpoint columns order numerically | `--resume-from` permuting every genome |
| benchmark problems still evaluate | breaking upstream while editing the fork |
| output dir resolution honors overrides | runs writing back into tracked `data/` |
| driver end to end, data/ untouched (`--full`) | the whole pipeline, incl. analyzer |

The first two look redundant but are not, and the distinction matters if you edit
`nn.Agent`: the constructor route (`nn.Agent(weights=...)`) is re-applied on every
`initialize()`, so it survives even a rebuilt network, while weights poked onto
`agent.brain` only survive if `initialize()` is idempotent. Only the second check
catches a regression in that guard — verified by reintroducing the bug.

Otherwise there is no test suite, no linter config, and no CI. `test.py` is an upstream
smoke test over all 15 problems and covers only the benchmark half.

**Conventions:**
- Upstream code uses docstring-comment blocks *above* the `def`, not inside. Match the surrounding style when editing `pcg_benchmark/`.
- camelCase for engine/game code (ported from Java), snake_case for the framework and the branch scripts. Don't unify them.
- Keep `pcg_benchmark/` edits minimal and surgical — it's a fork that may need to merge from `upstream/main`. New experiment code belongs at the repo root or under `probs/smb/engine/agents/`.
- Don't commit into `venv/`, and be aware `data/smb/weights.json` is a 3.8 MB binary already tracked.

**Fast orientation for common asks:**

| Task | Start here |
| --- | --- |
| Change the evolved network | `probs/smb/engine/agents/networks/simplenn.py` + `agents/nn.py` |
| Change fitness / measures | `driver.run()` (objectives + measures), `MarioResult` in `probs/smb/engine/core.py` |
| Change QD algorithm / archive | `driver.create_archive` / `create_emitters` / `create_scheduler` |
| Change what the agent sees | `MarioForwardModel.getScreenCompleteObservation` in `probs/smb/engine/core.py` |
| Add a benchmark problem | `probs/README.md`, then copy `probs/binary/` as the simplest template |
| Debug game-sim behavior | `probs/smb/engine/core.py` (`MarioWorld.update`) and `sprites.py` |
