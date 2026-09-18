# Sizing the experiment on a remote box

How to find the right `workers`, `torch_threads`, `n_emitters` and `batch_size` for a
machine you don't control — a JupyterHub container, a shared node, anything where the
resources you *appear* to have and the resources you *get* are different numbers.

Run everything below **on the target machine**, not on your laptop.

---

## 0. Why you can't just guess

Inside a container, `os.cpu_count()` reports the **host's** cores, not your cgroup
quota. Every library that sizes a thread pool from it — torch included — then
oversubscribes by whatever the ratio happens to be. With `W` Dask worker processes
each spawning `C` torch threads you get `W × C` runnable threads against a quota of
maybe four cores, and throughput goes *down* as you add workers.

Measured on a 12-core laptop at 4 workers, 16 evals: **23.5 s unpinned vs 9.2 s at one
torch thread per worker.** Same work, 2.5× the wall clock, purely from thread
contention. On a container with a 4-core quota and a 64-core host it is worse.

That's what this measures instead of guessing at.

---

## 1. One-time setup on the remote

The `venv/` committed in this repo is a **Windows** venv (`venv/Scripts/python.exe`).
It will not run on Linux. Build one there:

```bash
cd /path/to/pcg_benchmark
python3 -m venv venv-linux
. venv-linux/bin/activate
pip install -e .
```

Confirm the pieces import:

```bash
python -c "import torch, dask, distributed, ribs, psutil, runner, config; print('ok')"
```

If `psutil` is missing, `pip install psutil` — it normally arrives with `distributed`.

---

## 2. Look before you leap

```bash
./run_benchmark.sh --dry-run
```

This probes the machine and prints the plan **without evaluating anything**. Two
things to check in the output:

- **`-> effective cores`** — the `min` of `os.cpu_count()`, CPU affinity, and the
  cgroup quota. This, not `cpu_count`, is what you have. If the probe warns that they
  disagree, that's the container talking.
- **`ESTIMATED TOTAL`** — should be a few minutes. The sweep sizes each point's work
  to its worker count, so every point takes roughly the same wall time regardless of
  how wide it is.

Adjust the assumed per-eval cost if the estimate looks wrong for your box:

```bash
./run_benchmark.sh --dry-run --per-eval-estimate 5
```

---

## 3. Run it

```bash
tmux new -s bench          # JupyterHub will kill the process if your tab closes
./run_benchmark.sh
```

Everything is logged to `<output_dir>/bench/run-<host>-<stamp>.log`, with machine-
readable results beside it as `bench-<host>-<stamp>.json` and `.csv`.

Want steadier numbers? More evals per point:

```bash
./run_benchmark.sh --evals-per-worker 8
```

Only care about specific widths?

```bash
./run_benchmark.sh --workers 1,4,8,16
```

The sweep **skips** any worker count whose projected memory exceeds 80% of your
budget (`--headroom` to change). On a container, OOM-killing yourself mid-sweep loses
every result collected up to that point, and the whole exercise is to find that
ceiling without hitting it.

---

## 4. Reading the output

### Scaling table

```
workers  thr     wall     ev/s  speedup    eff   cores    peakRSS   per-wkr   boot
      1    1   21.76s    0.735    0.85x   85%    0.95      0.91G      472M   1.5s
      2    1   16.60s    0.964    1.12x   56%    1.65      1.38G      474M   2.2s
      4    1   12.28s    1.303    1.51x   38%    2.50      2.32G      478M   5.6s
```

- **`ev/s`** — the number that matters. Where it stops climbing is your answer.
- **`eff`** = speedup ÷ workers. Below ~70% you're paying for workers you don't get.
- **`cores`** = cpu-seconds per wall-second across the whole process tree. Compare it
  to `workers`: well *under* means starved (a busy shared node, or I/O), well *over*
  means thread pools are oversubscribing.
- **`per-wkr`** — peak RSS of the heaviest worker, ~475 MB and dominated by torch.
  This sets a worker ceiling that has nothing to do with core count.

### Thread table

```
    1 threads x 4 workers: 1.731 ev/s, 3.00 cores used  <-- best
    6 threads x 4 workers: 0.682 ev/s, 4.63 cores used
```

Expect 1 to win. If it doesn't on your box, that's a genuine surprise worth keeping.

### Recommendation block

The last section reports the throughput peak, the **knee** (cheapest config within 5%
of peak — past it you're buying noise and paying memory for it), the memory ceiling,
and a concrete `workers` value that respects both.

---

## 5. Applying the result

Edit `exp_config.yaml`:

```yaml
workers: 8             # from the recommendation
torch_threads: 1       # unless the thread sweep says otherwise

n_emitters: 4
batch_size: 2          # 4 x 2 = 8 evals/iter — a clean multiple of workers
```

**`n_emitters × batch_size` is the whole wave Dask gets per iteration.** Get this
wrong and the worker count doesn't matter:

- Fewer than `workers` → some workers idle *every single iteration*.
- Not a multiple of `workers` → every iteration ends on a ragged partial wave.

The recommendation block checks your current config against this and says which case
you're in.

Then confirm the real thing behaves like the benchmark said:

```bash
python driver.py
```

It prints `Compute: N workers x T torch thread(s)` at startup. The per-iteration time
in the progress bar should land near `evals_per_iter ÷ ev/s` from the sweep.

---

## 6. Troubleshooting

| Symptom | Cause |
| --- | --- |
| `eff` collapses immediately, `cores` stays near 1 | The cgroup quota is ~1 core. Check `probe`. More workers will not help. |
| `cores` far exceeds `workers` | Thread pinning isn't taking. Check `OMP_NUM_THREADS` in the probe and that `torch_threads` is set. |
| Sweep skips most of the ladder | Memory-bound, not CPU-bound. ~475 MB/worker against your budget is the binding constraint. |
| Throughput varies wildly between runs | Shared node — someone else is on it. Re-run with `--evals-per-worker 8` and compare logs. |
| `ModuleNotFoundError` | The Windows `venv/` got picked up, or `pip install -e .` wasn't run. Set `PYTHON=` explicitly. |

Per-eval cost also scales with `game_time` (default 20 s of simulated play) — if you
change it, the sizing changes with it.
