"""
Capacity benchmark for the Mario neuroevolution experiment.

Answers the three questions you need to set `workers`, `n_emitters` and
`batch_size` on a machine you do not control (e.g. a JupyterHub container):

    1. What does this box ACTUALLY give me?   -> `probe`
    2. How long is one evaluation, and how many threads does it try to use?
                                              -> `eval`
    3. Where does throughput stop scaling, and what does it cost in RAM?
                                              -> `scale` / `threads`

Run it on the target machine, not on your laptop:

    ./venv/Scripts/python.exe benchmark.py probe        # instant, no evaluation
    ./venv/Scripts/python.exe benchmark.py all          # full sweep + recommendation

Everything is written to <output_dir>/bench/ as JSON + CSV so runs are comparable.

Why this exists rather than "just try workers=8": in a container `os.cpu_count()`
reports the HOST's cores, and torch sizes its intra-op thread pool from that. With
W Dask workers each spawning C torch threads you get W*C runnable threads against a
cgroup quota of maybe 4 cores -- the box thrashes and throughput goes DOWN as you
add workers. This script measures that instead of guessing at it.
"""
import argparse
import csv
import json
import os
import platform
import statistics
import sys
import threading
import time

import psutil


# --------------------------------------------------------------------------------
# 1. Environment probe
# --------------------------------------------------------------------------------

def _read_first_line(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def _cgroup_cpu_limit():
    """Cores available per the cgroup CPU quota, or None if unlimited/unreadable."""
    # cgroup v2: "<quota> <period>" or "max <period>"
    v2 = _read_first_line("/sys/fs/cgroup/cpu.max")
    if v2:
        parts = v2.split()
        if len(parts) == 2 and parts[0] != "max":
            return int(parts[0]) / int(parts[1])
        return None
    # cgroup v1
    quota = _read_first_line("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
    period = _read_first_line("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    if quota and period and int(quota) > 0:
        return int(quota) / int(period)
    return None


def _cgroup_mem_limit():
    """Memory ceiling in bytes per the cgroup, or None if unlimited/unreadable."""
    for path in ("/sys/fs/cgroup/memory.max",
                 "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        raw = _read_first_line(path)
        if raw and raw != "max":
            val = int(raw)
            # v1 reports a sentinel near 2^63 when unlimited.
            if val < (1 << 62):
                return val
    return None


def _cgroup_mem_current():
    for path in ("/sys/fs/cgroup/memory.current",
                 "/sys/fs/cgroup/memory/memory.usage_in_bytes"):
        raw = _read_first_line(path)
        if raw:
            return int(raw)
    return None


def _affinity():
    try:
        return len(os.sched_getaffinity(0))                # Linux
    except AttributeError:
        try:
            return len(psutil.Process().cpu_affinity())    # Windows
        except Exception:
            return None


def probe():
    """Collect everything that constrains how many workers can actually run."""
    import torch

    vm = psutil.virtual_memory()
    cg_cpu = _cgroup_cpu_limit()
    cg_mem = _cgroup_mem_limit()
    aff = _affinity()

    # The number that matters: the tightest of the ways this box can cap us.
    candidates = [c for c in (os.cpu_count(), aff, cg_cpu) if c]
    effective_cores = int(min(candidates)) if candidates else 1

    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "os_cpu_count": os.cpu_count(),
        "cpu_affinity": aff,
        "cgroup_cpu_limit": cg_cpu,
        "effective_cores": effective_cores,
        "torch_default_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "omp_num_threads_env": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads_env": os.environ.get("MKL_NUM_THREADS"),
        "mem_total_bytes": vm.total,
        "mem_available_bytes": vm.available,
        "cgroup_mem_limit_bytes": cg_mem,
        "cgroup_mem_current_bytes": _cgroup_mem_current(),
        # What a memory cap actually means for us: the tighter of container vs host.
        "mem_budget_bytes": min([m for m in (cg_mem, vm.total) if m]),
    }


def _gb(n):
    return "n/a" if n is None else f"{n / 2**30:.2f} GB"


def print_probe(p):
    print("=" * 72)
    print("ENVIRONMENT")
    print("=" * 72)
    print(f"  host                 {p['hostname']}")
    print(f"  platform             {p['platform']}")
    print(f"  python / torch       {p['python']} / {p['torch']}")
    print()
    print(f"  os.cpu_count()       {p['os_cpu_count']}")
    print(f"  cpu affinity         {p['cpu_affinity']}")
    print(f"  cgroup cpu quota     {p['cgroup_cpu_limit']}")
    print(f"  -> effective cores   {p['effective_cores']}")
    print()
    print(f"  torch threads        {p['torch_default_threads']} intra-op, "
          f"{p['torch_interop_threads']} inter-op")
    print(f"  OMP_NUM_THREADS      {p['omp_num_threads_env'] or '(unset)'}")
    print()
    print(f"  host memory          {_gb(p['mem_total_bytes'])} total, "
          f"{_gb(p['mem_available_bytes'])} available")
    print(f"  cgroup memory limit  {_gb(p['cgroup_mem_limit_bytes'])}")
    print(f"  cgroup memory in use {_gb(p['cgroup_mem_current_bytes'])}")
    print(f"  -> memory budget     {_gb(p['mem_budget_bytes'])}")
    print()

    cores, threads = p["effective_cores"], p["torch_default_threads"]
    if threads > 1:
        print(f"  WARNING: torch defaults to {threads} intra-op threads per process.")
        print(f"           W Dask workers => {threads}W runnable threads against "
              f"{cores} core(s).")
        print("           Pin it (OMP_NUM_THREADS=1 / torch.set_num_threads(1)) "
              "before scaling workers.")
    if p["os_cpu_count"] and p["os_cpu_count"] > cores:
        print(f"  WARNING: os.cpu_count() reports {p['os_cpu_count']} but only "
              f"{cores} are usable.")
        print(f"           Anything sizing a pool from cpu_count() oversubscribes by "
              f"{p['os_cpu_count'] / cores:.1f}x.")
    print()


# --------------------------------------------------------------------------------
# 2. Workload
# --------------------------------------------------------------------------------

def build_genomes(n, sigma, seed, level, game_time):
    """
    n genomes drawn the way an emitter draws them: gaussian noise of width sigma0
    around the initial network. Fixed seed, so every worker count in the sweep
    evaluates the IDENTICAL workload and the comparison is apples to apples.
    """
    import numpy as np
    from pcg_benchmark.probs.smb.engine.agents import nn
    from pcg_benchmark.probs.smb.engine.core import MarioGame

    agent = nn.Agent(seed=seed)
    game = MarioGame()
    game.setAgent(agent)
    game.setup(level, game_time, 0)
    x0 = agent.brain.get_param()

    rng = np.random.default_rng(seed if seed is not None else 0)
    return [x0 + rng.normal(0, sigma, x0.shape).astype(x0.dtype) for _ in range(n)]


def _bench_task(level, genome, game_time, seed, threads):
    """
    One evaluation, instrumented. Runs in a Dask worker (or inline for the serial
    baseline), so it must not close over anything unpicklable.
    """
    import torch
    if threads:
        torch.set_num_threads(threads)

    from runner import runLevelWithNet

    proc = psutil.Process()
    cpu0 = proc.cpu_times()
    t0 = time.perf_counter()
    # Pass `threads` through rather than relying on the set_num_threads above:
    # runLevelWithNet pins threads itself (default 1), so the sweep's higher thread
    # counts would be silently reset back to 1 and every row would look identical.
    result = runLevelWithNet(level, genome, gameTime=game_time, seed=seed,
                             threads=threads)
    wall = time.perf_counter() - t0
    cpu1 = proc.cpu_times()

    return {
        "pid": proc.pid,
        "wall": wall,
        "cpu": (cpu1.user - cpu0.user) + (cpu1.system - cpu0.system),
        "rss": proc.memory_info().rss,
        "torch_threads": torch.get_num_threads(),
        "completion": result.getCompletionPercentage(),
    }


# --------------------------------------------------------------------------------
# 3. Resource sampler
# --------------------------------------------------------------------------------

class TreeSampler:
    """
    Samples RSS across this process and all its children while a benchmark runs.

    RSS double-counts pages shared between forked workers, so a single USS snapshot
    is taken mid-run as well -- USS is the number that answers "how many more of
    these fit in my memory budget".
    """

    def __init__(self, interval=0.25):
        self.interval = interval
        self._stop = threading.Event()
        self.peak_tree_rss = 0
        self.peak_child_rss = 0
        self.peak_child_count = 0
        self.uss_sample = None
        self._thread = None

    def _tree(self):
        me = psutil.Process()
        return [me] + me.children(recursive=True)

    def _loop(self):
        uss_taken = False
        while not self._stop.is_set():
            total, biggest, n = 0, 0, 0
            procs = self._tree()
            for p in procs:
                try:
                    rss = p.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
                total += rss
                if p.pid != os.getpid():
                    n += 1
                    biggest = max(biggest, rss)
            self.peak_tree_rss = max(self.peak_tree_rss, total)
            self.peak_child_rss = max(self.peak_child_rss, biggest)
            self.peak_child_count = max(self.peak_child_count, n)

            # One USS pass once the workers are warm. It walks the page map, so it is
            # far more expensive than RSS -- sampling it every tick would perturb the
            # very thing being measured.
            if not uss_taken and n > 0:
                uss = 0
                for p in procs:
                    try:
                        uss += p.memory_full_info().uss
                    except Exception:
                        uss = None
                        break
                self.uss_sample = uss
                uss_taken = True

            self._stop.wait(self.interval)

    def __enter__(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=2)
        return False


# --------------------------------------------------------------------------------
# 4. Benchmarks
# --------------------------------------------------------------------------------

def bench_serial(level, genomes, game_time, seed, threads):
    """Single-process baseline. Everything else is measured relative to this."""
    print("=" * 72)
    print(f"SERIAL BASELINE  ({len(genomes)} evals, torch threads="
          f"{threads or 'default'})")
    print("=" * 72)

    # Discard one evaluation first. torch initializes its thread pool and allocator
    # on the first forward, and the parallel path warms every worker before its clock
    # starts -- leaving this out makes the baseline artificially slow and hands the
    # sweep a free speedup it did not earn (efficiencies over 100%).
    t_warm = time.perf_counter()
    _bench_task(level, genomes[0], game_time, seed, threads)
    print(f"  warm-up      {time.perf_counter() - t_warm:.2f}s (discarded)")

    rss_before = psutil.Process().memory_info().rss
    records = []
    t0 = time.perf_counter()
    for i, g in enumerate(genomes):
        rec = _bench_task(level, g, game_time, seed, threads)
        records.append(rec)
        print(f"  eval {i + 1:>3}/{len(genomes)}  {rec['wall']:6.2f}s  "
              f"cpu {rec['cpu']:6.2f}s  completion {rec['completion']:.3f}")
    wall = time.perf_counter() - t0
    rss_after = psutil.Process().memory_info().rss

    walls = [r["wall"] for r in records]
    cpus = [r["cpu"] for r in records]
    out = {
        "evals": len(records),
        "wall_total": wall,
        "wall_mean": statistics.mean(walls),
        "wall_median": statistics.median(walls),
        "wall_min": min(walls),
        "wall_max": max(walls),
        "wall_stdev": statistics.stdev(walls) if len(walls) > 1 else 0.0,
        "cpu_mean": statistics.mean(cpus),
        # >1 means the eval itself is using more than one core (thread pool at work).
        "cpu_per_wall": sum(cpus) / wall,
        "throughput": len(records) / wall,
        "torch_threads": records[0]["torch_threads"],
        "rss_before": rss_before,
        "rss_after": rss_after,
        "rss_growth": rss_after - rss_before,
    }
    print()
    print(f"  per-eval    mean {out['wall_mean']:.2f}s  median "
          f"{out['wall_median']:.2f}s  min {out['wall_min']:.2f}s  "
          f"max {out['wall_max']:.2f}s  sd {out['wall_stdev']:.2f}s")
    print(f"  throughput  {out['throughput']:.3f} evals/s")
    print(f"  cores used  {out['cpu_per_wall']:.2f} (cpu-seconds per wall-second, "
          f"1 process)")
    print(f"  process rss {_gb(rss_after)} after warm-up "
          f"(+{_gb(out['rss_growth'])} during evals)")
    print()
    return out


def bench_parallel(level, genomes, game_time, seed, n_workers, threads, mem_limit):
    """One point on the scaling curve: n_workers Dask processes, threads each."""
    from dask.distributed import Client, LocalCluster

    # Belt and braces: the env vars cover libraries that size their pool at import
    # (OpenMP, MKL), torch.set_num_threads inside the task covers torch itself.
    env = {}
    if threads:
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS"):
            env[var] = str(threads)
            os.environ[var] = str(threads)

    t_boot = time.perf_counter()
    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=1,
        processes=True,
        # Measure raw demand; the nanny pausing or killing workers mid-run would
        # silently distort both the timing and the memory peak.
        memory_limit=mem_limit,
        dashboard_address=None,
        env=env or None,
        silence_logs=50,
    )
    client = Client(cluster)
    boot = time.perf_counter() - t_boot

    try:
        # Warm up: pay the torch import and first-forward cost on every worker before
        # the clock starts, so the curve measures steady-state throughput.
        t_warm = time.perf_counter()
        client.gather(client.map(_bench_task,
                                 [level] * n_workers,
                                 [genomes[0]] * n_workers,
                                 [game_time] * n_workers,
                                 [seed] * n_workers,
                                 [threads] * n_workers,
                                 pure=False))
        warm = time.perf_counter() - t_warm

        with TreeSampler() as sampler:
            t0 = time.perf_counter()
            futures = client.map(_bench_task,
                                 [level] * len(genomes),
                                 genomes,
                                 [game_time] * len(genomes),
                                 [seed] * len(genomes),
                                 [threads] * len(genomes),
                                 pure=False)
            records = client.gather(futures)
            wall = time.perf_counter() - t0
    finally:
        client.close()
        cluster.close()

    cpu_total = sum(r["cpu"] for r in records)
    walls = [r["wall"] for r in records]
    return {
        "workers": n_workers,
        "threads": threads or 0,
        "evals": len(records),
        "wall": wall,
        "boot": boot,
        "warmup": warm,
        "throughput": len(records) / wall,
        "eval_wall_mean": statistics.mean(walls),
        "eval_wall_max": max(walls),
        # Cores genuinely consumed. Well under `workers` means starved (or the box is
        # shared); well over means the thread pools are oversubscribing.
        "cores_used": cpu_total / wall,
        "cpu_total": cpu_total,
        "distinct_pids": len({r["pid"] for r in records}),
        "torch_threads_observed": records[0]["torch_threads"],
        "peak_tree_rss": sampler.peak_tree_rss,
        "peak_worker_rss": sampler.peak_child_rss,
        "peak_child_count": sampler.peak_child_count,
        "uss_sample": sampler.uss_sample,
    }


def print_scale_table(rows, baseline_throughput):
    print()
    print(f"{'workers':>7} {'thr':>4} {'wall':>8} {'ev/s':>8} {'speedup':>8} "
          f"{'eff':>6} {'cores':>7} {'peakRSS':>10} {'per-wkr':>9} {'boot':>6}")
    print("-" * 80)
    for r in rows:
        speedup = r["throughput"] / baseline_throughput if baseline_throughput else 0
        eff = speedup / r["workers"] if r["workers"] else 0
        print(f"{r['workers']:>7} {r['threads'] or 'def':>4} "
              f"{r['wall']:>7.2f}s {r['throughput']:>8.3f} {speedup:>7.2f}x "
              f"{eff:>5.0%} {r['cores_used']:>7.2f} "
              f"{r['peak_tree_rss'] / 2**30:>9.2f}G "
              f"{r['peak_worker_rss'] / 2**20:>8.0f}M {r['boot']:>5.1f}s")
    print()
    print("  eff   = speedup / workers. Below ~70% you are paying for workers you "
          "do not get.")
    print("  cores = cpu-seconds per wall-second across the whole process tree.")
    thin = [r for r in rows if r["evals"] < 4 * r["workers"]]
    if thin:
        print(f"  NOTE: {len(thin)} point(s) ran fewer than 4 evals per worker. "
              f"Each wave is one eval deep,")
        print("        so a single slow genome moves the whole number. Re-run those "
              "with a larger --evals.")
    print()


# --------------------------------------------------------------------------------
# 5. Recommendation
# --------------------------------------------------------------------------------

def recommend(env, serial, rows, config, headroom=0.8):
    print("=" * 72)
    print("RECOMMENDATION")
    print("=" * 72)

    best = max(rows, key=lambda r: r["throughput"])
    # The cheapest config within 5% of peak: past that the extra workers buy noise
    # and cost memory.
    near = [r for r in rows if r["throughput"] >= 0.95 * best["throughput"]]
    knee = min(near, key=lambda r: r["workers"])

    per_worker = best["peak_worker_rss"]
    budget = env["mem_budget_bytes"]
    baseline = serial["rss_after"] if serial else 0
    mem_cap = None
    if per_worker:
        mem_cap = int(max(1, (budget * headroom - baseline) // per_worker))

    workers = knee["workers"]
    if mem_cap is not None:
        workers = min(workers, mem_cap)

    print(f"  peak throughput      {best['throughput']:.3f} evals/s at "
          f"workers={best['workers']}")
    print(f"  knee (within 5%)     workers={knee['workers']} "
          f"({knee['throughput']:.3f} evals/s, {knee['cores_used']:.1f} cores used)")
    print(f"  memory per worker    {per_worker / 2**20:.0f} MB peak RSS")
    if mem_cap is not None:
        print(f"  memory ceiling       {mem_cap} workers "
              f"({headroom:.0%} of {_gb(budget)} minus {_gb(baseline)} baseline)")
    print()
    print(f"  -> workers: {workers}")

    # evals/iter below workers leaves workers idle every single iteration; not a
    # multiple of workers leaves a ragged tail on each one.
    n_em = config.get("n_emitters", 1)
    bs = config.get("batch_size", 1)
    evals_iter = n_em * bs
    print(f"  -> n_emitters x batch_size should be a multiple of {workers} "
          f"and >= {workers}")
    print(f"     current: {n_em} x {bs} = {evals_iter} evals/iter", end="")
    if evals_iter < workers:
        print(f"  <-- {workers - evals_iter} of {workers} workers idle EVERY iteration")
    elif evals_iter % workers:
        print(f"  <-- ragged tail: last wave runs {evals_iter % workers}/{workers} "
              f"workers")
    else:
        print("  (clean fit)")

    if env["torch_default_threads"] > 1:
        print()
        print(f"  -> Pin torch to 1 thread. It defaults to "
              f"{env['torch_default_threads']} here, and nothing in the repo sets it.")
        print("     export OMP_NUM_THREADS=1   (before launching driver.py)")

    est = evals_iter / best["throughput"] if best["throughput"] else 0
    iters = config.get("n_iterations", 0)
    print()
    print(f"  at peak throughput one iteration is ~{est:.1f}s; "
          f"{iters} iterations ~= {est * iters / 60:.1f} min")
    print()
    return {"workers": workers, "knee": knee["workers"], "peak": best["workers"],
            "mem_cap_workers": mem_cap, "per_worker_bytes": per_worker}


# --------------------------------------------------------------------------------
# 6. Entry point
# --------------------------------------------------------------------------------

# Cluster spin-up + per-worker warm-up, amortized. Only used for the time estimate,
# so a rough constant beats pretending to model it.
BOOT_OVERHEAD_S = 12


def print_plan(args, env, ladder, evals_at, pool_size, threads):
    """What `all` would do, and roughly how long, without evaluating anything."""
    per_eval = args.per_eval_estimate
    print("=" * 72)
    print("PLAN (dry run -- nothing was evaluated)")
    print("=" * 72)
    print(f"  assumed cost per eval   {per_eval:.2f}s "
          f"(override with --per-eval-estimate)")
    print(f"  torch threads           {threads or 'left at default'}")
    print()

    total = 0.0
    if args.mode in ("eval", "all"):
        t = (args.serial_evals + 1) * per_eval
        total += t
        print(f"  serial baseline         {args.serial_evals} evals "
              f"(+1 warm-up)        ~{t:>6.0f}s")

    if args.mode in ("scale", "all"):
        print(f"  scaling sweep:")
        for w in ladder:
            n = evals_at(w)
            t = n * per_eval / w + BOOT_OVERHEAD_S
            total += t
            print(f"    workers={w:<5} {n:>5} evals                    ~{t:>6.0f}s")

    if args.mode in ("threads", "all"):
        import torch
        tsweep = ([int(x) for x in args.thread_sweep.split(",")]
                  if args.thread_sweep else sorted({1, 2, torch.get_num_threads()}))
        w = args.thread_workers or env["effective_cores"]
        n = evals_at(w)
        print(f"  thread sweep at workers={w}:")
        for t_n in tsweep:
            t = n * per_eval / w + BOOT_OVERHEAD_S
            total += t
            print(f"    threads={t_n:<5} {n:>5} evals                    ~{t:>6.0f}s")

    print()
    print(f"  ESTIMATED TOTAL         ~{total / 60:.1f} min")
    print()
    print(f"  peak memory at the widest point: "
          f"~{ladder[-1] * 500 / 1024:.1f} GB "
          f"({ladder[-1]} workers x ~500 MB, budget "
          f"{_gb(env['mem_budget_bytes'])})")
    print(f"  points projected over {args.headroom:.0%} of budget are skipped "
          f"automatically.")
    print()
    print("  Re-run without --dry-run to execute.")
    print()


def default_worker_ladder(cores):
    """1, 2, 4, ... up to cores, plus one point past it to show the cliff."""
    ladder, w = [], 1
    while w < cores:
        ladder.append(w)
        w *= 2
    ladder.append(cores)
    ladder.append(cores * 2)
    return sorted(set(ladder))


def save_results(outdir, payload):
    os.makedirs(outdir, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    host = payload["env"]["hostname"].replace(" ", "_")
    base = os.path.join(outdir, f"bench-{host}-{stamp}")

    with open(base + ".json", "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"Wrote {base}.json")

    rows = (payload.get("scale") or []) + (payload.get("threads") or [])
    if rows:
        with open(base + ".csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {base}.csv")
    return base


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["probe", "eval", "scale", "threads", "all"],
                    help="probe: environment only. eval: serial baseline. "
                         "scale: worker sweep. threads: thread oversubscription. "
                         "all: everything + recommendation.")
    ap.add_argument("--config", default="exp_config.yaml")
    ap.add_argument("--evals-per-worker", type=int, default=4,
                    help="evals per worker at each sweep point (default 4). Sizing the "
                         "work to the worker count keeps every point about the same "
                         "wall-clock length.")
    ap.add_argument("--evals", type=int, default=None,
                    help="fixed evals at EVERY sweep point, overriding "
                         "--evals-per-worker. Identical workload per point, but the "
                         "1-worker point then takes as long as the widest one does "
                         "times its worker count.")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and a time estimate, then exit without "
                         "evaluating anything")
    ap.add_argument("--per-eval-estimate", type=float, default=2.5,
                    help="seconds per evaluation assumed by --dry-run (default 2.5)")
    ap.add_argument("--serial-evals", type=int, default=5,
                    help="evals for the serial baseline (default 5)")
    ap.add_argument("--workers", default=None,
                    help="comma-separated worker counts (default: 1,2,4,... around "
                         "the effective core count)")
    ap.add_argument("--threads", type=int, default=1,
                    help="torch intra-op threads per worker (0 = leave at default)")
    ap.add_argument("--thread-workers", type=int, default=None,
                    help="worker count to hold fixed in `threads` mode (default: the "
                         "best worker count from the scale sweep, else all cores)")
    ap.add_argument("--thread-sweep", default=None,
                    help="comma-separated thread counts for `threads` mode "
                         "(default: 1,2,<torch default>)")
    ap.add_argument("--game-time", type=int, default=None,
                    help="override game_time from the config")
    ap.add_argument("--sigma", type=float, default=None,
                    help="override sigma0 when drawing benchmark genomes")
    ap.add_argument("--mem-limit", default="0",
                    help="Dask per-worker memory_limit; 0 disables the nanny so raw "
                         "demand is measured (default 0)")
    ap.add_argument("--headroom", type=float, default=0.8,
                    help="fraction of the memory budget to plan against (default 0.8)")
    ap.add_argument("--outdir", default=None, help="default: <output_dir>/bench")
    args = ap.parse_args()

    env = probe()
    print_probe(env)
    if args.mode == "probe":
        return

    from config import load_config, get_output_dir
    config = load_config(args.config)
    game_time = args.game_time or config.get("game_time", 20)
    sigma = args.sigma if args.sigma is not None else config.get("sigma0", 1.0)
    seed = config.get("seed", 0)
    level_path = config.get("level_path", "./data/smb/original/lvl-1.txt")
    with open(level_path) as f:
        level = f.read()

    outdir = args.outdir or os.path.join(get_output_dir(config), "bench")
    threads = args.threads or None

    if args.workers:
        ladder = sorted({int(x) for x in args.workers.split(",") if int(x) > 0})
    else:
        ladder = default_worker_ladder(env["effective_cores"])

    # Evals at each sweep point. Scaling with the worker count is the default because
    # a fixed count sized for the widest point makes the 1-worker point take that
    # count x per_eval seconds -- on a 64-core box, hours. Scaled, every point runs
    # roughly evals_per_worker x per_eval seconds regardless of width.
    def evals_at(w):
        return args.evals if args.evals else max(4, args.evals_per_worker * w)

    n_evals = args.evals or max(4, args.evals_per_worker * max(ladder))
    payload = {"env": env, "args": vars(args), "config": config,
               "game_time": game_time, "sigma": sigma, "level_path": level_path}

    print(f"Workload: level={level_path}  game_time={game_time}s  sigma={sigma}  "
          f"seed={seed}")
    print()

    if args.dry_run:
        print_plan(args, env, ladder, evals_at, n_evals, threads)
        return

    serial = None
    if args.mode in ("eval", "all"):
        genomes = build_genomes(args.serial_evals, sigma, seed, level, game_time)
        serial = bench_serial(level, genomes, game_time, seed, threads)
        payload["serial"] = serial

    if args.mode in ("scale", "all"):
        # One pool, and every point takes a PREFIX of it, so the smaller runs are a
        # strict subset of the larger ones -- same genomes, same cost distribution.
        pool = build_genomes(n_evals, sigma, seed, level, game_time)
        per_eval = serial["wall_mean"] if serial else args.per_eval_estimate
        est = sum(evals_at(w) * per_eval / w + BOOT_OVERHEAD_S for w in ladder)
        print("=" * 72)
        print(f"SCALING SWEEP  workers={ladder}  torch threads={threads or 'default'}")
        print(f"  evals per point: "
              f"{'fixed ' + str(args.evals) if args.evals else str(args.evals_per_worker) + ' per worker'}")
        print(f"  rough estimate: {est / 60:.1f} min")
        print("=" * 72)

        rows = []
        ceiling = env["mem_budget_bytes"] * args.headroom
        for w in ladder:
            # Refuse to walk off the memory cliff. On a container, OOM-killing
            # yourself mid-sweep loses every result collected so far -- and the point
            # of the sweep is to FIND the ceiling, not to hit it.
            if rows:
                per_worker = rows[-1]["peak_worker_rss"]
                projected = per_worker * w
                if per_worker and projected > ceiling:
                    print(f"  workers={w} ... SKIPPED: projected "
                          f"{projected / 2**30:.2f}G > {ceiling / 2**30:.2f}G budget "
                          f"({args.headroom:.0%} of {_gb(env['mem_budget_bytes'])})")
                    continue
            n = evals_at(w)
            print(f"  workers={w:<4} {n:>4} evals ...", end="", flush=True)
            r = bench_parallel(level, pool[:n], game_time, seed, w, threads,
                               args.mem_limit)
            rows.append(r)
            print(f" {r['wall']:.2f}s  {r['throughput']:.3f} ev/s  "
                  f"{r['cores_used']:.2f} cores  "
                  f"{r['peak_tree_rss'] / 2**30:.2f}G peak")
        payload["scale"] = rows
        print_scale_table(rows, serial["throughput"] if serial
                          else rows[0]["throughput"])

    if args.mode in ("threads", "all"):
        import torch
        if args.thread_sweep:
            tsweep = sorted({int(x) for x in args.thread_sweep.split(",")})
        else:
            tsweep = sorted({1, 2, torch.get_num_threads()})
        # Hold workers fixed at whatever the scale sweep liked, so the only variable
        # is thread count. Falling back to effective_cores can be more workers than
        # the box has memory for, hence the explicit override.
        if args.thread_workers:
            w = args.thread_workers
        elif payload.get("scale"):
            w = max(payload["scale"], key=lambda r: r["throughput"])["workers"]
        else:
            w = env["effective_cores"]
        n = evals_at(w)
        genomes = build_genomes(n, sigma, seed, level, game_time)
        print("=" * 72)
        print(f"THREAD OVERSUBSCRIPTION  workers={w} fixed, {n} evals, "
              f"torch threads={tsweep}")
        print("=" * 72)
        trows = []
        for t in tsweep:
            print(f"  threads={t} ...", end="", flush=True)
            r = bench_parallel(level, genomes, game_time, seed, w, t, args.mem_limit)
            trows.append(r)
            print(f" {r['wall']:.2f}s  {r['throughput']:.3f} ev/s  "
                  f"{r['cores_used']:.2f} cores")
        payload["threads"] = trows
        best_t = max(trows, key=lambda r: r["throughput"])
        print()
        for r in trows:
            tag = "  <-- best" if r is best_t else ""
            print(f"  {r['threads']:>3} threads x {w} workers: "
                  f"{r['throughput']:.3f} ev/s, {r['cores_used']:.2f} cores used{tag}")
        print()

    if args.mode == "all" and payload.get("scale"):
        payload["recommendation"] = recommend(env, serial, payload["scale"], config,
                                              args.headroom)

    save_results(outdir, payload)


if __name__ == "__main__":
    main()
