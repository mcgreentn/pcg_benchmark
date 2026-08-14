import os
import re
import shutil
from functools import partial
import pandas as pd
from ribs.archives import GridArchive
from ribs.emitters import EvolutionStrategyEmitter
from ribs.schedulers import Scheduler
from pcg_benchmark.probs.smb.engine.agents import nn
from pcg_benchmark.probs.smb.engine.core import MarioGame
from dask.distributed import Client
from runner import runLevelWithNet
from config import load_config, get_output_dir
from measures import load_measures, describe, DEFAULT_GAME_TIME
from tqdm import tqdm



class MarioEvolutionDriver:
    def __init__(self, config_path="exp_config.yaml", resume_from=None):
        self.config = load_config(config_path)

        # Run products are untracked and live outside data/, which holds only inputs.
        self.output_dir = get_output_dir(self.config, create=True)
        self.run_dir = os.path.join(self.output_dir, "run")
        self.gen_dir = os.path.join(self.output_dir, "gen")
        self.listeners_file = os.path.join(self.output_dir, "listeners", "ids.txt")

        self.n_emitters = self.config.get("n_emitters", 1)
        self.batch_size = self.config.get("batch_size", 3)
        self.n_iterations = self.config.get("n_iterations", 2)
        self.checkpoint_interval = self.config.get("checkpoint_interval", 100)

        self.sigma0 = self.config.get("sigma0", 1.0)
        self.ranker = self.config.get("ranker", "2imp")
        self.game_time = self.config.get("game_time", DEFAULT_GAME_TIME)
        # None means "different search every run"; an int makes the run reproducible.
        self.seed = self.config.get("seed", None)

        self.measures = load_measures(self.config)
        self.workers = self.config.get("workers", 2)
        level_path = self.config.get("level_path", "./data/smb/original/lvl-1.txt")
        with open(level_path, 'r') as file:
            self.level = file.read()

        evals = self.n_emitters * self.batch_size
        print(f"Archive: {describe(self.measures)}")
        print(f"Search:  {self.n_emitters} emitters x batch {self.batch_size} = {evals} evals/iter, "
              f"{evals * self.n_iterations} total over {self.n_iterations} iterations")
        print(f"Seed:    {self.seed if self.seed is not None else 'none (search will differ each run)'}")
        self.archive, self.initial_model = self.create_archive()
        self.emitters = self.create_emitters(self.archive, self.initial_model)
        self.scheduler = self.create_scheduler(self.archive, self.emitters)

        self.start_iteration = 0
        if resume_from is not None:
            self.start_iteration = self._load_checkpoint(resume_from)

    def clean_dirs(self):
        for d in [self.run_dir, self.gen_dir]:
            if os.path.exists(d):
                shutil.rmtree(d)
            os.makedirs(d)

    def shutdown_listeners(self):
        if os.path.exists(self.listeners_file):
            with open(self.listeners_file, "r") as f:
                listener_ids = [line.strip() for line in f.readlines()]
                print(f"Found listener ids: {listener_ids}")
                for lid in listener_ids:
                    os.system(f"pkill -f 'python listener.py {lid}'")
        else:
            print("No listener ids found.")

    def create_archive(self):
        # the initial model is a NN. setup() builds the forward model and calls
        # initialize() on the agent, which is what constructs initial_model.brain.
        # Seeding here makes x0 -- and therefore the whole search -- reproducible.
        initial_model = nn.Agent(seed=self.seed)
        game = MarioGame()
        game.setAgent(initial_model)
        game.setup(self.level, self.game_time, 0)
        archive = GridArchive(
            solution_dim=initial_model.brain.get_param_size(),
            dims=[m.bins for m in self.measures],
            ranges=[m.range for m in self.measures],
            qd_score_offset=-600,
            seed=self.seed,
        )
        return archive, initial_model

    def archive_path(self, name):
        """Path for an archive CSV inside the run-output directory."""
        return os.path.join(self.output_dir, name)

    def save_archive(self, path):
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        df = self.archive.data(return_type="pandas")
        # save this DF at the filepath location
        df.to_csv(path)

    @staticmethod
    def _indexed_columns(df, prefix):
        """
        Columns named <prefix>N, ordered by N numerically.

        Plain sorted() is lexicographic, which puts solution_1000 before solution_2
        and silently permutes every genome on resume -- the elites come back with
        correct objectives attached to scrambled weights.
        """
        cols = [c for c in df.columns if c.startswith(prefix) and c[len(prefix):].isdigit()]
        return sorted(cols, key=lambda c: int(c[len(prefix):]))

    def _load_checkpoint(self, path):
        """Restore archive elites from a checkpoint CSV. Returns the iteration to resume from."""
        print(f"Resuming from checkpoint: {path}")
        df = pd.read_csv(path, index_col=0)
        solution_cols = self._indexed_columns(df, "solution_")
        measure_cols = self._indexed_columns(df, "measures_")

        expected_solution = self.archive.solution_dim
        if len(solution_cols) != expected_solution:
            raise ValueError(
                f"Checkpoint has {len(solution_cols)} solution columns but the current network "
                f"has {expected_solution} parameters. The architecture changed since this "
                f"checkpoint was written."
            )
        if len(measure_cols) != len(self.measures):
            raise ValueError(
                f"Checkpoint has {len(measure_cols)} measures but the config defines "
                f"{len(self.measures)} ({', '.join(m.key for m in self.measures)}). "
                f"Archives are not comparable across a measure change -- start a fresh run."
            )

        self.archive.add(
            df[solution_cols].values,
            df["objective"].values,
            df[measure_cols].values,
        )
        print(f"Restored {len(df)} elites from checkpoint.")
        match = re.search(r'archive_(\d+)\.csv$', path)
        return int(match.group(1)) + 1 if match else 0

    def create_emitters(self, archive, initial_model):
        emitters = [
            EvolutionStrategyEmitter(
                archive=archive,
                # get_param() (numpy) rather than get_weights() (torch Tensor): it is the
                # inverse of the set_param() path that load_weights(ndarray) uses, so the
                # emitter seed → evolve → load round-trip stays consistently numpy and
                # keeps the same traversal order even if the architecture changes.
                x0=initial_model.brain.get_param(),
                sigma0=self.sigma0,
                ranker=self.ranker,
                batch_size=self.batch_size,
                # Distinct seed per emitter, else every emitter searches identically.
                seed=None if self.seed is None else self.seed + i,
            )
            for i in range(self.n_emitters)
        ]
        return emitters

    def create_scheduler(self, archive, emitters):
        return Scheduler(archive, emitters)

    def run(self):
        if self.start_iteration == 0:
            self.clean_dirs()
        # self.shutdown_listeners()
        client = Client(
            n_workers=self.workers,  # Create this many worker processes using Dask LocalCluster.
            threads_per_worker=1,  # Each worker process is single-threaded.
        )
        run_fn = partial(runLevelWithNet, self.level,
                         gameTime=self.game_time, seed=self.seed)
        with tqdm(range(self.start_iteration, self.n_iterations), desc="Iterations", position=0) as pbar:
            for iteration in pbar:
                solutions = self.scheduler.ask()
                futures = client.map(run_fn, solutions)
                results = list(tqdm(client.gather(futures), desc="Evaluating", total=len(solutions), position=1, leave=False))

                objectives, measures = [], []
                for result in results:
                    objectives.append(result.getCompletionPercentage())
                    measures.append([m.extract(result) for m in self.measures])

                self.scheduler.tell(objectives, measures)
                archive_stats = self.archive.stats
                pbar.set_postfix(
                    coverage=f"{archive_stats.coverage:.3f}",
                    elites=archive_stats.num_elites,
                    best=f"{archive_stats.obj_max:.3f}",
                )
                if iteration % self.checkpoint_interval == 0:
                    self.save_archive(self.archive_path(f"archive_{iteration}.csv"))

        # save archive grid at the end
        final_path = self.archive_path("archive_final.csv")
        self.save_archive(final_path)
        print(f"Saved final archive to {final_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume-from", default=None, help="Path to a checkpoint CSV to resume from (e.g. output/smb/archive_500.csv)")
    args = parser.parse_args()
    driver = MarioEvolutionDriver(resume_from=args.resume_from)
    driver.run()