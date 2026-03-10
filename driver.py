import os
import re
import shutil
from functools import partial
import pandas as pd
from ribs.archives import GridArchive
from ribs.emitters import EvolutionStrategyEmitter
from ribs.schedulers import Scheduler
from pcg_benchmark.probs.smb.engine.agents import nn
from pcg_benchmark.probs.smb.engine.core import MarioForwardModel
from pcg_benchmark.probs.smbtile.engine.core import MarioGame
from dask.distributed import Client
from runner import runLevelWithNet
import yaml
from tqdm import tqdm



class MarioEvolutionDriver:
    def __init__(self, config_path="exp_config.yaml", resume_from=None):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.run_dir = os.path.join("data", "smb", "run")
        self.gen_dir = os.path.join("data", "smb", "gen")
        self.listeners_file = os.path.join("data", "smb", "listeners", "ids.txt")

        self.n_emitters = self.config.get("n_emitters", 1)
        self.n_iterations = self.config.get("n_iterations", 2)
        self.checkpoint_interval = self.config.get("checkpoint_interval", 100)

        self.coins_dim = self.config.get("coins_dim", 2)
        self.kills_dim = self.config.get("kills_dim", 2)
        self.workers = self.config.get("workers", 2)
        level_path = self.config.get("level_path", "./data/smb/original/lvl-1.txt")
        with open(level_path, 'r') as file:
            self.level = file.read()

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
        # the initial model is a NN
        initial_model = nn.Agent()
        game = MarioGame()
        game.setAgent(initial_model)
        game.setup(self.level, 20, 0)
        fwdModel = MarioForwardModel(game._world.clone())
        initial_model.initialize(fwdModel)
        archive = GridArchive(
            solution_dim=initial_model.brain.get_param_size(),
            dims=[self.coins_dim, self.kills_dim],
            ranges=[(0.0, self.coins_dim), (0.0, self.kills_dim)],
            qd_score_offset=-600,
        )
        return archive, initial_model
    def save_archive(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        df = self.archive.data(return_type="pandas")
        # save this DF at the filepath location
        df.to_csv(path)

    def _load_checkpoint(self, path):
        """Restore archive elites from a checkpoint CSV. Returns the iteration to resume from."""
        print(f"Resuming from checkpoint: {path}")
        df = pd.read_csv(path, index_col=0)
        solution_cols = sorted([c for c in df.columns if c.startswith("solution_")])
        measure_cols = sorted([c for c in df.columns if c.startswith("measures_")])
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
                # get_weights() returns a torch Tensor; get_param() returns numpy.
                # Both traverse parameters in the same order for this network, but
                # get_param() matches the set_param() path used by load_weights(ndarray),
                # making the full emitter seed → evolve → load round-trip consistently numpy.
                # Consider switching to get_param() if the network architecture ever changes.
                x0=initial_model.brain.get_weights().flatten(),
                sigma0=1.0,
                ranker="2imp",
                batch_size=3,
            )
            for _ in range(self.n_emitters)
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
        run_fn = partial(runLevelWithNet, self.level)
        with tqdm(range(self.start_iteration, self.n_iterations), desc="Iterations", position=0) as pbar:
            for iteration in pbar:
                solutions = self.scheduler.ask()
                futures = client.map(run_fn, solutions)
                results = list(tqdm(client.gather(futures), desc="Evaluating", total=len(solutions), position=1, leave=False))

                objectives, measures = [], []
                for result in results:
                    objectives.append(result.getCompletionPercentage())
                    measures.append([result.getNumCollectedTileCoins(), result.getKillsTotal()])

                self.scheduler.tell(objectives, measures)
                archive_stats = self.archive.stats
                pbar.set_postfix(
                    coverage=f"{archive_stats.coverage:.3f}",
                    elites=archive_stats.num_elites,
                    best=f"{archive_stats.obj_max:.3f}",
                )
                if iteration % self.checkpoint_interval == 0:
                    self.save_archive(f"./data/smb/archive_{iteration}.csv")
            

        # save archive grid at the end
        self.save_archive("./data/smb/archive_final.csv")
        
        

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume-from", default=None, help="Path to a checkpoint CSV to resume from (e.g. data/smb/archive_500.csv)")
    args = parser.parse_args()
    driver = MarioEvolutionDriver(resume_from=args.resume_from)
    driver.run()