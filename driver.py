import os
import shutil
import json
from xmlrpc import client
import numpy as np
from ribs.archives import GridArchive
from ribs.emitters import EvolutionStrategyEmitter
from ribs.schedulers import Scheduler
from pcg_benchmark.probs.smb.engine.agents import nn
from pcg_benchmark.probs.smb.engine.core import MarioForwardModel
from pcg_benchmark.probs.smbtile.engine.core import MarioGame
from dask.distributed import Client
from runner import runLevel
import yaml


class MarioEvolutionDriver:
    def __init__(self, config_path="exp_config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.run_dir = os.path.join("data", "smb", "run")
        self.gen_dir = os.path.join("data", "smb", "gen")
        self.listeners_file = os.path.join("data", "smb", "listeners", "ids.txt")

        self.n_emitters = self.config.get("n_emitters", 5)
        self.n_iterations = self.config.get("n_iterations", 1)

        self.coins_dim = self.config.get("coins_dim", 10)
        self.kills_dim = self.config.get("kills_dim", 10)
        self.workers = self.config.get("workers", 2)
        level_path = self.config.get("level_path", "./data/smb/original/lvl-1.txt")
        with open(level_path, 'r') as file:
            self.level = file.read()

        self.archive, self.initial_model = self.create_archive()
        self.emitters = self.create_emitters(self.archive, self.initial_model)
        self.scheduler = self.create_scheduler(self.archive, self.emitters)

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
            solution_dim=initial_model.brain.get_weights().numel(),
            dims=[self.coins_dim, self.kills_dim],
            ranges=[(0.0, self.coins_dim), (0.0, self.kills_dim)],
            qd_score_offset=-600,
        )
        return archive, initial_model

    def create_emitters(self, archive, initial_model):
        print(initial_model.brain.get_weights().flatten())
        emitters = [
            EvolutionStrategyEmitter(
                archive=archive,
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
        self.clean_dirs()
        # self.shutdown_listeners()
        client = Client(
            n_workers=self.workers,  # Create this many worker processes using Dask LocalCluster.
            threads_per_worker=1,  # Each worker process is single-threaded.
        )

        for iteration in range(self.n_iterations):
            print(f"=== Iteration {iteration} ===")
            solutions = self.scheduler.ask()
            
            # Evaluate the models and record the objectives and measures.
            futures = client.map(lambda model: runLevel(model, self.level), solutions)
            results = client.gather(futures)
            print(results)
            objectives, measures = [], []

if __name__ == "__main__":
    driver = MarioEvolutionDriver()
    driver.run()