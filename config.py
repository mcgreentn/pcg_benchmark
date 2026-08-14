"""
Shared configuration and path resolution for the Mario neuroevolution experiment.

Keeps a hard split between the two kinds of files the experiment touches:

  data/   tracked inputs  -- original levels, the initial weights checkpoint
  output/ untracked runs  -- archive CSVs, heatmaps, and scratch directories

Everything a run produces goes under the output directory so the repo stays clean.
Override it with `output_dir` in exp_config.yaml or the PCG_OUTPUT_DIR env var.
"""
import os
import yaml

DEFAULT_CONFIG_PATH = "exp_config.yaml"
DEFAULT_OUTPUT_DIR = os.path.join("output", "smb")


def load_config(config_path=DEFAULT_CONFIG_PATH):
    """Load the experiment config, tolerating a missing file."""
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def get_output_dir(config=None, config_path=DEFAULT_CONFIG_PATH, create=False):
    """
    Resolve the run-output directory. Precedence: PCG_OUTPUT_DIR env var, then
    `output_dir` in the config, then output/smb.
    """
    if config is None:
        config = load_config(config_path)

    output_dir = os.environ.get("PCG_OUTPUT_DIR") or config.get("output_dir", DEFAULT_OUTPUT_DIR)
    if create:
        os.makedirs(output_dir, exist_ok=True)
    return output_dir


def output_path(*parts, config=None, config_path=DEFAULT_CONFIG_PATH, create=False):
    """Build a path inside the run-output directory."""
    return os.path.join(get_output_dir(config, config_path, create), *parts)
