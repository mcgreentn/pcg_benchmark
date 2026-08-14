import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import load_config, get_output_dir
from measures import load_measures, describe


def _bin_index(values, measure):
    """Bin measure values the way pyribs does -- linear bins, out-of-range clipped."""
    width = (measure.high - measure.low) / measure.bins
    idx = np.floor((np.asarray(values, dtype=float) - measure.low) / width)
    return np.clip(idx, 0, measure.bins - 1).astype(int)


def project(df, measures, x_key, y_key):
    """
    Collapse an n-dimensional archive onto two axes.

    Cells are keyed by (x_bin, y_bin) and hold the best objective found anywhere
    along the axes being collapsed -- the standard way to read a high-dimensional
    MAP-Elites archive without pretending the other axes don't exist.
    """
    keys = [m.key for m in measures]
    x_measure = measures[keys.index(x_key)]
    y_measure = measures[keys.index(y_key)]

    xi = _bin_index(df[f"measures_{keys.index(x_key)}"], x_measure)
    yi = _bin_index(df[f"measures_{keys.index(y_key)}"], y_measure)
    objective = df["objective"].values

    grid = np.full((y_measure.bins, x_measure.bins), np.nan)
    for x, y, obj in zip(xi, yi, objective):
        if np.isnan(grid[y, x]) or obj > grid[y, x]:
            grid[y, x] = obj

    return grid, x_measure, y_measure


def analyze(archive_path=None, output_path=None, config_path="exp_config.yaml",
            x_key=None, y_key=None, show=True):
    config = load_config(config_path)
    measures = load_measures(config)
    output_dir = get_output_dir(config, config_path=config_path)

    if archive_path is None:
        archive_path = os.path.join(output_dir, "archive_final.csv")
    if output_path is None:
        output_path = os.path.join(output_dir, "archive_final.png")

    keys = [m.key for m in measures]
    x_key = x_key or keys[0]
    y_key = y_key or keys[1]
    for key, flag in [(x_key, "--x"), (y_key, "--y")]:
        if key not in keys:
            raise SystemExit(f"{flag}: unknown measure '{key}'. Archive axes are: {', '.join(keys)}")

    print(f"Analyzing archive from: {archive_path}")
    df = pd.read_csv(archive_path, index_col=0)
    n_measures = len([c for c in df.columns if c.startswith("measures_")])
    if n_measures != len(measures):
        raise SystemExit(
            f"Archive has {n_measures} measures but the config defines {len(measures)} "
            f"({', '.join(keys)}). Point --config at the config used for this run."
        )

    total_cells = int(np.prod([m.bins for m in measures]))
    print(f"  {describe(measures)}")
    print(f"  {len(df)} elites, {len(df) / total_cells:.1%} coverage, "
          f"best objective {df['objective'].max():.4f}")

    grid, x_measure, y_measure = project(df, measures, x_key, y_key)
    collapsed = [k for k in keys if k not in (x_key, y_key)]
    if collapsed:
        print(f"  projected onto {x_key} x {y_key}; best over collapsed {', '.join(collapsed)}")

    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad(color="none")

    fig, ax = plt.subplots(figsize=(8, 6))
    mesh = ax.imshow(
        np.ma.masked_invalid(grid),
        origin="lower",
        aspect="auto",
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        extent=[x_measure.low, x_measure.high, y_measure.low, y_measure.high],
    )
    ax.set_xticks(np.linspace(x_measure.low, x_measure.high, x_measure.bins + 1))
    ax.set_yticks(np.linspace(y_measure.low, y_measure.high, y_measure.bins + 1))
    ax.grid(color="0.7", linewidth=0.5, alpha=0.5)
    ax.set_xlabel(x_measure.label)
    ax.set_ylabel(y_measure.label)

    title = "Best completion per behavior cell"
    if collapsed:
        title += f"  (max over {', '.join(collapsed)})"
    ax.set_title(title)
    fig.colorbar(mesh, ax=ax, label="Completion %")
    fig.tight_layout()

    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    print(f"Saved heatmap to {output_path}")
    if show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", default=None, help="Archive CSV produced by driver.py (default: <output_dir>/archive_final.csv)")
    parser.add_argument("--output", default=None, help="Where to write the heatmap (default: <output_dir>/archive_final.png)")
    parser.add_argument("--config", default="exp_config.yaml", help="Config used for the run, for the archive geometry")
    parser.add_argument("--x", default=None, help="Measure to place on the x axis (default: first)")
    parser.add_argument("--y", default=None, help="Measure to place on the y axis (default: second)")
    parser.add_argument("--no-show", action="store_true", help="Save the figure without opening a window")
    args = parser.parse_args()
    analyze(args.archive, args.output, args.config, args.x, args.y, show=not args.no_show)
