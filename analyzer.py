import pandas as pd
from ribs.archives import ArchiveDataFrame
import matplotlib.pyplot as plt
from ribs.visualize import grid_archive_heatmap
from ribs.archives import GridArchive

def analyze(archive_path):
    print("Analyzing archive from: ", archive_path)
    df = ArchiveDataFrame(pd.read_csv(archive_path))
    print("Have DF")
    print(df.get_field("objective")[0])
    plt.figure(figsize=(8, 6))

    # this archive is fake and only exists so that the heatmap function doesnt freak out
    fake_archive = GridArchive(
            solution_dim=10,
            dims=[1, 1],
            ranges=[(0.0, 1), (0.0, 1)],
            qd_score_offset=-600,
        )
    # call this out to Matt, why on earth would you use parameters this way?
    # archive is IGNORED if df is used, but you still call functions with archive, which causes None to fail
    # this means the user MUST create a fake archive with no intention of using it, 
    # when doing analysis not in a jupyter notebok imediately after a run

    grid_archive_heatmap(fake_archive, df=df, vmin=-0, vmax=1)
    plt.ylabel("Collected Coins")
    plt.xlabel("Enemy Squishes")
    plt.savefig("./data/smb/archive_final.png")
    plt.show()

if __name__ == "__main__":
    archive_path = "data/smb/archive_final.csv"
    analyze(archive_path)