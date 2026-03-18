"""Test get_last_n_games: load box scores and show last N games for a player."""
import sys
from pathlib import Path

from nba_data import load_box_scores, get_last_n_games

ROOT = Path(__file__).resolve().parents[1]
BOX_PATH = ROOT / "data" / "boxscores_by_year" / "NBA_2023-2024_basic.csv"


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    df_games = load_box_scores(str(BOX_PATH))
    print("Box scores loaded, shape:", df_games.shape)

    last = get_last_n_games(df_games, "curryst01", n=10)
    print("\nCurry last 10 games (2023-2024 season):")
    print(last)


if __name__ == "__main__":
    main()
