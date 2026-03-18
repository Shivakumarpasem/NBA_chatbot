import sys
from pathlib import Path

from nba_data import (
    load_adj_shooting,
    player_3pt_by_season,
    top_3pt_pct,
    top_stat_leaderboard,
    player_summary,
    compare_players,
)
from call_tools import safe_call

ROOT = Path(__file__).resolve().parents[1]  # points to project root from /src
PATH = ROOT / "data" / "player_stats" / "totals_stats.csv"


def main():
    # Avoid Windows console encoding errors when printing player names (e.g. Dončić)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    df = load_adj_shooting(str(PATH))

    out = safe_call(player_3pt_by_season, df, "curryst01")
    print("\nCurry by season (3PT):")
    print(out["data"].tail(20) if out["ok"] else f"Error: {out['error']}")

    lb = safe_call(top_3pt_pct, df, season="2023-2024", min_g=40, min_mp=800, min_3pa=200, limit=10)
    print("\nTop 3P% leaderboard (2023-2024):")
    print(lb["data"] if lb["ok"] else f"Error: {lb['error']}")

    pts_lb = safe_call(top_stat_leaderboard, df, "2023-2024", "PTS", 40, 800, 10)
    print("\nTop PTS leaderboard (2023-2024):")
    print(pts_lb["data"] if pts_lb["ok"] else f"Error: {pts_lb['error']}")

    summary = safe_call(player_summary, df, "curryst01", "2023-2024")
    print("\nCurry 2023-2024 summary:")
    print(summary["data"] if summary["ok"] else f"Error: {summary['error']}")

    comp = safe_call(compare_players, df, ["curryst01", "jamesle01"], "2023-2024")
    print("\nCurry vs LeBron 2023-2024:")
    print(comp["data"] if comp["ok"] else f"Error: {comp['error']}")

    misses_lb = safe_call(top_stat_leaderboard, df, "2023-2024", "Misses", 40, 800, 10)
    print("\nTop 10 most FG misses (2023-2024):")
    print(misses_lb["data"] if misses_lb["ok"] else f"Error: {misses_lb['error']}")


if __name__ == "__main__":
    main()
