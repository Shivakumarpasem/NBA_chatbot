import pandas as pd
from typing import List, Optional, Set


def _require_columns(df: pd.DataFrame, required: Set[str], func_name: str) -> None:
    """
    Raise ValueError if any required columns are missing. Use before operating on df.

    Inputs:
        df: DataFrame to check.
        required: set of column names that must exist in df.
        func_name: name of the calling function (for error message).

    Outputs:
        None. Raises ValueError if any column in required is missing from df.
    """
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{func_name}: missing columns: {sorted(missing)}")


def load_adj_shooting(path: str) -> pd.DataFrame:
    """
    Load totals_stats.csv as the single source of truth. Cleans columns, filters NBA, renames Tm->Team.

    Inputs:
        path: full path to the CSV file (e.g. totals_stats.csv).

    Outputs:
        DataFrame with columns: Player, Player Reference, Team, Season, G, MP, 3P, 3PA, 3P%,
        plus PTS, AST, TRB, STL, BLK, FG, FGA, Misses (FGA-FG) if present. All string types for Player, Team, Season, Player Reference.
    """
    df = pd.read_csv(path)

    # Clean column names
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]

    # Filter NBA if League exists
    if "League" in df.columns:
        df = df[df["League"] == "NBA"].copy()

    # Normalize team column name
    if "Tm" in df.columns and "Team" not in df.columns:
        df = df.rename(columns={"Tm": "Team"})

    # Ensure key columns exist
    required = {
        "Player", "Team", "G", "MP",
        "3P", "3PA", "3P%",
        "Player Reference", "Season"
    }
    _require_columns(df, required, "load_adj_shooting")

    # Keep required columns plus optional stat columns (for leaderboards)
    keep_cols = list(required)
    for col in ("PTS", "AST", "TRB", "STL", "BLK", "FG", "FGA"):
        if col in df.columns and col not in keep_cols:
            keep_cols.append(col)
    df = df[keep_cols].copy()

    # Compute FG misses (FGA - FG) for "most misses" leaderboard
    if "FG" in df.columns and "FGA" in df.columns:
        df["Misses"] = df["FGA"].astype(float) - df["FG"].astype(float)

    # Normalize types
    df["Season"] = df["Season"].astype(str)
    df["Player Reference"] = df["Player Reference"].astype(str)
    df["Player"] = df["Player"].astype(str)
    df["Team"] = df["Team"].astype(str)

    return df


def get_player_debut(df: pd.DataFrame, player_id: str) -> Optional[str]:
    """Earliest season for a player (debut year). Returns e.g. '2009-2010' or None."""
    sub = df[df["Player Reference"] == player_id]["Season"]
    if sub.empty:
        return None
    return sub.min()


def find_players(df: pd.DataFrame, query: str, limit: int = 10) -> pd.DataFrame:
    """
    Search players by name (case-insensitive). Returns one row per unique player.
    Prioritizes: 1) Exact last name match, 2) More career games (established players).

    Inputs:
        df: from load_adj_shooting.
        query: search string (e.g. "curry", "kevin durant").
        limit: max number of players to return (default 10).

    Outputs:
        DataFrame with columns: Player, Player Reference, Season, Team. Sorted by relevance.
    """
    q = query.strip().lower()

    sub = df[df["Player"].str.lower().str.contains(q, na=False)].copy()
    
    if sub.empty:
        return sub[["Player", "Player Reference", "Season", "Team"]].head(limit)
    
    # Calculate career games per player (to prioritize established players)
    career_games = sub.groupby("Player Reference")["G"].sum().reset_index()
    career_games.columns = ["Player Reference", "CareerGames"]
    
    # Get most recent season per player
    sub = sub.sort_values("Season", ascending=False)
    sub = sub.drop_duplicates(subset=["Player Reference"], keep="first")
    
    # Merge career games
    sub = sub.merge(career_games, on="Player Reference", how="left")
    
    # Score: prefer recent players + exact name match
    def match_score(row):
        name = row["Player"].lower()
        season = row.get("Season", "1900-1901")
        
        # Base score from recency (most important - prefer active players)
        try:
            season_year = int(season.split("-")[0])
        except:
            season_year = 1900
        recency_score = (season_year - 1900) * 100  # Recent seasons get huge bonus
        
        # Name match bonus
        name_score = 0
        name_parts = name.split()
        
        # Exact last name match
        if name_parts and name_parts[-1] == q:
            name_score += 5000
        # Exact first name match  
        if name_parts and name_parts[0] == q:
            name_score += 3000
        # Full name exact match (highest priority)
        if name == q:
            name_score += 10000
        # Query matches full name closely (e.g., "stephen curry" matches "Stephen Curry")
        if q in name:
            name_score += 1000
            
        return recency_score + name_score
    
    sub["_score"] = sub.apply(match_score, axis=1)
    sub = sub.sort_values("_score", ascending=False)
    
    return sub[["Player", "Player Reference", "Season", "Team"]].head(limit)


def player_3pt_by_season(
    df: pd.DataFrame,
    player_id: str,
    season_from: Optional[str] = None,
    season_to: Optional[str] = None,
) -> pd.DataFrame:
    """
    3PT stats by season for one player. Optional season range.

    Inputs:
        df: from load_adj_shooting.
        player_id: Player Reference (e.g. "curryst01").
        season_from: optional start season (inclusive), e.g. "2020-2021".
        season_to: optional end season (inclusive), e.g. "2023-2024".

    Outputs:
        DataFrame with columns: Season, Team, G, MP, 3P, 3PA, 3P%. Sorted by Season. One row per team/season.
    """
    _require_columns(df, {"Season", "Team", "G", "MP", "3P", "3PA", "3P%", "Player Reference"}, "player_3pt_by_season")

    out = df[df["Player Reference"] == player_id].copy()

    if season_from is not None:
        out = out[out["Season"] >= season_from]
    if season_to is not None:
        out = out[out["Season"] <= season_to]

    out = out[["Season", "Team", "G", "MP", "3P", "3PA", "3P%"]].sort_values("Season")
    return out.reset_index(drop=True)


def player_stats_multi_season(
    df: pd.DataFrame,
    player_id: str,
    season_from: Optional[str] = None,
    season_to: Optional[str] = None,
    n_seasons: Optional[int] = None,
) -> pd.DataFrame:
    """
    Get general stats for a player across multiple seasons.
    Use for queries like "last 3 years", "career stats", "2015-2018 stats".
    
    Inputs:
        df: totals_stats DataFrame
        player_id: Player Reference (e.g. "novakst01")
        season_from: optional start season (e.g. "2014-2015")
        season_to: optional end season (e.g. "2016-2017")
        n_seasons: if provided, get the last N seasons (overrides season_from/to)
    
    Outputs:
        DataFrame with columns: Season, Team, G, MP, PTS, AST, TRB, STL, BLK, 3P, 3PA, 3P%
        Sorted by Season (most recent first if n_seasons used, chronological otherwise)
    """
    _require_columns(df, {"Season", "Team", "G", "MP", "PTS", "AST", "TRB", "STL", "BLK", "3P", "3PA", "3P%", "Player Reference"}, "player_stats_multi_season")
    
    out = df[df["Player Reference"] == player_id].copy()
    
    if out.empty:
        return pd.DataFrame()
    
    # If n_seasons specified, get the last N seasons
    if n_seasons is not None:
        out = out.sort_values("Season", ascending=False)
        out = out.head(n_seasons)
    else:
        # Use season range if provided
        if season_from is not None:
            out = out[out["Season"] >= season_from]
        if season_to is not None:
            out = out[out["Season"] <= season_to]
        out = out.sort_values("Season")
    
    # Select and return relevant columns
    cols = ["Season", "Team", "G", "MP", "PTS", "AST", "TRB", "STL", "BLK", "3P", "3PA", "3P%"]
    return out[cols].reset_index(drop=True)


def top_3pt_pct(
    df: pd.DataFrame,
    season: str,
    min_g: int = 40,
    min_mp: int = 800,
    min_3pa: int = 200,
    limit: int = 10,
) -> pd.DataFrame:
    """
    Top 3P% leaderboard for a season. Minimum filters to avoid small samples. Prefers TOT row per player.

    Inputs:
        df: from load_adj_shooting.
        season: e.g. "2023-2024".
        min_g: minimum games (default 40).
        min_mp: minimum minutes (default 800).
        min_3pa: minimum 3PA (default 200).
        limit: max rows to return (default 10).

    Outputs:
        DataFrame with columns: Player, Player Reference, Team, Season, G, MP, 3PA, 3P%. Sorted by 3P% descending.
    """
    _require_columns(df, {"Season", "Player", "Team", "G", "MP", "3PA", "3P%", "Player Reference"}, "top_3pt_pct")

    gs = df[df["Season"] == season].copy()
    gs = gs[(gs["G"] >= min_g) & (gs["MP"] >= min_mp) & (gs["3PA"] >= min_3pa)].copy()

    # Prefer TOT row if present for a player in that season
    gs["_is_tot"] = (gs["Team"] == "TOT").astype(int)
    gs = gs.sort_values(
        by=["Player Reference", "_is_tot", "MP"],
        ascending=[True, False, False],
    )
    gs = gs.drop_duplicates(subset=["Player Reference"], keep="first").copy()
    gs = gs.drop(columns=["_is_tot"], errors="ignore")

    gs = gs.sort_values(by=["3P%", "3PA"], ascending=[False, False])

    out_cols = ["Player", "Player Reference", "Team", "Season", "G", "MP", "3PA", "3P%"]
    return gs[out_cols].head(limit).reset_index(drop=True)


def top_stat_leaderboard(
    df: pd.DataFrame,
    season: str,
    stat: str,
    min_g: int = 0,
    min_mp: int = 0,
    limit: int = 10,
) -> pd.DataFrame:
    """
    Generic leaderboard for a numeric stat (e.g. PTS, AST, TRB, 3PA).
    All computation in Python. Prefers TOT row per player when present.

    Inputs:
        df: from load_adj_shooting (must contain Season, G, MP, Player, Team, Player Reference, and stat).
        season: e.g. "2023-2024".
        stat: column name for the stat to rank by (e.g. "PTS", "AST", "TRB").
        min_g: minimum games (default 0).
        min_mp: minimum minutes (default 0).
        limit: max rows to return (default 10).

    Outputs:
        DataFrame with columns: Player, Player Reference, Team, Season, G, MP, <stat>, sorted by stat descending.
    """
    base_required = {"Season", "Player", "Team", "G", "MP", "Player Reference"}
    _require_columns(df, base_required, "top_stat_leaderboard")
    if stat not in df.columns:
        raise ValueError(f"top_stat_leaderboard: stat column '{stat}' not in dataframe")
    if not pd.api.types.is_numeric_dtype(df[stat]):
        raise ValueError(f"top_stat_leaderboard: column '{stat}' is not numeric")

    gs = df[df["Season"] == season].copy()
    gs = gs[(gs["G"] >= min_g) & (gs["MP"] >= min_mp)]

    # Prefer TOT row if present for a player in that season
    gs["_is_tot"] = (gs["Team"] == "TOT").astype(int)
    gs = gs.sort_values(
        by=["Player Reference", "_is_tot", "MP"],
        ascending=[True, False, False],
    )
    gs = gs.drop_duplicates(subset=["Player Reference"], keep="first").copy()
    gs = gs.drop(columns=["_is_tot"], errors="ignore")

    gs = gs.sort_values(by=stat, ascending=False)
    out_cols = ["Player", "Player Reference", "Team", "Season", "G", "MP", stat]
    return gs[out_cols].head(limit).reset_index(drop=True)


def player_summary(
    df: pd.DataFrame,
    player_id: str,
    season: str,
) -> pd.DataFrame:
    """
    One row of key stats for a player in a given season. Prefers TOT row if present.

    Inputs:
        df: from load_adj_shooting.
        player_id: Player Reference (e.g. "curryst01").
        season: e.g. "2023-2024".

    Outputs:
        DataFrame with one row (or empty if not found): Player, Player Reference, Team, Season, G, MP,
        plus 3P, 3PA, 3P%, PTS, AST, TRB, STL, BLK (only columns that exist in df).
    """
    _require_columns(df, {"Season", "Player", "Team", "G", "MP", "Player Reference"}, "player_summary")

    sub = df[(df["Player Reference"] == player_id) & (df["Season"] == season)].copy()
    if sub.empty:
        return sub.reset_index(drop=True)

    # Prefer TOT row if present
    sub["_is_tot"] = (sub["Team"] == "TOT").astype(int)
    sub = sub.sort_values(by=["_is_tot", "MP"], ascending=[False, False])
    row = sub.head(1).drop(columns=["_is_tot"], errors="ignore")

    # Return key stat columns that exist
    key_cols = ["Player", "Player Reference", "Team", "Season", "G", "MP", "3P", "3PA", "3P%", "PTS", "AST", "TRB", "STL", "BLK"]
    out_cols = [c for c in key_cols if c in row.columns]
    return row[out_cols].reset_index(drop=True)


def compare_players(
    df: pd.DataFrame,
    player_ids: List[str],
    season: str,
) -> pd.DataFrame:
    """
    Side-by-side key stats for multiple players in one season. One row per player.
    Prefers TOT row per player when present.

    Inputs:
        df: from load_adj_shooting.
        player_ids: list of Player Reference IDs (e.g. ["curryst01", "jamesle01"]).
        season: e.g. "2023-2024".

    Outputs:
        DataFrame with one row per player: same key columns as player_summary.
    """
    _require_columns(df, {"Season", "Player", "Team", "G", "MP", "Player Reference"}, "compare_players")

    key_cols = ["Player", "Player Reference", "Team", "Season", "G", "MP", "3P", "3PA", "3P%", "PTS", "AST", "TRB", "STL", "BLK"]
    rows = []

    for player_id in player_ids:
        sub = df[(df["Player Reference"] == player_id) & (df["Season"] == season)].copy()
        if sub.empty:
            continue
        sub["_is_tot"] = (sub["Team"] == "TOT").astype(int)
        sub = sub.sort_values(by=["_is_tot", "MP"], ascending=[False, False])
        one = sub.head(1).drop(columns=["_is_tot"], errors="ignore")
        out_cols = [c for c in key_cols if c in one.columns]
        rows.append(one[out_cols])

    if not rows:
        return pd.DataFrame()
    result = pd.concat(rows, ignore_index=True)
    out_cols = [c for c in key_cols if c in result.columns]
    return result[out_cols].reset_index(drop=True)


def compare_careers(
    df: pd.DataFrame,
    player_ids: List[str],
) -> pd.DataFrame:
    """
    Career totals comparison for multiple players (all seasons combined).
    One row per player with career G, MP, PTS, AST, TRB, STL, BLK, 3P, 3PA, 3P%.
    Prefers TOT row per season to avoid double-counting traded players.

    Inputs:
        df: from load_adj_shooting.
        player_ids: list of Player Reference IDs (e.g. ["duranke01", "bryanko01"]).

    Outputs:
        DataFrame with one row per player: Player, Player Reference, G, MP, PTS, AST, TRB, STL, BLK, 3P, 3PA, 3P%.
    """
    _require_columns(df, {"Player", "Player Reference", "Season", "Team", "G", "MP", "Player Reference"}, "compare_careers")
    sum_cols = ["G", "MP", "3P", "3PA", "PTS", "AST", "TRB", "STL", "BLK"]
    sum_cols = [c for c in sum_cols if c in df.columns]
    rows = []
    for player_id in player_ids:
        sub = df[df["Player Reference"] == player_id].copy()
        if sub.empty:
            continue
        sub["_is_tot"] = (sub["Team"] == "TOT").astype(int)
        sub = sub.sort_values(by=["Season", "_is_tot", "MP"], ascending=[True, False, False])
        sub = sub.drop_duplicates(subset=["Season"], keep="first")
        agg = sub[sum_cols].sum()
        pct = sub["3P"].sum() / sub["3PA"].sum() if sub["3PA"].sum() > 0 else 0
        row = pd.DataFrame({
            "Player": [sub["Player"].iloc[0]],
            "Player Reference": [player_id],
            **{c: [agg[c]] for c in sum_cols},
            "3P%": [pct],
        })
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    result = pd.concat(rows, ignore_index=True)
    out_cols = ["Player", "Player Reference"] + sum_cols + ["3P%"]
    return result[[c for c in out_cols if c in result.columns]].reset_index(drop=True)


# Team name/abbrev for schedule queries
TEAM_ABBREVS = {
    "gsw": "GSW", "golden state": "GSW", "golden state warriors": "GSW", "warriors": "GSW",
    "lal": "LAL", "lakers": "LAL", "los angeles lakers": "LAL",
    "bos": "BOS", "celtics": "BOS", "boston": "BOS",
    "dal": "DAL", "mavs": "DAL", "mavericks": "DAL",
    "den": "DEN", "nuggets": "DEN", "denver": "DEN",
    "mia": "MIA", "heat": "MIA", "miami": "MIA",
    "phx": "PHO", "pho": "PHO", "suns": "PHO", "phoenix": "PHO",
    "cle": "CLE", "cavaliers": "CLE", "cavs": "CLE", "cleveland": "CLE",
    "nyk": "NYK", "knicks": "NYK", "new york": "NYK",
    "mil": "MIL", "bucks": "MIL", "milwaukee": "MIL",
    "sas": "SAS", "spurs": "SAS", "san antonio": "SAS",
    "okc": "OKC", "thunder": "OKC", "oklahoma city": "OKC",
    "hou": "HOU", "rockets": "HOU", "houston": "HOU",
    "lac": "LAC", "clippers": "LAC", "la clippers": "LAC",
    "min": "MIN", "timberwolves": "MIN", "wolves": "MIN", "minnesota": "MIN",
    "sac": "SAC", "kings": "SAC", "sacramento": "SAC",
    "por": "POR", "blazers": "POR", "portland": "POR",
    "atl": "ATL", "hawks": "ATL", "atlanta": "ATL",
    "chi": "CHI", "bulls": "CHI", "chicago": "CHI",
    "ind": "IND", "pacers": "IND", "indiana": "IND",
    "mem": "MEM", "grizzlies": "MEM", "memphis": "MEM",
    "nop": "NOP", "pelicans": "NOP", "new orleans": "NOP",
    "orl": "ORL", "magic": "ORL", "orlando": "ORL",
    "phi": "PHI", "76ers": "PHI", "sixers": "PHI", "philadelphia": "PHI",
    "tor": "TOR", "raptors": "TOR", "toronto": "TOR",
    "uta": "UTA", "jazz": "UTA", "utah": "UTA",
    "was": "WAS", "wizards": "WAS", "washington": "WAS",
    "det": "DET", "pistons": "DET", "detroit": "DET",
    "cha": "CHO", "cho": "CHO", "hornets": "CHO", "charlotte": "CHO",
    "brk": "BRK", "bkn": "BRK", "nets": "BRK", "brooklyn": "BRK",
}


def load_schedule(path: str) -> pd.DataFrame:
    """Load schedule CSV. Columns: Date, Visitor, Home, Season, etc."""
    df = pd.read_csv(path)
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df.dropna(subset=["Date"]).sort_values("Date", ascending=False)


def get_team_games(
    df_sched: pd.DataFrame,
    team_abbrev: str,
    n: int = 5,
) -> pd.DataFrame:
    """
    Next/most recent n games for a team. Schedule sorted by Date desc (most recent first).
    """
    mask = (df_sched["Visitor"] == team_abbrev) | (df_sched["Home"] == team_abbrev)
    sub = df_sched[mask].head(n)
    cols = ["Date", "Visitor", "Home", "Visitor PTS", "Home PTS", "Arena"]
    return sub[[c for c in cols if c in sub.columns]].reset_index(drop=True)


def team_roster_stats(
    df: pd.DataFrame,
    team_abbrev: str,
    seasons: List[str],
    stats_mode: str = "all"
) -> pd.DataFrame:
    """
    Get roster stats for a team across one or more seasons.
    Returns all players who played for the team in those seasons.
    
    Inputs:
        df: totals_stats DataFrame
        team_abbrev: Team abbreviation (e.g. "BOS", "LAL")
        seasons: List of seasons (e.g. ["2022-2023", "2023-2024"])
        stats_mode: "basic" (12 key stats) or "all" (all available columns)
    
    Outputs:
        DataFrame with Player, Season, and stats columns.
        One row per player per season.
    """
    _require_columns(df, {"Player", "Team", "Season", "G", "MP", "PTS"}, "team_roster_stats")
    
    # Filter for team and seasons
    mask = (df["Team"] == team_abbrev) & (df["Season"].isin(seasons))
    result = df[mask].copy()
    
    if result.empty:
        return pd.DataFrame()
    
    # Define column sets
    basic_cols = [
        "Player", "Player Reference", "Season", "Team", 
        "G", "GS", "MP", "PTS", "AST", "TRB", "STL", "BLK", 
        "FG%", "3P%", "FT%"
    ]
    
    if stats_mode == "basic":
        # Keep only basic columns that exist
        cols_to_keep = [c for c in basic_cols if c in result.columns]
        result = result[cols_to_keep]
    else:
        # Keep all columns (already filtered by team/season)
        pass
    
    # Sort by Season (desc) then by MP (desc) to show key players first
    if "MP" in result.columns:
        result = result.sort_values(["Season", "MP"], ascending=[False, False])
    else:
        result = result.sort_values("Season", ascending=False)
    
    return result.reset_index(drop=True)


def load_box_scores(path: str) -> pd.DataFrame:
    """
    Load one box score CSV (e.g. from boxscores_by_year/NBA_2023-2024_basic.csv).
    One row per player per game. Use with get_last_n_games for game-level stats.

    Inputs:
        path: full path to the box score CSV file.

    Outputs:
        DataFrame with columns: Game Reference, Team, Player Name, Player Reference, MP, PTS, FG, FGA, 3P, 3PA, AST, TRB, etc.
        Game Reference is sortable (YYYYMMDD + suffix) for chronological order.
    """
    df = pd.read_csv(path)
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
    required = {"Game Reference", "Player Reference"}
    _require_columns(df, required, "load_box_scores")
    if "Period" in df.columns:
        df = df[df["Period"] == "game"].copy()
    df["Game Reference"] = df["Game Reference"].astype(str)
    df["Player Reference"] = df["Player Reference"].astype(str)
    return df.reset_index(drop=True)


def get_last_n_games(
    df_games: pd.DataFrame,
    player_id: str,
    n: int = 10,
) -> pd.DataFrame:
    """
    Last N games for a player (most recent first). All computation in Python.

    Inputs:
        df_games: from load_box_scores (one or more seasons concatenated). Must have Game Reference, Player Reference.
        player_id: Player Reference (e.g. "curryst01").
        n: number of most recent games to return (default 10).

    Outputs:
        DataFrame with one row per game: Game Reference, Player Name (if present), Team, MP, PTS, FG, FGA, 3P, 3PA, AST, TRB, STL, BLK, etc. Sorted by Game Reference descending (most recent first).
    """
    _require_columns(df_games, {"Game Reference", "Player Reference"}, "get_last_n_games")

    sub = df_games[df_games["Player Reference"] == player_id].copy()
    if sub.empty:
        return sub.reset_index(drop=True)

    sub = sub.sort_values(by="Game Reference", ascending=False).head(n)
    key_cols = [
        "Game Reference", "Player Name", "Player Reference", "Team",
        "MP", "PTS", "FG", "FGA", "3P", "3PA", "AST", "TRB", "STL", "BLK",
    ]
    out_cols = [c for c in key_cols if c in sub.columns]
    return sub[out_cols].reset_index(drop=True)
