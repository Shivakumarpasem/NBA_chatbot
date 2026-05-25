"""
src/nba_api_client.py
Official NBA.com data client using the nba_api package.
Covers every data type: team stats, game logs, draft, bios, advanced metrics.
All free, official data from NBA.com (1946 to present).
"""
import time
import re
from datetime import datetime
from typing import Optional, Dict, List, Tuple
import pandas as pd

NBA_API_SLEEP = 0.7  # seconds between calls to avoid NBA.com rate limiting

# ── Player nicknames ──────────────────────────────────────────────────────────
PLAYER_NICKNAMES = {
    # Modern stars
    "greek freak": "Giannis Antetokounmpo",
    "giannis": "Giannis Antetokounmpo",
    "the greek freak": "Giannis Antetokounmpo",
    "king james": "LeBron James",
    "lbj": "LeBron James",
    "the king": "LeBron James",
    "bron": "LeBron James",
    "lebron": "LeBron James",
    "the beard": "James Harden",
    "harden": "James Harden",
    "the chef": "Stephen Curry",
    "chef curry": "Stephen Curry",
    "steph": "Stephen Curry",
    "steph curry": "Stephen Curry",
    "curry": "Stephen Curry",
    "kd": "Kevin Durant",
    "slim reaper": "Kevin Durant",
    "the slim reaper": "Kevin Durant",
    "the durantula": "Kevin Durant",
    "durant": "Kevin Durant",
    "dame": "Damian Lillard",
    "dame time": "Damian Lillard",
    "lillard": "Damian Lillard",
    "luka": "Luka Doncic",
    "luka magic": "Luka Doncic",
    "doncic": "Luka Doncic",
    "joker": "Nikola Jokic",
    "the joker": "Nikola Jokic",
    "jokic": "Nikola Jokic",
    "nikola": "Nikola Jokic",
    "ant": "Anthony Edwards",
    "ant-man": "Anthony Edwards",
    "antman": "Anthony Edwards",
    "ant man": "Anthony Edwards",
    "book": "Devin Booker",
    "booker": "Devin Booker",
    "ja": "Ja Morant",
    "morant": "Ja Morant",
    "bam": "Bam Adebayo",
    "trae": "Trae Young",
    "ice trae": "Trae Young",
    "shai": "Shai Gilgeous-Alexander",
    "sga": "Shai Gilgeous-Alexander",
    "kawhi": "Kawhi Leonard",
    "the klaw": "Kawhi Leonard",
    "klaw": "Kawhi Leonard",
    "wemby": "Victor Wembanyama",
    "wembanyama": "Victor Wembanyama",
    "zion": "Zion Williamson",
    "chet": "Chet Holmgren",
    "holmgren": "Chet Holmgren",
    "paulo": "Paolo Banchero",
    "scottie": "Scottie Barnes",
    "lauri": "Lauri Markkanen",
    "spida": "Donovan Mitchell",
    "jt": "Jayson Tatum",
    "tatum": "Jayson Tatum",
    "jb": "Jaylen Brown",
    "kyrie": "Kyrie Irving",
    "uncle drew": "Kyrie Irving",
    "irving": "Kyrie Irving",
    "pg13": "Paul George",
    "the point god": "Chris Paul",
    "cp3": "Chris Paul",
    "klay": "Klay Thompson",
    "splash brother": "Klay Thompson",
    "draymond": "Draymond Green",
    "russ": "Russell Westbrook",
    "westbrook": "Russell Westbrook",
    "brodie": "Russell Westbrook",
    "d rose": "Derrick Rose",
    "melo": "Carmelo Anthony",
    "carmelo": "Carmelo Anthony",
    "d wade": "Dwyane Wade",
    "flash": "Dwyane Wade",
    "wade": "Dwyane Wade",
    "kemba": "Kemba Walker",
    # Legends
    "mj": "Michael Jordan",
    "air jordan": "Michael Jordan",
    "his airness": "Michael Jordan",
    "jordan": "Michael Jordan",
    "kobe": "Kobe Bryant",
    "black mamba": "Kobe Bryant",
    "mamba": "Kobe Bryant",
    "bean": "Kobe Bryant",
    "shaq": "Shaquille O'Neal",
    "diesel": "Shaquille O'Neal",
    "shaquille": "Shaquille O'Neal",
    "magic": "Magic Johnson",
    "the logo": "Jerry West",
    "larry legend": "Larry Bird",
    "bird": "Larry Bird",
    "penny": "Anfernee Hardaway",
    "the mailman": "Karl Malone",
    "malone": "Karl Malone",
    "the glove": "Gary Payton",
    "payton": "Gary Payton",
    "admiral": "David Robinson",
    "the admiral": "David Robinson",
    "the answer": "Allen Iverson",
    "ai": "Allen Iverson",
    "iverson": "Allen Iverson",
    "tmac": "Tracy McGrady",
    "vinsanity": "Vince Carter",
    "half man half amazing": "Vince Carter",
    "the big fundamental": "Tim Duncan",
    "timmy d": "Tim Duncan",
    "duncan": "Tim Duncan",
    "dirk": "Dirk Nowitzki",
    "german wunderkind": "Dirk Nowitzki",
    "nowitzki": "Dirk Nowitzki",
    "sir charles": "Charles Barkley",
    "barkley": "Charles Barkley",
    "the round mound of rebound": "Charles Barkley",
    "hakeem": "Hakeem Olajuwon",
    "the dream": "Hakeem Olajuwon",
    "clyde the glide": "Clyde Drexler",
    "pippen": "Scottie Pippen",
    "pip": "Scottie Pippen",
    "rodman": "Dennis Rodman",
    "the worm": "Dennis Rodman",
    "reggie": "Reggie Miller",
    "stockton": "John Stockton",
    "wilt": "Wilt Chamberlain",
    "the stilt": "Wilt Chamberlain",
    "chamberlain": "Wilt Chamberlain",
    "the big o": "Oscar Robertson",
    "kareem": "Kareem Abdul-Jabbar",
    "cap": "Kareem Abdul-Jabbar",
    "jabbar": "Kareem Abdul-Jabbar",
    "dr j": "Julius Erving",
    "pistol pete": "Pete Maravich",
    "maravich": "Pete Maravich",
    "zeke": "Isiah Thomas",
    "ewing": "Patrick Ewing",
}

# NBA abbreviation normalization (some teams use different abbrevs)
ABBREV_NORMALIZE = {
    "BRK": "BKN", "CHO": "CHA", "PHO": "PHX", "NJN": "BKN",
    "NOH": "NOP", "NOK": "NOP", "SEA": "OKC", "VAN": "MEM",
}


def _sleep():
    time.sleep(NBA_API_SLEEP)


def nba_season_str(year_start: int) -> str:
    """Convert start year to NBA format: 2024 → '2024-25'"""
    return f"{year_start}-{str(year_start + 1)[-2:]}"


def current_season_year() -> int:
    """Returns start year of current NBA season (e.g. 2025 for 2025-26)."""
    now = datetime.now()
    return now.year if now.month >= 10 else now.year - 1


def parse_season(s) -> str:
    """Convert any season format to nba_api format '2024-25'."""
    if not s:
        return nba_season_str(current_season_year())
    s = str(s).strip()
    # Already correct: "2024-25"
    if re.match(r'^\d{4}-\d{2}$', s):
        return s
    # Full: "2023-2024" → "2023-24"
    m = re.match(r'^(\d{4})-(\d{4})$', s)
    if m:
        return f"{m.group(1)}-{m.group(2)[-2:]}"
    # End year only: "2024" → "2023-24"
    m = re.match(r'^(\d{4})$', s)
    if m:
        year = int(m.group(1))
        return f"{year - 1}-{str(year)[-2:]}"
    return nba_season_str(current_season_year())


def resolve_player_name(name: str) -> str:
    """Resolve nickname to canonical player name."""
    lower = name.lower().strip()
    return PLAYER_NICKNAMES.get(lower, name)


def find_player_id(name: str) -> Optional[Tuple[int, str]]:
    """
    Find NBA player ID and full name by name or nickname.
    Returns (player_id, full_name) or None.
    """
    try:
        from nba_api.stats.static import players as nba_players
        resolved = resolve_player_name(name)
        search = resolved.lower().strip()
        all_players = nba_players.get_players()

        # Exact match first
        for p in all_players:
            if p["full_name"].lower() == search:
                return (p["id"], p["full_name"])

        candidates = []
        search_parts = search.split()

        for p in all_players:
            full = p["full_name"].lower()
            full_parts = full.split()
            score = 0

            if search in full:
                score += 500
            if len(search_parts) >= 2 and all(part in full_parts for part in search_parts):
                score += 800
            elif len(search_parts) == 1:
                # Single word: match last name or first name
                if full_parts and full_parts[-1] == search_parts[0]:
                    score += 400
                elif full_parts and full_parts[0] == search_parts[0]:
                    score += 300
                elif any(search_parts[0] in part for part in full_parts):
                    score += 100

            if score > 0:
                # Bonus for active players
                score += 200 if p.get("is_active") else 0
                candidates.append((p, score))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[1], reverse=True)
        p = candidates[0][0]
        return (p["id"], p["full_name"])
    except Exception:
        return None


def find_team_id(team_abbrev: str) -> Optional[int]:
    """Find NBA team ID from abbreviation."""
    try:
        from nba_api.stats.static import teams as nba_teams
        abbrev = ABBREV_NORMALIZE.get(team_abbrev.upper(), team_abbrev.upper())
        for t in nba_teams.get_teams():
            if t["abbreviation"].upper() == abbrev:
                return t["id"]
        return None
    except Exception:
        return None


# ── Main data functions ───────────────────────────────────────────────────────

def get_team_season_stats(
    team_abbrev: str,
    season=None,
    stats_type: str = "base",
    per_mode: str = "PerGame",
) -> pd.DataFrame:
    """
    Get all players' stats for a team in a season.
    Perfect for downloading full team datasets for analytics projects.

    Args:
        team_abbrev: e.g. "GSW", "LAL", "BOS"
        season: any format ("2023-24", "2023-2024", "2024") or None for current
        stats_type: "base" (traditional) or "advanced" (PER/TS%/ratings)
        per_mode: "PerGame" or "Totals"
    """
    try:
        from nba_api.stats.endpoints import leaguedashplayerstats

        season_str = parse_season(season)
        team_id = find_team_id(team_abbrev)
        measure = "Advanced" if stats_type == "advanced" else "Base"

        _sleep()
        stats = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season_str,
            per_mode_detailed=per_mode,
            measure_type_detailed_defense=measure,
            season_type_all_star="Regular Season",
            team_id_nullable=team_id or "",
            timeout=30,
        )
        df = stats.get_data_frames()[0]
        if df.empty:
            return pd.DataFrame()

        if stats_type == "advanced":
            cols = {
                "PLAYER_NAME": "Player", "TEAM_ABBREVIATION": "Team", "AGE": "Age",
                "GP": "GP", "MIN": "MIN", "OFF_RATING": "OffRtg",
                "DEF_RATING": "DefRtg", "NET_RATING": "NetRtg",
                "TS_PCT": "TS%", "USG_PCT": "USG%", "AST_PCT": "AST%",
                "REB_PCT": "REB%", "EFG_PCT": "eFG%", "PIE": "PIE",
            }
        else:
            cols = {
                "PLAYER_NAME": "Player", "TEAM_ABBREVIATION": "Team", "AGE": "Age",
                "GP": "GP", "MIN": "MIN", "PTS": "PTS",
                "FGM": "FGM", "FGA": "FGA", "FG_PCT": "FG%",
                "FG3M": "3PM", "FG3A": "3PA", "FG3_PCT": "3P%",
                "FTM": "FTM", "FTA": "FTA", "FT_PCT": "FT%",
                "OREB": "OREB", "DREB": "DREB", "REB": "REB",
                "AST": "AST", "TOV": "TOV", "STL": "STL",
                "BLK": "BLK", "PF": "PF", "PLUS_MINUS": "+/-",
            }

        available = {k: v for k, v in cols.items() if k in df.columns}
        result = df[list(available.keys())].rename(columns=available)
        sort_col = "MIN" if "MIN" in result.columns else result.columns[-1]
        return result.sort_values(sort_col, ascending=False).reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def get_player_gamelog(
    player_name: str,
    season=None,
    last_n: int = None,
    season_type: str = "Regular Season",
) -> pd.DataFrame:
    """
    Get game-by-game stats for a player in a season.
    Returns each game as a row — perfect for trend analysis.
    """
    try:
        from nba_api.stats.endpoints import playergamelog

        player_info = find_player_id(player_name)
        if not player_info:
            return pd.DataFrame()
        player_id, full_name = player_info
        season_str = parse_season(season)

        _sleep()
        log = playergamelog.PlayerGameLog(
            player_id=player_id,
            season=season_str,
            season_type_all_star=season_type,
            timeout=30,
        )
        df = log.get_data_frames()[0]
        if df.empty:
            return pd.DataFrame()

        cols = {
            "GAME_DATE": "Date", "MATCHUP": "Matchup", "WL": "W/L",
            "MIN": "MIN", "PTS": "PTS", "FGM": "FGM", "FGA": "FGA",
            "FG_PCT": "FG%", "FG3M": "3PM", "FG3A": "3PA", "FG3_PCT": "3P%",
            "FTM": "FTM", "FTA": "FTA", "FT_PCT": "FT%",
            "REB": "REB", "AST": "AST", "STL": "STL", "BLK": "BLK",
            "TOV": "TOV", "PF": "PF", "PLUS_MINUS": "+/-",
        }
        available = {k: v for k, v in cols.items() if k in df.columns}
        result = df[list(available.keys())].rename(columns=available)
        result.insert(0, "Player", full_name)

        if last_n:
            result = result.head(last_n)
        return result.reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def get_draft_class(year: int, team: str = None) -> pd.DataFrame:
    """
    Get all NBA draft picks for a given year.
    Optionally filter by team. Great for historical research.
    """
    try:
        from nba_api.stats.endpoints import drafthistory

        _sleep()
        draft = drafthistory.DraftHistory(
            season_year_nullable=str(year),
            league_id="00",
            timeout=30,
        )
        df = draft.get_data_frames()[0]
        if df.empty:
            return pd.DataFrame()

        cols = {
            "OVERALL_PICK": "Pick", "ROUND_NUMBER": "Round",
            "ROUND_PICK": "Round Pick", "PLAYER_NAME": "Player",
            "TEAM_ABBREVIATION": "Team", "ORGANIZATION": "College/Team",
            "SEASON": "Draft Year",
        }
        available = {k: v for k, v in cols.items() if k in df.columns}
        result = df[list(available.keys())].rename(columns=available)

        if team and "Team" in result.columns:
            result = result[result["Team"].str.upper() == team.upper()]

        return result.sort_values("Pick").reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def get_player_bio(player_name: str) -> Optional[Dict]:
    """
    Get player biography: height, weight, position, draft info, college, nationality.
    """
    try:
        from nba_api.stats.endpoints import commonplayerinfo

        player_info = find_player_id(player_name)
        if not player_info:
            return None
        player_id, full_name = player_info

        _sleep()
        info = commonplayerinfo.CommonPlayerInfo(player_id=player_id, timeout=30)
        df = info.get_data_frames()[0]
        if df.empty:
            return None

        row = df.iloc[0]

        def g(col, default="N/A"):
            val = row.get(col, default)
            if val is None or str(val).strip() in ("", "nan", "None"):
                return default
            return str(val).strip()

        bio = {
            "name": g("DISPLAY_FIRST_LAST"),
            "position": g("POSITION"),
            "height": g("HEIGHT"),
            "weight": g("WEIGHT") + " lbs" if g("WEIGHT") != "N/A" else "N/A",
            "birthdate": g("BIRTHDATE", "")[:10],
            "country": g("COUNTRY"),
            "college": g("SCHOOL"),
            "years_pro": g("SEASON_EXP"),
            "team": g("TEAM_NAME"),
            "team_abbrev": g("TEAM_ABBREVIATION"),
            "jersey": g("JERSEY"),
            "draft_year": g("DRAFT_YEAR"),
            "draft_round": g("DRAFT_ROUND"),
            "draft_pick": g("DRAFT_NUMBER"),
            "status": "Active" if g("ROSTERSTATUS") == "Active" else "Inactive/Retired",
        }
        return bio
    except Exception:
        return None


def bio_to_text(bio: Dict) -> str:
    """Convert player bio dict to readable text for chat context."""
    if not bio:
        return "Player information not found."
    lines = [
        f"Name: {bio.get('name', 'N/A')}",
        f"Position: {bio.get('position', 'N/A')}",
        f"Height: {bio.get('height', 'N/A')}  |  Weight: {bio.get('weight', 'N/A')}",
        f"Country: {bio.get('country', 'N/A')}",
        f"College/School: {bio.get('college', 'N/A')}",
        f"Status: {bio.get('status', 'N/A')}  |  Current Team: {bio.get('team', 'N/A')}",
        f"Jersey #: {bio.get('jersey', 'N/A')}  |  Years Pro: {bio.get('years_pro', 'N/A')}",
    ]
    d_year = bio.get("draft_year", "N/A")
    d_round = bio.get("draft_round", "N/A")
    d_pick = bio.get("draft_pick", "N/A")
    if d_year != "N/A":
        lines.append(f"Draft: {d_year}  Round {d_round}  Pick #{d_pick}")
    return "\n".join(lines)


def get_advanced_player_stats(
    player_name: str = None,
    season=None,
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Get advanced analytics for players: TS%, usage rate, offensive/defensive rating, PIE.
    If player_name given, returns that player. Otherwise returns top N league leaders by PIE.
    """
    try:
        from nba_api.stats.endpoints import leaguedashplayerstats

        season_str = parse_season(season)
        _sleep()
        stats = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season_str,
            per_mode_detailed="PerGame",
            measure_type_detailed_defense="Advanced",
            season_type_all_star="Regular Season",
            timeout=30,
        )
        df = stats.get_data_frames()[0]
        if df.empty:
            return pd.DataFrame()

        cols = {
            "PLAYER_NAME": "Player", "TEAM_ABBREVIATION": "Team", "AGE": "Age",
            "GP": "GP", "MIN": "MIN",
            "OFF_RATING": "OffRtg", "DEF_RATING": "DefRtg", "NET_RATING": "NetRtg",
            "TS_PCT": "TS%", "USG_PCT": "USG%", "AST_PCT": "AST%",
            "REB_PCT": "REB%", "EFG_PCT": "eFG%",
            "OREB_PCT": "OREB%", "DREB_PCT": "DREB%", "PIE": "PIE",
        }
        available = {k: v for k, v in cols.items() if k in df.columns}
        result = df[list(available.keys())].rename(columns=available)

        if player_name:
            resolved = resolve_player_name(player_name)
            last = resolved.split()[-1].lower()
            mask = result["Player"].str.lower().str.contains(last, na=False)
            player_rows = result[mask]
            if not player_rows.empty:
                return player_rows.reset_index(drop=True)

        if "PIE" in result.columns:
            result = result.sort_values("PIE", ascending=False)
        return result.head(top_n).reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def get_league_player_stats(
    season=None,
    stats_type: str = "base",
    per_mode: str = "PerGame",
) -> pd.DataFrame:
    """
    Get stats for ALL NBA players in a season — the complete league dataset.
    Perfect for analytics projects where you need every player's numbers.
    """
    try:
        from nba_api.stats.endpoints import leaguedashplayerstats

        season_str = parse_season(season)
        measure = "Advanced" if stats_type == "advanced" else "Base"
        _sleep()
        stats = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season_str,
            per_mode_detailed=per_mode,
            measure_type_detailed_defense=measure,
            season_type_all_star="Regular Season",
            timeout=30,
        )
        df = stats.get_data_frames()[0]
        if df.empty:
            return pd.DataFrame()

        if stats_type == "advanced":
            cols = {
                "PLAYER_NAME": "Player", "TEAM_ABBREVIATION": "Team", "AGE": "Age",
                "GP": "GP", "MIN": "MIN",
                "OFF_RATING": "OffRtg", "DEF_RATING": "DefRtg", "NET_RATING": "NetRtg",
                "TS_PCT": "TS%", "USG_PCT": "USG%", "AST_PCT": "AST%",
                "REB_PCT": "REB%", "EFG_PCT": "eFG%",
                "OREB_PCT": "OREB%", "DREB_PCT": "DREB%", "PIE": "PIE",
            }
        else:
            cols = {
                "PLAYER_NAME": "Player", "TEAM_ABBREVIATION": "Team", "AGE": "Age",
                "GP": "GP", "MIN": "MIN", "PTS": "PTS",
                "FGM": "FGM", "FGA": "FGA", "FG_PCT": "FG%",
                "FG3M": "3PM", "FG3A": "3PA", "FG3_PCT": "3P%",
                "FTM": "FTM", "FTA": "FTA", "FT_PCT": "FT%",
                "OREB": "OREB", "DREB": "DREB", "REB": "REB",
                "AST": "AST", "TOV": "TOV", "STL": "STL",
                "BLK": "BLK", "PF": "PF", "PLUS_MINUS": "+/-",
            }

        available = {k: v for k, v in cols.items() if k in df.columns}
        result = df[list(available.keys())].rename(columns=available)
        sort_col = "PTS" if "PTS" in result.columns else ("PIE" if "PIE" in result.columns else result.columns[-1])
        return result.sort_values(sort_col, ascending=False).reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def get_all_time_leaders(stat: str = "PTS", limit: int = 25) -> pd.DataFrame:
    """
    Get all-time NBA statistical leaders.
    Stat options: PTS, AST, REB, STL, BLK, FG3M, GP
    """
    try:
        from nba_api.stats.endpoints import alltimeleadersgrids

        _sleep()
        leaders = alltimeleadersgrids.AllTimeLeadersGrids(
            league_id="00",
            per_mode_simple="Totals",
            season_type="Regular Season",
            topx=limit,
            timeout=30,
        )
        dfs = leaders.get_data_frames()
        stat_map = {"PTS": 0, "AST": 1, "REB": 2, "STL": 3, "BLK": 4, "FG3M": 5, "GP": 6}
        idx = stat_map.get(stat.upper(), 0)

        if idx < len(dfs) and not dfs[idx].empty:
            df = dfs[idx].copy()
            # Standardize column names
            df.columns = [c.replace("ATL_", "").replace("_", " ").title() for c in df.columns]
            return df.head(limit).reset_index(drop=True)
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def get_player_career_full(player_name: str, season_type: str = "Regular Season") -> pd.DataFrame:
    """
    Get complete career stats for a player across all seasons — every year as a row.
    Great for career trajectory analysis.
    """
    try:
        from nba_api.stats.endpoints import playercareerstats

        player_info = find_player_id(player_name)
        if not player_info:
            return pd.DataFrame()
        player_id, full_name = player_info

        _sleep()
        career = playercareerstats.PlayerCareerStats(
            player_id=player_id,
            per_mode36="PerGame",
            timeout=30,
        )
        dfs = career.get_data_frames()

        target = "SeasonTotalsPostSeason" if season_type.lower().startswith("post") else "SeasonTotalsRegularSeason"
        df = next((d for d in dfs if hasattr(d, "name") and d.name == target), dfs[0] if dfs else pd.DataFrame())

        if df.empty:
            return pd.DataFrame()

        cols = {
            "SEASON_ID": "Season", "TEAM_ABBREVIATION": "Team",
            "GP": "GP", "GS": "GS", "MIN": "MIN",
            "FGM": "FGM", "FGA": "FGA", "FG_PCT": "FG%",
            "FG3M": "3PM", "FG3A": "3PA", "FG3_PCT": "3P%",
            "FTM": "FTM", "FTA": "FTA", "FT_PCT": "FT%",
            "OREB": "OREB", "DREB": "DREB", "REB": "REB",
            "AST": "AST", "STL": "STL", "BLK": "BLK",
            "TOV": "TOV", "PF": "PF", "PTS": "PTS",
        }
        available = {k: v for k, v in cols.items() if k in df.columns}
        result = df[list(available.keys())].rename(columns=available)
        result.insert(0, "Player", full_name)
        return result.reset_index(drop=True)
    except Exception:
        return pd.DataFrame()
