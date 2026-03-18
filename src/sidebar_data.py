"""
Sidebar data: standings, player info, team schedules, leaderboards.
Uses NBA CDN (free, no API key) + our CSV data.
"""
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# Try to load .env
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

USER_AGENT = "NBA-Chatbot/1.0 (Educational)"


# ============================================================
# TEAM DATA
# ============================================================

TEAM_INFO = {
    "ATL": {"name": "Atlanta Hawks", "conference": "East", "championships": 1, "best_year": "1958"},
    "BOS": {"name": "Boston Celtics", "conference": "East", "championships": 18, "best_year": "2024"},
    "BRK": {"name": "Brooklyn Nets", "conference": "East", "championships": 2, "best_year": "1976"},
    "CHO": {"name": "Charlotte Hornets", "conference": "East", "championships": 0, "best_year": "-"},
    "CHI": {"name": "Chicago Bulls", "conference": "East", "championships": 6, "best_year": "1996"},
    "CLE": {"name": "Cleveland Cavaliers", "conference": "East", "championships": 1, "best_year": "2016"},
    "DAL": {"name": "Dallas Mavericks", "conference": "West", "championships": 1, "best_year": "2011"},
    "DEN": {"name": "Denver Nuggets", "conference": "West", "championships": 1, "best_year": "2023"},
    "DET": {"name": "Detroit Pistons", "conference": "East", "championships": 3, "best_year": "2004"},
    "GSW": {"name": "Golden State Warriors", "conference": "West", "championships": 7, "best_year": "2017"},
    "HOU": {"name": "Houston Rockets", "conference": "West", "championships": 2, "best_year": "1995"},
    "IND": {"name": "Indiana Pacers", "conference": "East", "championships": 0, "best_year": "-"},
    "LAC": {"name": "LA Clippers", "conference": "West", "championships": 0, "best_year": "-"},
    "LAL": {"name": "Los Angeles Lakers", "conference": "West", "championships": 17, "best_year": "2020"},
    "MEM": {"name": "Memphis Grizzlies", "conference": "West", "championships": 0, "best_year": "-"},
    "MIA": {"name": "Miami Heat", "conference": "East", "championships": 3, "best_year": "2013"},
    "MIL": {"name": "Milwaukee Bucks", "conference": "East", "championships": 2, "best_year": "2021"},
    "MIN": {"name": "Minnesota Timberwolves", "conference": "West", "championships": 0, "best_year": "-"},
    "NOP": {"name": "New Orleans Pelicans", "conference": "West", "championships": 0, "best_year": "-"},
    "NYK": {"name": "New York Knicks", "conference": "East", "championships": 2, "best_year": "1973"},
    "OKC": {"name": "Oklahoma City Thunder", "conference": "West", "championships": 1, "best_year": "1979"},
    "ORL": {"name": "Orlando Magic", "conference": "East", "championships": 0, "best_year": "-"},
    "PHI": {"name": "Philadelphia 76ers", "conference": "East", "championships": 3, "best_year": "1983"},
    "PHO": {"name": "Phoenix Suns", "conference": "West", "championships": 0, "best_year": "-"},
    "POR": {"name": "Portland Trail Blazers", "conference": "West", "championships": 1, "best_year": "1977"},
    "SAC": {"name": "Sacramento Kings", "conference": "West", "championships": 1, "best_year": "1951"},
    "SAS": {"name": "San Antonio Spurs", "conference": "West", "championships": 5, "best_year": "2014"},
    "TOR": {"name": "Toronto Raptors", "conference": "East", "championships": 1, "best_year": "2019"},
    "UTA": {"name": "Utah Jazz", "conference": "West", "championships": 0, "best_year": "-"},
    "WAS": {"name": "Washington Wizards", "conference": "East", "championships": 1, "best_year": "1978"},
}

# Aliases for team name lookups
TEAM_ALIASES = {
    "warriors": "GSW", "golden state": "GSW", "gsw": "GSW",
    "lakers": "LAL", "los angeles lakers": "LAL", "lal": "LAL",
    "celtics": "BOS", "boston": "BOS", "bos": "BOS",
    "nets": "BRK", "brooklyn": "BRK", "brk": "BRK", "bkn": "BRK",
    "hornets": "CHO", "charlotte": "CHO", "cho": "CHO", "cha": "CHO",
    "bulls": "CHI", "chicago": "CHI", "chi": "CHI",
    "cavaliers": "CLE", "cavs": "CLE", "cleveland": "CLE", "cle": "CLE",
    "mavericks": "DAL", "mavs": "DAL", "dallas": "DAL", "dal": "DAL",
    "nuggets": "DEN", "denver": "DEN", "den": "DEN",
    "pistons": "DET", "detroit": "DET", "det": "DET",
    "rockets": "HOU", "houston": "HOU", "hou": "HOU",
    "pacers": "IND", "indiana": "IND", "ind": "IND",
    "clippers": "LAC", "la clippers": "LAC", "lac": "LAC",
    "grizzlies": "MEM", "memphis": "MEM", "mem": "MEM",
    "heat": "MIA", "miami": "MIA", "mia": "MIA",
    "bucks": "MIL", "milwaukee": "MIL", "mil": "MIL",
    "timberwolves": "MIN", "wolves": "MIN", "minnesota": "MIN", "min": "MIN",
    "pelicans": "NOP", "new orleans": "NOP", "nop": "NOP",
    "knicks": "NYK", "new york": "NYK", "nyk": "NYK",
    "thunder": "OKC", "oklahoma city": "OKC", "okc": "OKC",
    "magic": "ORL", "orlando": "ORL", "orl": "ORL",
    "76ers": "PHI", "sixers": "PHI", "philadelphia": "PHI", "phi": "PHI",
    "suns": "PHO", "phoenix": "PHO", "pho": "PHO", "phx": "PHO",
    "trail blazers": "POR", "blazers": "POR", "portland": "POR", "por": "POR",
    "kings": "SAC", "sacramento": "SAC", "sac": "SAC",
    "spurs": "SAS", "san antonio": "SAS", "sas": "SAS",
    "raptors": "TOR", "toronto": "TOR", "tor": "TOR",
    "jazz": "UTA", "utah": "UTA", "uta": "UTA",
    "wizards": "WAS", "washington": "WAS", "was": "WAS",
    "hawks": "ATL", "atlanta": "ATL", "atl": "ATL",
}


def resolve_team_abbrev(query: str) -> Optional[str]:
    """Convert team name/alias to standard abbreviation."""
    q = query.strip().lower()
    if q.upper() in TEAM_INFO:
        return q.upper()
    return TEAM_ALIASES.get(q)


# ============================================================
# STANDINGS (from ESPN API - more reliable than NBA.com)
# ============================================================

# ESPN abbreviation mapping to our standard abbreviations
ESPN_ABBREV_MAP = {
    "NY": "NYK", "NO": "NOP", "UTAH": "UTA", "SA": "SAS", 
    "GS": "GSW", "PHX": "PHO", "WSH": "WAS", "CHA": "CHO",
}


def fetch_conference_standings(conference: str = "East") -> List[Dict]:
    """
    Fetch conference standings from ESPN API.
    
    Args:
        conference: "East" or "West"
    
    Returns:
        List of {"rank": 1, "team": "CLE", "name": "Cavaliers", "wins": 43, "losses": 14, "pct": ".754"}
    """
    try:
        import requests
        
        url = "https://site.api.espn.com/apis/v2/sports/basketball/nba/standings"
        headers = {"User-Agent": "Mozilla/5.0"}
        
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        data = r.json()
        
        # ESPN returns data.children = [East conf, West conf]
        children = data.get("children", [])
        
        # Find the requested conference
        conf_data = None
        for child in children:
            conf_name = child.get("name", "").lower()
            if conference.lower() in conf_name:
                conf_data = child
                break
        
        if not conf_data:
            return []
        
        entries = conf_data.get("standings", {}).get("entries", [])
        if not entries:
            return []
        
        # Extract team data
        teams = []
        for entry in entries:
            team = entry.get("team", {})
            abbrev_raw = team.get("abbreviation", "")
            # Map ESPN abbreviation to our standard
            abbrev = ESPN_ABBREV_MAP.get(abbrev_raw, abbrev_raw)
            name = team.get("shortDisplayName", team.get("name", ""))
            
            stats = entry.get("stats", [])
            wins = losses = 0
            for stat in stats:
                if stat.get("name") == "wins":
                    wins = int(stat.get("value", 0))
                elif stat.get("name") == "losses":
                    losses = int(stat.get("value", 0))
            
            pct = wins / (wins + losses) if (wins + losses) > 0 else 0
            teams.append({
                "team": abbrev,
                "name": name,
                "wins": wins,
                "losses": losses,
                "pct": f"{pct:.3f}",
            })
        
        # Sort by wins (descending) and assign ranks
        teams.sort(key=lambda x: x["wins"], reverse=True)
        for i, t in enumerate(teams, 1):
            t["rank"] = i
        
        return teams[:10]
    except Exception:
        return []


def fetch_all_standings() -> Dict[str, List[Dict]]:
    """Fetch both conference standings."""
    return {
        "east": fetch_conference_standings("East"),
        "west": fetch_conference_standings("West"),
    }


def get_team_standing(team_abbrev: str) -> Optional[Dict]:
    """Get standing info for a specific team."""
    standings = fetch_all_standings()
    for conf in ["east", "west"]:
        for t in standings.get(conf, []):
            if t.get("team", "").upper() == team_abbrev.upper():
                return t
    return None


def fetch_team_standing_for_season(team_abbrev: str, season: str) -> Optional[Dict]:
    """
    Get team's standing for a specific season.
    season format: "2024-25"
    """
    try:
        import requests
        
        url = "https://stats.nba.com/stats/leaguestandingsv3"
        params = {
            "LeagueID": "00",
            "Season": season,
            "SeasonType": "Regular Season",
        }
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Referer": "https://www.nba.com/",
            "Accept": "application/json",
            "Origin": "https://www.nba.com",
        }
        
        r = requests.get(url, params=params, headers=headers, timeout=20)
        r.raise_for_status()
        data = r.json()
        
        result_set = data.get("resultSets", [{}])[0]
        headers_list = result_set.get("headers", [])
        rows = result_set.get("rowSet", [])
        
        if not headers_list or not rows:
            return None
        
        try:
            abbrev_idx = headers_list.index("TeamAbbreviation")
            wins_idx = headers_list.index("WINS")
            losses_idx = headers_list.index("LOSSES")
            rank_idx = headers_list.index("PlayoffRank")
            conf_idx = headers_list.index("Conference")
        except ValueError:
            return None
        
        for row in rows:
            if row[abbrev_idx].upper() == team_abbrev.upper():
                return {
                    "rank": row[rank_idx],
                    "wins": row[wins_idx],
                    "losses": row[losses_idx],
                    "conference": row[conf_idx],
                }
        return None
    except Exception:
        return None


def get_team_full_info(team_abbrev: str) -> Optional[Dict]:
    """
    Get complete team info: current standing, previous year, championships.
    """
    team_abbrev = team_abbrev.upper()
    if team_abbrev not in TEAM_INFO:
        return None
    
    info = TEAM_INFO[team_abbrev].copy()
    info["abbrev"] = team_abbrev
    
    # Get current season standing
    current = get_team_standing(team_abbrev)
    if current:
        info["current_rank"] = current.get("rank")
        info["current_record"] = f"{current.get('wins', 0)}-{current.get('losses', 0)}"
    
    # Get previous season standing
    now = datetime.now()
    year = now.year
    if now.month >= 10:
        prev_season = f"{year-1}-{str(year)[-2:]}"
    else:
        prev_season = f"{year-2}-{str(year-1)[-2:]}"
    
    prev = fetch_team_standing_for_season(team_abbrev, prev_season)
    if prev:
        info["prev_season"] = prev_season
        info["prev_rank"] = prev.get("rank")
        info["prev_record"] = f"{prev.get('wins', 0)}-{prev.get('losses', 0)}"
    
    return info


# ============================================================
# SCHEDULE (from live_data.py)
# ============================================================

def get_team_next_games(team_abbrev: str, n: int = 3) -> List[Dict]:
    """Get next N upcoming games for a team."""
    try:
        from src.live_data import _fetch_nba_cdn_schedule, _parse_nba_cdn_games
        data = _fetch_nba_cdn_schedule()
        if data:
            games = _parse_nba_cdn_games(data, team_abbrev, n=n, upcoming=True)
            return games
    except Exception:
        pass
    return []


def get_all_upcoming_games(n: int = 5) -> List[Dict]:
    """Get next N games across all teams (today's games first)."""
    try:
        from src.live_data import _fetch_nba_cdn_schedule
        data = _fetch_nba_cdn_schedule()
        if not data:
            return []
        
        games = []
        today = datetime.now().strftime("%Y-%m-%d")
        dates = data.get("leagueSchedule", {}).get("gameDates", [])
        
        for gd in dates:
            gdate_raw = gd.get("gameDate", "")
            try:
                if " " in gdate_raw:
                    gdate_raw = gdate_raw.split()[0]
                parts = gdate_raw.split("/")
                if len(parts) == 3:
                    dt = f"{parts[2]}-{parts[0].zfill(2)}-{parts[1].zfill(2)}"
                else:
                    dt = gdate_raw[:10]
            except Exception:
                dt = gdate_raw[:10]
            
            if dt < today:
                continue
            
            for g in gd.get("games", []):
                home = (g.get("homeTeam") or {}).get("teamTricode", "")
                away = (g.get("awayTeam") or {}).get("teamTricode", "")
                status = g.get("gameStatusText", "")
                time_str = g.get("gameTimeUTC", "")
                
                games.append({
                    "date": dt,
                    "home": home,
                    "away": away,
                    "status": status,
                    "time": time_str,
                })
                
                if len(games) >= n:
                    return games
    except Exception:
        pass
    return []


# ============================================================
# LIVE SCOREBOARD (today's games with live scores)
# ============================================================

def get_todays_scoreboard() -> List[Dict]:
    """
    Fetch today's NBA scoreboard with live scores from NBA CDN.
    Returns list of games with scores, status, quarter info.
    """
    try:
        import requests
        
        today = datetime.now().strftime("%Y%m%d")  # Format: 20260228
        url = f"https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json"
        
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=10)
        r.raise_for_status()
        data = r.json()
        
        games = []
        scoreboard = data.get("scoreboard", {})
        
        for g in scoreboard.get("games", []):
            home_team = g.get("homeTeam", {})
            away_team = g.get("awayTeam", {})
            
            game_status = g.get("gameStatus", 1)  # 1=Not started, 2=In progress, 3=Final
            game_status_text = g.get("gameStatusText", "")
            period = g.get("period", 0)
            game_clock = g.get("gameClock", "")
            
            # Clean up game clock (remove "PT" prefix from ISO format)
            if game_clock.startswith("PT"):
                mins = game_clock.replace("PT", "").replace("M", ":").replace("S", "").replace(".00", "")
                if ":" in mins:
                    parts = mins.split(":")
                    if len(parts) == 2:
                        game_clock = f"{parts[0]}:{parts[1].zfill(2)}"
                    else:
                        game_clock = mins
                else:
                    game_clock = mins
            
            game_info = {
                "home_team": home_team.get("teamTricode", ""),
                "away_team": away_team.get("teamTricode", ""),
                "home_score": home_team.get("score", 0),
                "away_score": away_team.get("score", 0),
                "status": game_status,  # 1=Scheduled, 2=Live, 3=Final
                "status_text": game_status_text,
                "period": period,
                "clock": game_clock,
            }
            games.append(game_info)
        
        return games
    except Exception:
        return []


# ============================================================
# PLAYER DATA (from CSV)
# ============================================================

def get_player_headshot_url(player_id: str) -> str:
    """
    Get NBA headshot URL for a player.
    Uses nba.com CDN with player ID mapped from basketball-reference ID.
    Falls back to a placeholder if not found.
    """
    # NBA.com uses numeric IDs, but we have basketball-reference IDs
    # For now, return a search URL or placeholder
    # In production, you'd maintain a mapping file
    return f"https://www.basketball-reference.com/req/202106291/images/headshots/{player_id}.jpg"


def search_players(df: pd.DataFrame, query: str, limit: int = 5) -> List[Dict]:
    """Search players by name, return list of matches with basic info."""
    q = query.strip().lower()
    if len(q) < 2:
        return []
    
    matches = df[df["Player"].str.lower().str.contains(q, na=False)].copy()
    matches = matches.sort_values("Season", ascending=False)
    matches = matches.drop_duplicates(subset=["Player Reference"], keep="first")
    
    results = []
    for _, row in matches.head(limit).iterrows():
        results.append({
            "name": row["Player"],
            "player_id": row["Player Reference"],
            "team": row["Team"],
            "last_season": row["Season"],
        })
    return results


def get_player_career_stats(df: pd.DataFrame, player_id: str) -> Optional[Dict]:
    """Get career totals for a player."""
    sub = df[df["Player Reference"] == player_id].copy()
    if sub.empty:
        return None
    
    # Prefer TOT rows to avoid double-counting
    sub["_is_tot"] = (sub["Team"] == "TOT").astype(int)
    sub = sub.sort_values(by=["Season", "_is_tot", "MP"], ascending=[True, False, False])
    sub = sub.drop_duplicates(subset=["Season"], keep="first")
    
    # Get first and last seasons
    seasons = sorted(sub["Season"].unique())
    first_season = seasons[0] if seasons else ""
    last_season = seasons[-1] if seasons else ""
    first_team = sub[sub["Season"] == first_season]["Team"].iloc[0] if not sub.empty else ""
    current_team = sub[sub["Season"] == last_season]["Team"].iloc[0] if not sub.empty else ""
    
    # Sum stats
    sum_cols = ["G", "MP", "PTS", "AST", "TRB", "STL", "BLK", "3P", "3PA"]
    totals = {}
    for col in sum_cols:
        if col in sub.columns:
            totals[col] = int(sub[col].sum())
    
    # Calculate 3P%
    if totals.get("3PA", 0) > 0:
        totals["3P%"] = round(totals["3P"] / totals["3PA"], 3)
    
    return {
        "name": sub["Player"].iloc[0],
        "player_id": player_id,
        "first_season": first_season,
        "last_season": last_season,
        "first_team": first_team,
        "current_team": current_team,
        "seasons_played": len(seasons),
        "career": totals,
    }


def get_player_season_stats(df: pd.DataFrame, player_id: str, season: str) -> Optional[Dict]:
    """Get stats for a specific season."""
    sub = df[(df["Player Reference"] == player_id) & (df["Season"] == season)].copy()
    if sub.empty:
        return None
    
    # Prefer TOT row
    sub["_is_tot"] = (sub["Team"] == "TOT").astype(int)
    sub = sub.sort_values(by=["_is_tot", "MP"], ascending=[False, False])
    row = sub.iloc[0]
    
    stats = {"season": season, "team": row["Team"]}
    for col in ["G", "MP", "PTS", "AST", "TRB", "STL", "BLK", "3P", "3PA", "3P%"]:
        if col in row.index:
            val = row[col]
            stats[col] = int(val) if col not in ["3P%"] else round(float(val), 3)
    
    return stats


def get_player_full_info(df: pd.DataFrame, player_id: str) -> Optional[Dict]:
    """Get complete player info: career stats, current/last season stats, basic info."""
    career = get_player_career_stats(df, player_id)
    if not career:
        return None
    
    # Get current season (2025-2026)
    now = datetime.now()
    year = now.year
    current_season = f"{year}-{year+1}" if now.month >= 10 else f"{year-1}-{year}"
    
    current_stats = get_player_season_stats(df, player_id, current_season)
    
    # If no current season stats, get last available season
    last_season_stats = None
    if not current_stats:
        last_season_stats = get_player_season_stats(df, player_id, career["last_season"])
    
    # Determine if player is active (played in current or last season)
    is_active = current_stats is not None or career["last_season"] >= f"{year-2}-{year-1}"
    
    return {
        **career,
        "current_season_stats": current_stats,
        "last_season_stats": last_season_stats,
        "is_active": is_active,
        "headshot_url": get_player_headshot_url(player_id),
    }


# ============================================================
# LEADERBOARDS (from CSV)
# ============================================================

def get_season_leaders(df: pd.DataFrame, season: str, stat: str = "PTS", limit: int = 5) -> List[Dict]:
    """Get top players for a stat in a season."""
    if stat not in df.columns:
        return []
    
    sub = df[df["Season"] == season].copy()
    sub = sub[sub["G"] >= 20]  # Minimum games filter
    
    # Prefer TOT rows
    sub["_is_tot"] = (sub["Team"] == "TOT").astype(int)
    sub = sub.sort_values(by=["Player Reference", "_is_tot", "MP"], ascending=[True, False, False])
    sub = sub.drop_duplicates(subset=["Player Reference"], keep="first")
    
    sub = sub.sort_values(by=stat, ascending=False)
    
    results = []
    for _, row in sub.head(limit).iterrows():
        results.append({
            "name": row["Player"],
            "team": row["Team"],
            "value": int(row[stat]) if stat != "3P%" else round(float(row[stat]), 3),
        })
    return results


def get_current_season_string() -> str:
    """Get current season string like '2025-2026'."""
    now = datetime.now()
    year = now.year
    return f"{year}-{year+1}" if now.month >= 10 else f"{year-1}-{year}"


# ============================================================
# LIVE LEADERS (from NBA.com stats API)
# ============================================================

def fetch_live_leaders(stat: str = "PTS", limit: int = 5, per_game: bool = True) -> List[Dict]:
    """
    Fetch live season leaders from NBA.com stats API.
    
    Args:
        stat: One of "PTS", "AST", "REB", "STL", "BLK"
        limit: Number of leaders to return
        per_game: If True, show per-game averages (like official NBA). If False, show totals.
    
    Returns:
        List of {"rank": 1, "name": "Player", "team": "ABC", "value": 32.7}
    """
    try:
        import requests
        
        # Get current season in NBA format (2025-26)
        now = datetime.now()
        year = now.year
        if now.month >= 10:
            season = f"{year}-{str(year+1)[-2:]}"
        else:
            season = f"{year-1}-{str(year)[-2:]}"
        
        url = "https://stats.nba.com/stats/leagueLeaders"
        params = {
            "LeagueID": "00",
            "PerMode": "PerGame" if per_game else "Totals",
            "Scope": "S",
            "Season": season,
            "SeasonType": "Regular Season",
            "StatCategory": stat,
        }
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Referer": "https://www.nba.com/",
            "Accept": "application/json",
        }
        
        r = requests.get(url, params=params, headers=headers, timeout=15)
        r.raise_for_status()
        data = r.json()
        
        # Parse response
        result_set = data.get("resultSet", {})
        headers_list = result_set.get("headers", [])
        rows = result_set.get("rowSet", [])
        
        # Find column indices
        try:
            rank_idx = headers_list.index("RANK")
            name_idx = headers_list.index("PLAYER")
            team_idx = headers_list.index("TEAM")
            # Find the stat column
            stat_idx = headers_list.index(stat) if stat in headers_list else -1
        except ValueError:
            return []
        
        leaders = []
        for row in rows[:limit]:
            value = row[stat_idx] if stat_idx >= 0 else 0
            # Format: show 1 decimal for per-game, whole number for totals
            if per_game:
                value = round(float(value), 1)
            else:
                value = int(value)
            
            leaders.append({
                "rank": row[rank_idx],
                "name": row[name_idx],
                "team": row[team_idx],
                "value": value,
            })
        
        return leaders
    except Exception:
        return []


def fetch_all_live_leaders() -> Dict[str, List[Dict]]:
    """
    Fetch leaders for multiple stat categories.
    Returns dict with keys: "points", "assists", "rebounds"
    """
    return {
        "points": fetch_live_leaders("PTS", limit=3),
        "assists": fetch_live_leaders("AST", limit=3),
        "rebounds": fetch_live_leaders("REB", limit=3),
    }
