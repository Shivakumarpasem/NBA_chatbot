"""
Live data: schedules, upcoming games. Uses API or web scraping.
Add new sources by implementing fetch functions and calling them from get_live_schedule().
See docs/DATA_SOURCES.md for how to modify.
"""
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

# Load .env
ROOT = Path(__file__).resolve().parents[1]
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

USER_AGENT = "NBA-Chatbot/1.0 (Educational)"


def _fetch_nba_cdn_schedule() -> Optional[dict]:
    """Fetch schedule from NBA CDN (no API key). Returns raw JSON or None."""
    try:
        import requests
        url = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json"
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def _fetch_balldontlie_games(team_abbr: str, days_ahead: int = 30) -> Optional[List[dict]]:
    """Fetch games from balldontlie.io (requires BALDONTLIE_API_KEY in .env)."""
    key = os.environ.get("BALDONTLIE_API_KEY")
    if not key:
        return None
    try:
        import requests
        # Map our abbrev to balldontlie team IDs if needed; for now use dates
        today = datetime.now().date()
        end = today + timedelta(days=days_ahead)
        url = "https://api.balldontlie.io/v1/games"
        r = requests.get(
            url,
            headers={"Authorization": key},
            params={"start_date": str(today), "end_date": str(end), "per_page": 100},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        games = data.get("data", [])
        # Filter by team (home/visitor)
        team_lower = team_abbr.lower()
        out = []
        for g in games:
            h = (g.get("home_team", {}) or {}).get("abbreviation", "").lower()
            v = (g.get("visitor_team", {}) or {}).get("abbreviation", "").lower()
            if team_lower in (h, v):
                out.append(g)
        return out[:10] if out else None
    except Exception:
        return None


def _parse_nba_cdn_games(data: dict, team_abbr: str, n: int = 10, upcoming: bool = True) -> List[dict]:
    """Parse NBA CDN JSON into list of {date, visitor, home, arena, status}."""
    games = []
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        dates = data.get("leagueSchedule", {}).get("gameDates", [])
        team_upper = team_abbr.upper()
        for gd in dates:
            gdate_raw = gd.get("gameDate", "")
            # Parse "10/02/2025 00:00:00" -> 2025-10-02
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
            if upcoming and dt < today:
                continue
            for g in gd.get("games", []):
                home = (g.get("homeTeam") or {}).get("teamTricode", "")
                away = (g.get("awayTeam") or {}).get("teamTricode", "")
                if team_upper not in (home, away):
                    continue
                if not dt:
                    dt = g.get("gameDateEst", "")[:10]
                status = g.get("gameStatusText", "")
                arena = g.get("arenaName", "")
                games.append({
                    "date": dt,
                    "visitor": away,
                    "home": home,
                    "arena": arena,
                    "status": status,
                })
                if len(games) >= n:
                    return games
    except Exception:
        pass
    return games


def get_live_schedule(team_abbr: str, n: int = 10, upcoming: bool = True) -> Optional[str]:
    """
    Fetch live schedule for a team. Tries balldontlie (if key) then NBA CDN.
    Returns formatted string for display, or None on failure.
    """
    # 1. Try balldontlie if key present
    bdl = _fetch_balldontlie_games(team_abbr, days_ahead=60)
    if bdl:
        lines = []
        for g in bdl[:n]:
            date = g.get("date", "")[:10]
            home = (g.get("home_team") or {}).get("full_name", "?")
            visitor = (g.get("visitor_team") or {}).get("full_name", "?")
            status = g.get("status", "")
            lines.append(f"{date} | {visitor} @ {home} | {status}")
        if lines:
            return "Live schedule (balldontlie):\n" + "\n".join(lines)

    # 2. NBA CDN
    data = _fetch_nba_cdn_schedule()
    if data:
        parsed = _parse_nba_cdn_games(data, team_abbr, n=n, upcoming=upcoming)
        if not parsed and upcoming:
            parsed = _parse_nba_cdn_games(data, team_abbr, n=n, upcoming=False)
        if parsed:
            label = "Upcoming" if upcoming else "Recent"
            lines = [f"{g['date']} | {g['visitor']} @ {g['home']} | {g['status']} | {g['arena']}" for g in parsed]
            return f"Live schedule (NBA.com) - {label} games:\n" + "\n".join(lines)

    return None


def get_live_schedule_text(team_abbr: str, n: int = 10, query: str = "", target_date: Optional[str] = None) -> str:
    """
    Public helper: returns schedule string or empty. Used by orchestrator.
    If target_date (YYYY-MM-DD) is set, returns only that day's game(s).
    """
    if target_date:
        single = get_game_on_date(team_abbr, target_date)
        return single or ""
    text = get_live_schedule(team_abbr, n=n, upcoming=True)
    return text or ""


def get_game_on_date(team_abbr: str, target_date: str) -> Optional[str]:
    """Return a single game for the given date, or None."""
    data = _fetch_nba_cdn_schedule()
    if not data:
        return None
    today = datetime.now().strftime("%Y-%m-%d")
    dates = data.get("leagueSchedule", {}).get("gameDates", [])
    team_upper = team_abbr.upper()
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
        if dt != target_date:
            continue
        for g in gd.get("games", []):
            home = (g.get("homeTeam") or {}).get("teamTricode", "")
            away = (g.get("awayTeam") or {}).get("teamTricode", "")
            if team_upper not in (home, away):
                continue
            status = g.get("gameStatusText", "")
            arena = g.get("arenaName", "")
            return f"{away} @ {home} | {status} | {arena}"
    return None


def _nba_api_headers():
    return {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://www.nba.com/",
        "Accept": "application/json",
    }


def _get_nba_season_string(year_start: int) -> str:
    """Convert 2024 -> '2024-25' (NBA API format)."""
    return f"{year_start}-{str(year_start + 1)[-2:]}"


def _get_current_season_year() -> int:
    """Returns the start year of the current NBA season (e.g., 2025 for 2025-26)."""
    now = datetime.now()
    return now.year if now.month >= 10 else now.year - 1


def _find_nba_player_id(player_name: str) -> Optional[tuple]:
    """
    Find NBA.com player ID by name. Returns (player_id, full_name, team) or None.
    Searches current season active players.
    """
    try:
        import requests
        
        season_str = _get_nba_season_string(_get_current_season_year())
        
        r = requests.get(
            "https://stats.nba.com/stats/commonallplayers",
            params={"LeagueID": "00", "Season": season_str, "IsOnlyCurrentSeason": "0"},
            headers=_nba_api_headers(),
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        
        result_set = data.get("resultSets", [{}])[0]
        headers_list = result_set.get("headers", [])
        rows = result_set.get("rowSet", [])
        
        id_idx = headers_list.index("PERSON_ID")
        name_idx = headers_list.index("DISPLAY_FIRST_LAST")
        team_idx = headers_list.index("TEAM_ABBREVIATION") if "TEAM_ABBREVIATION" in headers_list else -1
        from_year_idx = headers_list.index("FROM_YEAR") if "FROM_YEAR" in headers_list else -1
        to_year_idx = headers_list.index("TO_YEAR") if "TO_YEAR" in headers_list else -1
        
        search_name = re.sub(r"'s\b", "", player_name.lower()).strip()
        search_parts = search_name.split()
        if not search_parts:
            return None
        # Handle possessive typos like "durants" -> "durant" (only for last token)
        alt_last = search_parts[-1][:-1] if search_parts[-1].endswith("s") and len(search_parts[-1]) > 3 else search_parts[-1]
        search_name_alt = " ".join(search_parts[:-1] + [alt_last]).strip()
        
        candidates = []
        for row in rows:
            full_name = row[name_idx].lower()
            name_parts = full_name.split()
            team_abbrev = row[team_idx] if team_idx >= 0 else ""
            from_year = int(row[from_year_idx]) if from_year_idx >= 0 and str(row[from_year_idx]).isdigit() else 0
            to_year = int(row[to_year_idx]) if to_year_idx >= 0 and str(row[to_year_idx]).isdigit() else 0
            
            last_name_match = bool(
                name_parts and (name_parts[-1] == search_parts[-1] or name_parts[-1] == alt_last)
            )
            full_name_match = (search_name in full_name) or (search_name_alt and search_name_alt in full_name)
            if not full_name_match and not last_name_match:
                continue
                
            score = 0
            if full_name == search_name or full_name == search_name_alt:
                score += 10000
            if last_name_match:
                score += 1000
            if len(search_parts) > 1 and name_parts:
                # Accept "steph" for "stephen"
                if name_parts[0] == search_parts[0]:
                    score += 5000
                elif name_parts[0].startswith(search_parts[0]) or search_parts[0].startswith(name_parts[0]):
                    score += 3500
            # For ambiguous one-word queries (e.g., "curry"), prefer active + recent + longer-tenure players
            if len(search_parts) == 1:
                score += to_year * 2
                score += max(0, to_year - from_year)
            if team_abbrev:
                score += 5000
                
            candidates.append((row[id_idx], row[name_idx], team_abbrev, score))
        
        if not candidates:
            return None
        
        candidates.sort(key=lambda x: x[3], reverse=True)
        pid, pname, pteam, _ = candidates[0]
        return (pid, pname, pteam)
    except Exception:
        return None


def _fetch_player_season_stats(player_id: int, season_str: str, player_name: str = "", player_team: str = "") -> Optional[dict]:
    """
    Fetch stats for a specific player + season from NBA.com API.
    season_str: NBA format like '2024-25'
    """
    try:
        import requests
        
        r = requests.get(
            "https://stats.nba.com/stats/playerdashboardbygeneralsplits",
            params={
                "PlayerID": player_id,
                "Season": season_str,
                "SeasonType": "Regular Season",
                "MeasureType": "Base",
                "PerMode": "Totals",
                "PlusMinus": "N", "PaceAdjust": "N", "Rank": "N",
                "LeagueID": "00",
                "DateFrom": "", "DateTo": "", "GameSegment": "",
                "LastNGames": 0, "Location": "", "Month": 0,
                "OpponentTeamID": 0, "Outcome": "", "Period": 0,
                "SeasonSegment": "", "VsConference": "", "VsDivision": "",
            },
            headers=_nba_api_headers(),
            timeout=15,
        )
        r.raise_for_status()
        stats_data = r.json()
        
        overall = stats_data.get("resultSets", [{}])[0]
        stat_headers = overall.get("headers", [])
        stat_rows = overall.get("rowSet", [])
        
        if not stat_rows:
            return None
        
        row = stat_rows[0]
        
        def v(col):
            try:
                return row[stat_headers.index(col)]
            except (ValueError, IndexError):
                return None
        
        full_season = season_str.replace("-", "-20")  # "2024-25" -> "2024-2025"
        
        return {
            "player_name": player_name,
            "season": full_season,
            "team": v("TEAM_ABBREVIATION") or player_team or "N/A",
            "G": v("GP") or 0,
            "MP": v("MIN") or 0,
            "PTS": v("PTS") or 0,
            "AST": v("AST") or 0,
            "TRB": v("REB") or 0,
            "STL": v("STL") or 0,
            "BLK": v("BLK") or 0,
            "3P": v("FG3M") or 0,
            "3PA": v("FG3A") or 0,
            "3P%": round((v("FG3_PCT") or 0) * 100, 1),
        }
    except Exception:
        return None


def get_player_career_stats(player_name: str, season_type: str = "Regular Season") -> Optional[dict]:
    """
    Fetch a player's COMPLETE career stats (all seasons) from NBA.com in one API call.
    Uses playercareerstats endpoint which is fast and returns everything.
    
    Returns: {"player_name": str, "team": str, "seasons": [list of season dicts]} or None
    """
    player_info = _find_nba_player_id(player_name)
    if not player_info:
        return None
    
    pid, pname, pteam = player_info
    
    try:
        import requests
        
        r = requests.get(
            "https://stats.nba.com/stats/playercareerstats",
            params={
                "PlayerID": pid,
                "PerMode": "Totals",
                "LeagueID": "00",
            },
            headers=_nba_api_headers(),
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        
        result_sets = data.get("resultSets", [])
        if not result_sets:
            return None
        # Pick regular season or playoffs set by name for clarity
        target_name = "SeasonTotalsPostSeason" if season_type.lower().startswith("post") else "SeasonTotalsRegularSeason"
        result_set = next((rs for rs in result_sets if rs.get("name") == target_name), result_sets[0])
        hdrs = result_set.get("headers", [])
        rows = result_set.get("rowSet", [])
        
        if not rows:
            return None
        
        def v(row, col):
            try:
                return row[hdrs.index(col)]
            except (ValueError, IndexError):
                return None
        
        seasons = []
        for row in rows:
            season_id = v(row, "SEASON_ID")  # Format: "2024-25"
            if not season_id:
                continue
            full_season = season_id.replace("-", "-20")  # "2024-25" -> "2024-2025"
            
            gp = v(row, "GP") or 0
            if gp == 0:
                continue
            
            fg3_pct = v(row, "FG3_PCT") or 0
            
            seasons.append({
                "season": full_season,
                "season_type": "Playoffs" if season_type.lower().startswith("post") else "Regular Season",
                "team": v(row, "TEAM_ABBREVIATION") or "N/A",
                "G": gp,
                "MP": v(row, "MIN") or 0,
                "PTS": v(row, "PTS") or 0,
                "AST": v(row, "AST") or 0,
                "TRB": v(row, "REB") or 0,
                "STL": v(row, "STL") or 0,
                "BLK": v(row, "BLK") or 0,
                "3P": v(row, "FG3M") or 0,
                "3PA": v(row, "FG3A") or 0,
                "3P%": round(fg3_pct * 100, 1) if fg3_pct else 0,
            })
        
        return {
            "player_name": pname,
            "team": pteam,
            "seasons": seasons,
        }
    except Exception:
        return None


def get_player_stats_live(
    player_name: str,
    seasons: list = None,
    n_last: int = None,
    season_type: str = "Regular Season",
) -> list:
    """
    Fetch player stats from NBA.com. Gets complete career, then filters.
    
    Args:
        player_name: e.g., "Stephen Curry"
        seasons: list of season start years to filter, e.g., [2024, 2023]
                 If None and n_last is None, returns current season only.
        n_last: return the N most recent completed seasons (excludes current)
    
    Returns: list of stat dicts (one per season), or empty list.
    """
    career = get_player_career_stats(player_name, season_type=season_type)
    if not career or not career.get("seasons"):
        return []
    
    all_seasons = career["seasons"]
    canonical_name = career.get("player_name", player_name)

    def _with_name(rows: list) -> list:
        return [{**r, "player_name": canonical_name} for r in rows]
    
    if n_last is not None:
        # "Last N seasons" = N most recent COMPLETED seasons (not current)
        current_year = _get_current_season_year()
        current_season_str = f"{current_year}-{current_year + 1}"
        # Filter out current season, then take last N
        past = [s for s in all_seasons if s["season"] != current_season_str]
        # Sort by season descending
        past.sort(key=lambda x: x["season"], reverse=True)
        return _with_name(past[:n_last])
    
    if seasons is not None:
        # Filter to specific season years
        wanted = {f"{yr}-{yr + 1}" for yr in seasons}
        return _with_name([s for s in all_seasons if s["season"] in wanted])
    
    # Default: current season only
    current_year = _get_current_season_year()
    current_season_str = f"{current_year}-{current_year + 1}"
    return _with_name([s for s in all_seasons if s["season"] == current_season_str])


def get_current_season_player_stats(player_name: str) -> Optional[dict]:
    """Backward-compatible wrapper. Fetches current season only."""
    results = get_player_stats_live(player_name, seasons=None, season_type="Regular Season")
    return results[0] if results else None
