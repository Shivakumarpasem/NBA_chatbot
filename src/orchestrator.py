"""
Orchestrator: routes user query to Stats (tool-calling) or RAG (retrieval), then generates response.
Uses Google Gemini API. Requires GOOGLE_API_KEY or GEMINI_API_KEY in .env or environment.
"""
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Load .env from project root
ROOT = Path(__file__).resolve().parents[1]
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_PATH = ROOT / "data"
TOTALS_CSV = DATA_PATH / "player_stats" / "totals_stats.csv"
BOX_SCORE_CSV = DATA_PATH / "boxscores_by_year" / "NBA_2023-2024_basic.csv"
SCHEDULE_CSV = DATA_PATH / "schedule.csv"
RAG_DOCS_DIR = DATA_PATH / "rag_docs"

_df_totals = None
_df_games = None
_df_schedule = None


def _get_api_key() -> Optional[str]:
    return os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")


def _load_df():
    global _df_totals, _df_games, _df_schedule
    if _df_totals is None:
        from src.nba_data import load_adj_shooting, load_box_scores
        _df_totals = load_adj_shooting(str(TOTALS_CSV))
    if _df_games is None and BOX_SCORE_CSV.exists():
        from src.nba_data import load_box_scores
        _df_games = load_box_scores(str(BOX_SCORE_CSV))
    if _df_schedule is None and SCHEDULE_CSV.exists():
        from src.nba_data import load_schedule
        _df_schedule = load_schedule(str(SCHEDULE_CSV))
    return _df_totals, _df_games, _df_schedule


# Gemini function declarations
GEMINI_TOOLS = [
    {
        "name": "find_players",
        "description": "Search for players by name (e.g. 'curry', 'LeBron'). Returns player IDs and teams.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "player_summary",
        "description": "Get key stats for one player in a season. Use player_id from find_players.",
        "parameters": {
            "type": "object",
            "properties": {
                "player_id": {"type": "string"},
                "season": {"type": "string"},
            },
            "required": ["player_id", "season"],
        },
    },
    {
        "name": "top_stat_leaderboard",
        "description": "Leaderboard for a stat: PTS, AST, TRB, STL, BLK, Misses, 3P, etc.",
        "parameters": {
            "type": "object",
            "properties": {
                "season": {"type": "string"},
                "stat": {"type": "string"},
                "min_g": {"type": "integer"},
                "min_mp": {"type": "integer"},
                "limit": {"type": "integer"},
            },
            "required": ["season", "stat"],
        },
    },
    {
        "name": "top_3pt_pct",
        "description": "Top 3-point percentage shooters for a season.",
        "parameters": {
            "type": "object",
            "properties": {"season": {"type": "string"}, "limit": {"type": "integer"}},
            "required": ["season"],
        },
    },
    {
        "name": "compare_players",
        "description": "Compare stats of multiple players in one season. player_ids are from find_players (e.g. duranke01, bryanko01).",
        "parameters": {
            "type": "object",
            "properties": {
                "player_ids": {"type": "array", "items": {"type": "string"}},
                "season": {"type": "string"},
            },
            "required": ["player_ids", "season"],
        },
    },
    {
        "name": "compare_players_by_names",
        "description": "Compare two players by their names (e.g. 'Kevin Durant' vs 'Kobe Bryant'). Use for 'who is better X or Y'. Automatically finds player IDs.",
        "parameters": {
            "type": "object",
            "properties": {
                "name1": {"type": "string"},
                "name2": {"type": "string"},
                "season": {"type": "string"},
            },
            "required": ["name1", "name2"],
        },
    },
    {
        "name": "player_stats_multi_season",
        "description": "Get player stats across multiple seasons. Use for 'last 3 years', 'career stats', '2015-2018'. Returns Season, Team, G, MP, PTS, AST, TRB, STL, BLK, 3P, 3PA, 3P%.",
        "parameters": {
            "type": "object",
            "properties": {
                "player_id": {"type": "string", "description": "Player Reference from find_players"},
                "n_seasons": {"type": "integer", "description": "Number of most recent seasons (e.g. 3 for 'last 3 years')"},
                "season_from": {"type": "string", "description": "Start season (e.g. '2014-2015')"},
                "season_to": {"type": "string", "description": "End season (e.g. '2016-2017')"},
            },
            "required": ["player_id"],
        },
    },
    {
        "name": "player_3pt_by_season",
        "description": "3-point stats by season for one player.",
        "parameters": {
            "type": "object",
            "properties": {
                "player_id": {"type": "string"},
                "season_from": {"type": "string"},
                "season_to": {"type": "string"},
            },
            "required": ["player_id"],
        },
    },
    {
        "name": "get_last_n_games",
        "description": "Last N games for a player (game-by-game stats).",
        "parameters": {
            "type": "object",
            "properties": {"player_id": {"type": "string"}, "n": {"type": "integer"}},
            "required": ["player_id"],
        },
    },
    {
        "name": "team_roster_stats",
        "description": "Get roster stats for a team in specific season(s). Use for queries like 'Celtics roster 2023-24' or 'Lakers players last 3 years'. Returns ALL players who played for the team.",
        "parameters": {
            "type": "object",
            "properties": {
                "team_abbrev": {"type": "string", "description": "Team abbreviation (e.g. 'BOS', 'LAL', 'GSW')"},
                "seasons": {"type": "array", "items": {"type": "string"}, "description": "List of seasons (e.g. ['2022-2023', '2023-2024'])"},
                "stats_mode": {"type": "string", "enum": ["basic", "all"], "description": "Use 'basic' for 12 key stats, 'all' for complete stats (20-30 columns). Default 'all'."},
            },
            "required": ["team_abbrev", "seasons"],
        },
    },
]

ROUTING_SYSTEM = """You are an NBA assistant. For STATS: use tools.
- "Who is better X or Y?" or player comparisons by name → use compare_players_by_names(name1, name2, season).
- "Last N years/seasons" or multi-season player stats → use player_stats_multi_season with n_seasons parameter.
- "Team roster" queries → use team_roster_stats.
- Leaderboards, single player stats → use other tools. Use find_players first if you need player_id.
- News, history, narratives → answer from context, no tools.

CRITICAL: For any stats query, ALWAYS call the appropriate tool first. Then format the tool output as a table."""

# Global storage for DataFrames (for export feature)
_DATAFRAME_STORAGE = {
    "dataframes": [],  # List of {"df": DataFrame, "filename": str, "label": str}
    "enabled": False   # Set to True when export is requested
}


def get_export_dataframes():
    """Get stored DataFrames for export. Called by app.py."""
    return _DATAFRAME_STORAGE["dataframes"]


def clear_export_dataframes():
    """Clear stored DataFrames. Called by app.py after export or on new query."""
    _DATAFRAME_STORAGE["dataframes"] = []


def _get_player_stats_universal(
    player_name: str,
    season: str = None,
    n_seasons: int = None,
    include_current: bool = False,
    season_type: str = "Regular Season",
) -> tuple:
    """
    UNIVERSAL player stats handler. LIVE API FIRST, CSV as fallback.
    
    Logic:
    - "last 2 seasons" → 2024-25 and 2023-24 (past completed, NOT current)
    - "current season" → 2025-26 from live API
    - "2016 stats" → 2015-2016 from live API (career stats has all seasons)
    - "curry stats" → current season from live API
    """
    import pandas as pd
    from src.live_data import get_player_stats_live
    
    result_df = None
    player_full_name = player_name
    source = None
    
    # ── LIVE API (primary source) ──
    try:
        if n_seasons:
            # "Last N seasons" - uses n_last param which excludes current season
            results = get_player_stats_live(player_name, n_last=n_seasons, season_type=season_type)
        elif include_current:
            # "Current season" - get only current
            results = get_player_stats_live(player_name, seasons=None, season_type=season_type)  # Default = current
        elif season:
            # Specific season like "2015-2016" - extract year
            season_year = int(season.split("-")[0])
            results = get_player_stats_live(player_name, seasons=[season_year], season_type=season_type)
        else:
            # General query, no time specified → current season
            results = get_player_stats_live(player_name, seasons=None, season_type=season_type)
        
        if results:
            player_full_name = results[0].get("player_name", player_name)
            # Rename keys to match our DataFrame columns
            rows = []
            for s in results:
                rows.append({
                    "Type": s.get("season_type", season_type),
                    "Season": s["season"],
                    "Team": s["team"],
                    "G": s["G"],
                    "MP": s["MP"],
                    "PTS": s["PTS"],
                    "AST": s["AST"],
                    "TRB": s["TRB"],
                    "STL": s["STL"],
                    "BLK": s["BLK"],
                    "3P": s["3P"],
                    "3PA": s["3PA"],
                    "3P%": s["3P%"],
                })
            result_df = pd.DataFrame(rows)
            source = "nba.com"
    except Exception:
        pass
    
    # ── CSV fallback (only if live API failed) ──
    if result_df is None or result_df.empty:
        try:
            df, _, _ = _load_df()
            from src.nba_data import find_players, player_summary, player_stats_multi_season
            
            found = find_players(df, player_name, limit=1)
            if not found.empty:
                player_full_name = found.iloc[0]["Player"]
                pid = found.iloc[0]["Player Reference"]
                
                if n_seasons:
                    result_df = player_stats_multi_season(df, pid, n_seasons=n_seasons)
                elif season:
                    result_df = player_summary(df, pid, season)
                else:
                    result_df = player_stats_multi_season(df, pid, n_seasons=1)
                
                if result_df is not None and not result_df.empty:
                    source = "csv"
        except Exception:
            pass
    
    return result_df, player_full_name, source


def _call_tool(name: str, args: Dict[str, Any]) -> str:
    from src.nba_data import (
        find_players,
        player_summary,
        top_stat_leaderboard,
        top_3pt_pct,
        compare_players,
        compare_careers,
        player_3pt_by_season,
        player_stats_multi_season,
        team_roster_stats,
        get_last_n_games,
    )
    from src.call_tools import safe_call

    df, df_games, df_sched = _load_df()

    def _compare_by_names():
        n1, n2 = args["name1"], args["name2"]
        use_career = args.get("career", True)
        p1 = find_players(df, n1, limit=1)
        p2 = find_players(df, n2, limit=1)
        if p1.empty or p2.empty:
            return f"Could not find one or both players. Searched for '{n1}' and '{n2}'."
        ids = [p1.iloc[0]["Player Reference"], p2.iloc[0]["Player Reference"]]
        if use_career:
            return compare_careers(df, ids).to_string(index=False)
        return compare_players(df, ids, args.get("season", "2023-2024")).to_string(index=False)

    tools_map = {
        "find_players": lambda: find_players(df, args["query"]),
        "player_summary": lambda: player_summary(df, args["player_id"], args["season"]),
        "top_stat_leaderboard": lambda: top_stat_leaderboard(
            df, args["season"], args["stat"],
            args.get("min_g", 40), args.get("min_mp", 800), args.get("limit", 10),
        ),
        "top_3pt_pct": lambda: top_3pt_pct(df, args["season"], limit=args.get("limit", 10)),
        "compare_players": lambda: compare_players(df, args.get("player_ids", []), args.get("season", "2023-2024")),
        "compare_players_by_names": _compare_by_names,
        "player_3pt_by_season": lambda: player_3pt_by_season(
            df, args["player_id"], args.get("season_from"), args.get("season_to")
        ),
        "player_stats_multi_season": lambda: player_stats_multi_season(
            df, args["player_id"], args.get("season_from"), args.get("season_to"), args.get("n_seasons")
        ),
        "team_roster_stats": lambda: team_roster_stats(
            df, args["team_abbrev"], args["seasons"], args.get("stats_mode", "all")
        ),
        "get_last_n_games": lambda: get_last_n_games(df_games, args["player_id"], args.get("n", 10)) if df_games else "Box score data not available.",
    }

    if name not in tools_map:
        return f"Unknown tool: {name}"
    result = safe_call(tools_map[name])
    if not result["ok"]:
        return f"Error: {result['error']}"
    data = result["data"]
    
    # Store DataFrame for export if it's a DataFrame
    if hasattr(data, "to_string") and not data.empty if hasattr(data, "empty") else True:
        # Store for export
        import pandas as pd
        if isinstance(data, pd.DataFrame):
            # Generate filename based on tool and args
            filename_parts = [name]
            if "team_abbrev" in args:
                filename_parts.append(args["team_abbrev"])
            if "player_id" in args:
                filename_parts.append(args["player_id"].split("0")[0])  # Remove trailing numbers
            if "season" in args:
                filename_parts.append(args["season"].replace("-", "_"))
            if "seasons" in args and args["seasons"]:
                if len(args["seasons"]) == 1:
                    filename_parts.append(args["seasons"][0].replace("-", "_"))
                else:
                    filename_parts.append(f"{len(args['seasons'])}_seasons")
            
            filename = "_".join(filename_parts)
            label = args.get("seasons", [args.get("season", "")])[0] if "seasons" in args else args.get("season", "Data")
            
            _DATAFRAME_STORAGE["dataframes"].append({
                "df": data.copy(),
                "filename": filename,
                "label": label,
                "source": "CSV fallback",
            })
            
        # Return plain text for now - we'll format after Gemini's response
        return data.to_string(index=False)
    return str(data)


def _parse_gemini_args(fc) -> Dict[str, Any]:
    """Convert Gemini function_call args (protobuf Struct) to plain dict."""
    args_dict = {}
    if not hasattr(fc, "args") or not fc.args:
        return args_dict
    try:
        from google.protobuf import json_format
        args_dict = json_format.MessageToDict(fc.args)
    except Exception:
        pass
    if not args_dict:
        try:
            if hasattr(fc.args, "items"):
                args_dict = dict(fc.args)
            elif hasattr(fc.args, "fields"):
                for k, v in fc.args.fields.items():
                    if hasattr(v, "string_value"):
                        args_dict[k] = v.string_value
                    elif hasattr(v, "number_value"):
                        args_dict[k] = int(v.number_value) if v.number_value == int(v.number_value) else v.number_value
                    elif hasattr(v, "list_value") and v.list_value:
                        args_dict[k] = [x.string_value for x in v.list_value.values]
        except Exception:
            pass
    return args_dict


def _extract_season_from_query(query: str) -> Optional[str]:
    """Extract season (e.g. 2023-2024) if user mentions a year. Else None = use career."""
    q = query
    # "2023-2024" or "2023-24"
    m = re.search(r"(20\d{2})[- ](20?\d{2})", q)
    if m:
        y1, y2 = m.group(1), m.group(2)
        if len(y2) <= 2:
            y2 = y1[:2] + y2.zfill(2)
        return f"{y1}-{y2}"
    # "in 2024", "2024 season", "during 2016" — NBA season named by end year
    m = re.search(r"(?:in|during|for|season)?\s*(20\d{2})\b", q)
    if m:
        year = int(m.group(1))
        if 1990 <= year <= 2030:
            return f"{year - 1}-{year}"
    return None


def _strip_season_from_name(name: str) -> str:
    """Remove 'in 2024', 'during 2016', '2023-2024' etc. from player name."""
    return re.sub(r"\s+(?:in|during|for)\s+20\d{2}(?:-\d{2,4})?\b", "", name, flags=re.I).strip()


def _extract_compare_players(query: str) -> Optional[tuple]:
    """If query is 'who is better X or Y' or 'compare X vs Y', return (name1, name2, season_or_none). season=None means career."""
    q = query.lower().strip().rstrip("?")
    season = _extract_season_from_query(query)
    
    # Normalize separators: "vs", "versus", "and" -> "or"
    q_normalized = q.replace(" vs ", " or ").replace(" versus ", " or ").replace(" and ", " or ")
    
    prefixes = ("who is better ", "who's better ", "compare ", "who was better ")
    rest = None
    for p in prefixes:
        if q_normalized.startswith(p):
            rest = q_normalized[len(p):].strip()
            break
    
    if rest is None:
        if " or " in q_normalized and ("better" in q_normalized or "compare" in q_normalized):
            parts = q_normalized.split(" or ", 1)
            if len(parts) == 2:
                n1 = _strip_season_from_name(parts[0].replace("who is better", "").replace("who's better", ""))
                n2 = _strip_season_from_name(parts[1])
                if n1 and n2 and len(n1) > 2 and len(n2) > 2:
                    return (n1, n2, season)
        return None
    
    if " or " in rest:
        parts = rest.split(" or ", 1)
        n1, n2 = _strip_season_from_name(parts[0]), _strip_season_from_name(parts[1])
        # Remove trailing words like "career stats", "stats"
        n1 = re.sub(r"\s*(career|stats|statistics).*$", "", n1).strip()
        n2 = re.sub(r"\s*(career|stats|statistics).*$", "", n2).strip()
        if n1 and n2 and len(n1) > 2 and len(n2) > 2:
            return (n1, n2, season)
    return None


def _is_schedule_query(query: str) -> bool:
    """Next match, schedule, who is playing, upcoming games/matches, tomorrow, today."""
    q = query.lower()
    return any(w in q for w in [
        "next match", "next game", "next matches", "next games", "matches", "upcoming",
        "schedule", "who is playing", "who are they playing", "playing next", "next opponent",
        "who are they playing with", "who do they play",
        "tomorrow", "today's game", "today game",
        "games of", "games for", "next ", " upcoming",
        "game between", "game with",
    ])


def _resolve_team(query: str) -> Optional[str]:
    """Extract team abbrev from query (e.g. gsw, golden state -> GSW)."""
    from src.nba_data import TEAM_ABBREVS
    q = query.lower()
    for hint, abbr in TEAM_ABBREVS.items():
        if hint in q:
            return abbr
    return None


def _extract_schedule_n(query: str) -> int:
    """Extract number of games from 'next 10 matches', 'next 5 games', etc. Default 10."""
    m = re.search(r"next\s+(\d+)\s*(?:matches?|games?)", query.lower())
    if m:
        n = int(m.group(1))
        return min(max(n, 1), 20)
    return 10


def _extract_target_date(query: str) -> Optional[str]:
    """Extract specific date from 'tomorrow', 'today', '2/28/2026', etc. Returns YYYY-MM-DD."""
    from datetime import datetime, timedelta
    q = query.lower()
    if "tomorrow" in q:
        d = datetime.now().date() + timedelta(days=1)
        return d.strftime("%Y-%m-%d")
    if "today" in q:
        return datetime.now().date().strftime("%Y-%m-%d")
    # 2/28/2026, 02-28-2026
    m = re.search(r"(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})", query)
    if m:
        mo, day, yr = m.group(1), m.group(2), m.group(3)
        yr = "20" + yr if len(yr) == 2 else yr
        try:
            d = datetime(int(yr), int(mo), int(day))
            return d.strftime("%Y-%m-%d")
        except ValueError:
            pass
    return None


def _resolve_team_from_history(history: List[dict]) -> Optional[str]:
    """From recent USER messages only, find last mentioned team for 'they'/'them' follow-ups."""
    from src.nba_data import TEAM_ABBREVS
    hints_sorted = sorted(TEAM_ABBREVS.items(), key=lambda x: -len(x[0]))
    for msg in reversed(history[-6:]):
        if msg.get("role") != "user":
            continue
        content = (msg.get("content") or "").lower()
        for hint, abbr in hints_sorted:
            if len(hint) > 2 and hint in content:
                return abbr
    return None


def _is_opinion_only_query(query: str) -> bool:
    """Questions that need Gemini's general knowledge, no tools (avoids tool-call bugs)."""
    q = query.lower()
    return any(w in q for w in ["goat", "greatest", "best player ever", "best nba player", "best basketball player"])


def _is_news_query(query: str) -> bool:
    """News today - use Gemini directly, not RAG (RAG has stale content)."""
    q = query.lower()
    return any(w in q for w in ["news today", "any nba news", "latest news", "nba news", "today's news"])


def _is_rag_query(query: str) -> bool:
    q = query.lower()
    if _is_news_query(query):
        return False  # News goes to direct Gemini
    triggers = [
        "what happened", "tell me about", "story", "finals", "championship",
        "ray allen", "michael jordan", "last shot", "comeback", "dynasty",
        "history", "recap", "game 7", "game 6", "rings", "titles", "won",
        "career", "legacy", "greatest", "goat", "hall of fame",
    ]
    return any(t in q for t in triggers)


def _extract_player_and_season(query: str) -> Optional[tuple]:
    """Extract player name and season from queries like 'curry 2021 stats', 'lebron 2021 season stats'."""
    q = query.lower().strip().rstrip("?")
    q = re.sub(r"'s\b", "", q)  # Remove possessives
    
    # "player 2021 stats" or "player 2021 season stats"
    m = re.search(r"(.+?)\s+(20\d{2})\s*[-]?\s*(?:season\s+)?(?:stats|statistics?)", q)
    if m:
        return (m.group(1).strip(), f"{int(m.group(2))-1}-{m.group(2)}")
    # "2021 stats for/of player"
    m = re.search(r"(20\d{2})\s*[-]?\s*(?:season\s+)?(?:stats|statistics?)\s+(?:for|of)\s+(.+)", q)
    if m:
        return (m.group(2).strip(), f"{int(m.group(1))-1}-{m.group(1)}")
    # "player 2020-2021" explicit range
    m = re.search(r"(.+?)\s+(20\d{2})\s*[-]\s*(20\d{2})", q)
    if m:
        return (m.group(1).strip(), f"{m.group(2)}-{m.group(3)}")
    # Just "player 2021" with no "stats" keyword
    m = re.search(r"(.+?)\s+(20\d{2})(?:\s+season)?$", q)
    if m:
        name = m.group(1).strip()
        if name and len(name) > 2:
            return (name, f"{int(m.group(2))-1}-{m.group(2)}")
    return None


def _extract_debut_query(query: str) -> Optional[str]:
    """If 'when did X debut', return player name."""
    q = query.lower().replace("?", "").strip()
    if "debut" not in q and "first season" not in q:
        return None
    m = re.search(r"when\s+(?:did|was)\s+(.+?)\s+debut", q)
    if m:
        name = m.group(1).replace("in the nba", "").replace("in nba", "").strip()
        if len(name) > 2:
            return name
    m = re.search(r"debut\s+(?:of|for)\s+(.+)", q)
    if m:
        return m.group(1).strip()
    return None


def _is_live_web_query(query: str) -> bool:
    """Queries that need real-time web search: leaders, standings, news, playoffs, current info."""
    q = query.lower()
    
    # IMPORTANT: Player stats queries should NOT go to web search
    # They should use our live NBA API instead
    stats_indicators = ["stats", "statistics", "points", "assists", "rebounds", "average", "averaging"]
    is_player_stats = any(w in q for w in stats_indicators)
    
    # If it's a player stats query with "current/this season", don't use web search
    # Our live API handles this better
    if is_player_stats and ("current season" in q or "this season" in q):
        return False
    
    # Check simple triggers first
    simple_triggers = [
        "leading", "standings", "who's leading", "top scorer", "leaders",
        "what's going on", "currently", "latest news", "nba news", "news today",
        "playoff", "playoffs", "finals date", "when is the final", "when are the finals",
        "current record", "right now",
        "mvp race", "mvp candidates", "all-star", "trade deadline",
        "injured", "injury report", "who got traded", "trade news",
        "all time", "all-time", "in history",
    ]
    # Removed "current season" and "this season" - handled by our live API for stats
    if any(w in q for w in simple_triggers):
        return True
    
    # Check "most" + achievement patterns (handles "most nba championships", "most rings", etc.)
    if "most" in q and any(w in q for w in ["championship", "ring", "title", "mvp", "award", "win"]):
        return True
    
    # Check "who has/won" patterns
    if ("who has" in q or "who won" in q or "who got" in q) and any(w in q for w in ["most", "championship", "ring", "title", "mvp"]):
        return True
    
    return False


def _needs_factual_answer(query: str) -> bool:
    """Queries asking for specific facts that Google Search can answer."""
    q = query.lower()
    return any(w in q for w in [
        "how many", "how much", "when did", "where did", "who won",
        "championships", "rings", "titles", "mvp", "awards", "records",
        "drafted", "traded", "signed", "contract", "salary",
    ])


def _chat_with_google_search(query: str, history: Optional[List[dict]], api_key: str) -> str:
    """Use Gemini with Google Search grounding for live web info."""
    from datetime import datetime
    today = datetime.now().strftime("%B %d, %Y")
    
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        
        # Build explicit prompt that forces web search usage
        context = ""
        if history and len(history) > 0:
            recent = "\n".join(
                f"{'User' if m.get('role')=='user' else 'Assistant'}: {m.get('content','')}"
                for m in history[-4:]
            )
            context = f"Recent conversation:\n{recent}\n\n"
        
        # Analyze conversation context to understand if talking about players or teams
        context_hint = ""
        if history and len(history) > 0:
            recent_text = " ".join([m.get("content", "") for m in history[-4:]]).lower()
            # Check if recent conversation is about teams
            team_words = ["team", "celtics", "lakers", "bulls", "warriors", "heat", "cavaliers", "nuggets", "franchise", "organization"]
            player_words = ["player", "he ", "his ", "curry", "lebron", "jordan", "durant", "russell", "kobe"]
            
            team_count = sum(1 for w in team_words if w in recent_text)
            player_count = sum(1 for w in player_words if w in recent_text)
            
            if team_count > player_count:
                context_hint = "Based on the conversation, the user is asking about TEAMS."
            elif player_count > team_count:
                context_hint = "Based on the conversation, the user is asking about PLAYERS."
        
        # Very explicit instructions to use Google Search and return current data
        prompt = f"""{context}Today is {today}. The user is asking about NBA information.

User question: {query}

{context_hint}

INSTRUCTIONS:
1. You MUST use Google Search to find the answer. Do NOT use your training data.
2. Include actual numbers/records you find (e.g., "43-14", "11 championships").
3. If you find conflicting info, use the most recent source (NBA.com, ESPN, etc).
4. Be concise - just answer the question with facts.
5. Context rules for "who has most championships" type questions:
   - If conversation context is about TEAMS → answer about teams (Celtics: 18)
   - If conversation context is about PLAYERS → answer about players (Bill Russell: 11)
   - If no clear context → answer about players first (since "who" usually means a person)
   - If user asks "which team" or "what team" → always answer about teams

Search the web and answer:"""

        resp = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config={"tools": [{"google_search": {}}]},
        )
        
        text = (resp.text or "").strip()
        if not text:
            return "I couldn't get live results. Check nba.com/standings for current standings."
        
        # If response seems like it's not using search (apologetic, no numbers), warn user
        if any(phrase in text.lower() for phrase in ["i don't have", "i cannot access", "my training data", "i apologize"]):
            return f"{text}\n\n(For the most accurate info, visit nba.com/standings)"
        
        return text
        
    except ImportError:
        return "Install google-genai for live web search: pip install google-genai"
    except Exception as e:
        err = str(e)
        if not err or err in ("object", "'object'", "Object"):
            return "Live search failed. Check nba.com/standings for current standings."
        return f"Error: {err}"


def _get_system_instruction() -> str:
    from datetime import datetime
    now = datetime.now()
    today = now.strftime("%Y-%m-%d")
    year = now.year
    season = f"{year}-{str(year+1)[-2:]}" if now.month >= 10 else f"{year-1}-{str(year)[-2:]}"
    return f"""You are an NBA assistant. Today: {today}. Season: {season}.

CRITICAL OUTPUT RULES:
1. Write ONLY a brief 1-2 sentence summary of the data.
2. DO NOT create tables - the app will add tables automatically.
3. DO NOT use <details> tags - the app handles this.
4. DO NOT list stats in bullet points.
5. Just summarize the key findings in plain text.

GOOD RESPONSE EXAMPLE:
"Stephen Curry's stats for the 2024-25 and 2025-26 seasons show he averaged 27 points per game with GSW."

BAD RESPONSE (DO NOT DO THIS):
"Here are the stats:
| Season | PTS | AST |
|--------|-----|-----|
| 2025 | 1000 | 300 |"

RULES:
- Use the data provided below. It is factual.
- Keep responses concise - one or two sentences.
- The detailed table will be shown automatically below your response.
- If data has "Source: live_api", it's current season data from NBA.com.

Example responses:
- "LeBron James scored 1,500 points in the 2024-25 season with the Lakers."
- "The comparison shows Durant has higher PPG (27.3) than LeBron (27.2), but LeBron has more career points."
- "The Warriors' next 3 games are against the Lakers, Celtics, and Heat."

DO NOT CREATE TABLES. JUST SUMMARIZE IN 1-2 SENTENCES."""


def _safe_gemini_text(resp) -> str:
    """Safely extract text from Gemini response."""
    try:
        if hasattr(resp, "text") and resp.text:
            return str(resp.text).strip()
        if hasattr(resp, "candidates") and resp.candidates:
            c = resp.candidates[0]
            if hasattr(c, "content") and c.content and hasattr(c.content, "parts"):
                for p in c.content.parts:
                    if hasattr(p, "text") and p.text:
                        return str(p.text).strip()
    except Exception:
        pass
    return ""


def _gather_context(query: str, history: Optional[List[dict]]) -> str:
    """Fetch relevant data (schedule, stats, RAG, etc.) based on query. Returns context string for the LLM."""
    ctx_parts = []
    hist = history or []

    # Schedule
    is_schedule = _is_schedule_query(query)
    if not is_schedule and hist:
        last = next((m for m in reversed(hist[-4:]) if m.get("role") == "assistant"), None)
        team = _resolve_team(query)
        if team and last and ("@" in (last.get("content") or "") or "schedule" in (last.get("content") or "").lower()):
            is_schedule = True
    if is_schedule:
        team_abbr = _resolve_team(query) or _resolve_team_from_history(hist)
        if team_abbr:
            target_date = _extract_target_date(query)
            n = _extract_schedule_n(query)
            try:
                from src.live_data import get_live_schedule_text
                if target_date:
                    single = get_live_schedule_text(team_abbr, n=1, query=query, target_date=target_date)
                    if single:
                        ctx_parts.append(f"[Schedule data for {team_abbr} on {target_date}]\n{single}")
                    else:
                        ctx_parts.append(f"[No game found for {team_abbr} on {target_date}]")
                else:
                    live = get_live_schedule_text(team_abbr, n=n, query=query)
                    if live:
                        ctx_parts.append(f"[Schedule data for {team_abbr}]\n{live}")
                    else:
                        _, _, df_sched = _load_df()
                        if df_sched is not None:
                            from src.nba_data import get_team_games
                            games = get_team_games(df_sched, team_abbr, n=n)
                            if not games.empty:
                                ctx_parts.append(f"[Past games for {team_abbr}]\n{games.to_string(index=False)}")
            except Exception:
                pass

    # News (when not using Google Search—fallback)
    if _is_news_query(query):
        try:
            from src.rag import retrieve
            chunks = retrieve(RAG_DOCS_DIR, query, top_k=4)
            if chunks:
                ctx_parts.append("[NBA news from our sources]\n" + "\n\n".join(chunks))
        except Exception:
            pass

    # Debut
    player_debut = _extract_debut_query(query)
    if player_debut:
        try:
            df, _, _ = _load_df()
            from src.nba_data import find_players, get_player_debut
            found = find_players(df, player_debut, limit=1)
            if not found.empty:
                pid, pname = found.iloc[0]["Player Reference"], found.iloc[0]["Player"]
                first = get_player_debut(df, pid)
                if first:
                    ctx_parts.append(f"[Debut info]\n{pname} debuted in the {first} NBA season.")
        except Exception:
            pass

    # =============================================================================
    # EMBEDDING-BASED QUERY PARSER (replaces regex)
    # Uses sentence embeddings to classify intent + extract entities
    # =============================================================================
    
    from src.query_parser import parse_query
    
    parsed = parse_query(query, history=hist)
    intent = parsed["intent"]
    pname = parsed["player_name"]
    
    # Handle player stats intents
    if intent.startswith("player_stats") and pname:
        q_lower = query.lower()
        is_playoffs = any(w in q_lower for w in ["playoff", "playoffs", "postseason"])
        season_type = "Post Season" if is_playoffs else "Regular Season"
        n_seasons = parsed["n_seasons"]
        season_year = parsed["season_year"]
        season = None
        label = ""
        include_current = False
        
        if intent == "player_stats_multi_season" and n_seasons:
            label = f"Last {n_seasons} Seasons ({'Playoffs' if is_playoffs else 'Regular Season'})"
            include_current = False
        elif intent == "player_stats_current":
            label = f"Current Season ({'Playoffs' if is_playoffs else 'Regular Season'})"
            include_current = True
        elif intent == "player_stats_single_season" and season_year:
            season = f"{season_year - 1}-{season_year}"
            label = f"{season} ({'Playoffs' if is_playoffs else 'Regular Season'})"
            include_current = False
        else:
            # General stats query = current season
            label = f"Current Season ({'Playoffs' if is_playoffs else 'Regular Season'})"
            include_current = True
        
        result_df, player_full_name, source = _get_player_stats_universal(
            pname, season=season, n_seasons=n_seasons, include_current=include_current, season_type=season_type
        )
        
        if result_df is not None and not result_df.empty:
            safe_name = player_full_name.lower().replace(" ", "_")
            _DATAFRAME_STORAGE["dataframes"].append({
                "df": result_df.copy(),
                "filename": f"player_stats_{safe_name}_{label.replace(' ', '_')}",
                "label": label,
                "source": "NBA.com" if source == "nba.com" else "CSV fallback",
            })
            
            source_note = f" (Source: {source})" if source else ""
            ctx_parts.append(f"[Stats for {player_full_name} - {label}{source_note}]\n{result_df.to_string(index=False)}")

    # Comparison
    compare_names = _extract_compare_players(query)
    if compare_names:
        name1, name2, season = compare_names[0], compare_names[1], compare_names[2] if len(compare_names) > 2 else None
        use_career = season is None
        try:
            result = _call_tool("compare_players_by_names", {
                "name1": name1, "name2": name2, "career": use_career,
                "season": season or "2023-2024",
            })
            if "Could not find" not in result:
                label = "Career totals" if use_career else f"{season} stats"
                ctx_parts.append(f"[{label} for {name1} vs {name2}]\n{result}")
        except Exception:
            pass

    # RAG (narrative / history) - if no other context
    if _is_rag_query(query) and not ctx_parts:
        try:
            from src.rag import retrieve
            chunks = retrieve(RAG_DOCS_DIR, query, top_k=4)
            if chunks:
                ctx_parts.append("[NBA history and bios]\n" + "\n\n".join(chunks))
        except Exception:
            pass

    return "\n\n".join(ctx_parts) if ctx_parts else ""


def _chat_conversational(
    query: str,
    history: Optional[List[dict]],
    context: str,
    api_key: str,
) -> str:
    """Single conversational response using Gemini with full chat history."""
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            "gemini-2.0-flash",
            system_instruction=_get_system_instruction(),
        )
        genai_history = []
        for m in (history or [])[-12:]:
            role = "user" if m.get("role") == "user" else "model"
            content = (m.get("content") or "").strip()
            if content:
                genai_history.append({"role": role, "parts": [{"text": content}]})
        current_text = query
        if context.strip():
            # Add reminder to use table format when data is present
            has_stats_data = any(word in context.lower() for word in ['season', 'team', 'player', 'pts', 'ast', 'games'])
            reminder = "\n\n⚠️ REMEMBER: Format the data above as a summary + expandable table. See system instruction examples." if has_stats_data else ""
            current_text = f"[Data for this response]\n{context.strip()}{reminder}\n\n[User says]\n{query}"
        if not genai_history:
            resp = model.generate_content(current_text)
        else:
            chat = model.start_chat(history=genai_history)
            resp = chat.send_message(current_text)
        out = _safe_gemini_text(resp)
        return out or "I'm not sure how to answer that. Try asking about a team's schedule, player stats, or NBA history!"
    except Exception as e:
        err = str(e)
        if not err or err in ("object", "'object'", "Object"):
            return "Something went wrong on my end. Give it another try?"
        return f"Error: {err}"


def run(query: str, use_llm: bool = True, history: Optional[List[dict]] = None) -> str:
    # Clear any stored DataFrames from previous query
    clear_export_dataframes()
    
    api_key = _get_api_key()
    if not api_key and use_llm:
        return "Add GOOGLE_API_KEY or GEMINI_API_KEY to your .env file. Get a free key at https://aistudio.google.com/apikey"

    # Use embedding parser to check if this is a stats query (bypass web search for those)
    try:
        from src.query_parser import classify_intent
        intent, _ = classify_intent(query)
        is_stats_query = intent.startswith("player_stats") or intent == "team_roster_stats"
    except Exception:
        is_stats_query = False
    
    # Live web queries: use Gemini with Google Search - but NOT for stats queries
    if not is_stats_query and _is_live_web_query(query) and use_llm and api_key:
        return _chat_with_google_search(query, history, api_key)

    # Comparison hard fail: pass to chat for a friendly "couldn't find" message
    compare_names = _extract_compare_players(query)
    if compare_names:
        try:
            r = _call_tool("compare_players_by_names", {
                "name1": compare_names[0], "name2": compare_names[1],
                "career": compare_names[2] is None, "season": compare_names[2] or "2023-2024",
            })
            if "Could not find" in r:
                return _chat_conversational(query, history, r, api_key)
        except Exception:
            pass

    # Gather context and respond conversationally (single flow for all queries)
    context = _gather_context(query, history)
    
    # If no context found from our data sources, use Google Search as fallback
    # This handles ANY question we don't have data for
    if not context.strip() and api_key:
        return _chat_with_google_search(query, history, api_key)
    
    if not use_llm or not api_key:
        return context or "Add GOOGLE_API_KEY to .env for full answers."
    return _chat_conversational(query, history, context, api_key)
