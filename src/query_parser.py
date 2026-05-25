"""
Query parser using sentence embeddings instead of regex.
Uses the same SentenceTransformer (all-MiniLM-L6-v2) already loaded for RAG.

Two jobs:
1. INTENT CLASSIFICATION - What does the user want?
   (player_stats, multi_season, current_season, comparison, schedule, debut, etc.)
2. ENTITY EXTRACTION - Who/what are they asking about?
   (player name, season year, number of seasons, team name)
"""
import re
from typing import Optional, Dict, Any
import numpy as np

_model = None
_intent_embeddings = None

# Intent examples - the model compares user query against these
INTENT_EXAMPLES = {
    "player_stats_single_season": [
        "curry 2021 stats",
        "lebron james 2019 season statistics",
        "give me kevin durant 2018 stats",
        "show me harden's 2020 season",
        "what were giannis stats in 2023",
        "kobe 2006 numbers",
        "dame lillard stats from 2021",
    ],
    "player_stats_multi_season": [
        "curry last two seasons stats",
        "lebron's last 3 years",
        "give me last 5 seasons of durant",
        "show steph curry stats for last 2 years",
        "past three seasons of jokic",
        "previous 2 seasons stats of tatum",
        "luka last four year statistics",
    ],
    "player_stats_current": [
        "curry current season stats",
        "lebron this season",
        "how is durant doing this year",
        "giannis stats this season",
        "what are curry's numbers this year",
        "show me jokic current season performance",
        "tatum 2025-2026 stats",
    ],
    "player_stats_general": [
        "curry stats",
        "tell me about lebron's stats",
        "how good is jokic",
        "show me durant numbers",
        "give me luka stats",
        "what are steph's statistics",
    ],
    "player_comparison": [
        "compare lebron vs durant",
        "curry vs dame stats",
        "who is better lebron or jordan",
        "lebron james compared to kevin durant",
        "giannis vs jokic career stats",
    ],
    "team_schedule": [
        "warriors next game",
        "when do lakers play",
        "celtics schedule",
        "upcoming games for bucks",
        "next 3 games of heat",
    ],
    "player_debut": [
        "when did curry debut",
        "lebron's first season",
        "when did durant start playing",
        "rookie year of tatum",
    ],
    "team_roster_stats": [
        "celtics roster stats",
        "lakers team stats 2024",
        "warriors roster this season",
        "give me all players stats from bucks",
    ],
    "general_nba": [
        "who has the most championships",
        "nba standings",
        "who is leading the league",
        "mvp race",
        "latest nba news",
        "playoff schedule",
    ],
    "team_season_stats": [
        "gsw 2023-24 full team stats",
        "celtics roster statistics 2022-23",
        "warriors all players stats this season",
        "give me lakers complete team stats",
        "nuggets 2023 full season stats",
        "download heat roster stats",
        "bulls 2024 season statistics all players",
        "show me spurs team numbers",
    ],
    "player_gamelog": [
        "curry game by game stats 2024",
        "lebron game log this season",
        "giannis every game played 2024-25",
        "show me each game tatum played",
        "durant game by game stats",
        "luka game log last 20 games",
        "all games jokic played this year",
    ],
    "draft_class": [
        "2003 nba draft",
        "who was drafted by the warriors 2022",
        "first overall pick 2019 draft",
        "nba draft class 2017",
        "who did lakers draft in 2010",
        "draft history 2008",
        "lebron draft year picks",
    ],
    "player_bio": [
        "tell me about stephen curry",
        "lebron james background info",
        "how tall is giannis",
        "where did kobe go to college",
        "when was jordan drafted",
        "curry jersey number",
        "where is wemby from",
        "what position does luka play",
    ],
    "advanced_stats": [
        "curry advanced stats this season",
        "lebron efficiency metrics 2024",
        "giannis PER 2024-25",
        "best true shooting percentage leaders",
        "highest usage rate players",
        "top offensive rating nba",
        "net rating leaders this season",
        "player impact estimate top 20",
    ],
    "league_stats": [
        "all nba player stats 2023-24",
        "full league player statistics download",
        "every player stats this season",
        "download all players scoring data",
        "complete nba stats dataset 2024",
        "all players points rebounds assists",
        "full league per game stats",
    ],
}

# Number words mapping
WORD_TO_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "1": 1, "2": 2, "3": 3, "4": 4, "5": 5,
    "6": 6, "7": 7, "8": 8, "9": 9, "10": 10,
}


def _get_model():
    global _model
    if _model is None:
        from src.rag import _get_embedding_model
        _model = _get_embedding_model()
    return _model


def _get_intent_embeddings():
    """Pre-compute embeddings for all intent examples (cached)."""
    global _intent_embeddings
    if _intent_embeddings is not None:
        return _intent_embeddings
    
    model = _get_model()
    _intent_embeddings = {}
    
    for intent, examples in INTENT_EXAMPLES.items():
        _intent_embeddings[intent] = model.encode(examples)
    
    return _intent_embeddings


def classify_intent(query: str) -> tuple:
    """
    Classify query intent using cosine similarity with intent examples.
    Returns (intent_name, confidence_score).
    """
    model = _get_model()
    intent_embs = _get_intent_embeddings()
    
    query_emb = model.encode([query])[0]
    
    best_intent = "general_nba"
    best_score = 0.0
    
    for intent, embeddings in intent_embs.items():
        # Cosine similarity with each example, take max
        similarities = np.dot(embeddings, query_emb) / (
            np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_emb)
        )
        max_sim = float(np.max(similarities))
        
        if max_sim > best_score:
            best_score = max_sim
            best_intent = intent
    
    return best_intent, best_score


def extract_player_name(query: str, history: list = None) -> Optional[str]:
    """
    Extract player name from query. Uses simple NLP instead of regex patterns.
    Falls back to chat history for pronouns.
    """
    # Normalize
    q = re.sub(r"'s\b", "", query)
    q = re.sub(r"[^\w\s]", " ", q)
    q = " ".join(q.split())
    
    stop_words = {
        'the', 'a', 'an', 'last', 'stats', 'statistics', 'of', 'for', 'give', 'me',
        'can', 'you', 'show', 'current', 'this', 'season', 'seasons', 'get', 'what',
        'are', 'two', 'three', 'four', 'five', 'year', 'years', 'in', 'and', 'or',
        'how', 'many', 'tell', 'about', 'from', 'his', 'her', 'their', 'he', 'she',
        'compare', 'vs', 'versus', 'compared', 'to', 'with', 'past', 'previous',
        'next', 'upcoming', 'when', 'did', 'was', 'is', 'has', 'have', 'do', 'does',
        'team', 'roster', 'game', 'games', 'play', 'playing', 'played', 'number',
        'numbers', 'performance', 'scoring', 'points', 'assists', 'rebounds', 'debut',
        'first', 'career', 'all', 'time', 'record', 'best', 'worst', 'most', 'least',
        'playoff', 'playoffs', 'postseason', 'post', 'seasonal',
        'not', 'but', 'just', 'only', 'also', 'please', 'thanks', 'thank', 'hey',
        'hello', 'hi', 'sir', 'that', 'these', 'those', 'much', 'well', 'good',
    }
    
    # Remove year numbers so they don't interfere
    cleaned = re.sub(r'\b20\d{2}\b', '', q).strip()
    cleaned = re.sub(r'\b\d+\b', '', cleaned).strip()
    cleaned = " ".join(cleaned.split())
    
    # Extract candidate names: sequences of non-stop words
    words = cleaned.split()
    candidates = []
    current_name = []
    
    for word in words:
        if word.lower() not in stop_words and len(word) > 1:
            current_name.append(word)
        else:
            if current_name:
                candidates.append(" ".join(current_name))
                current_name = []
    if current_name:
        candidates.append(" ".join(current_name))
    
    # Return the longest candidate (likely full name)
    if candidates:
        candidates.sort(key=len, reverse=True)
        return candidates[0]
    
    # Pronoun resolution from history
    if history:
        for msg in reversed(history[-6:]):
            content = msg.get("content", "")
            # Look for "FirstName LastName" pattern in recent messages
            name_match = re.search(r'\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b', content)
            if name_match:
                candidate = name_match.group(1)
                if candidate.lower() not in stop_words:
                    return candidate
    
    return None


def extract_season_year(query: str) -> Optional[int]:
    """
    Extract season END year from query.
    '2023-2024' → 2024, '2023-24' → 2024, '2024' → 2024.
    """
    # Full range "2023-2024"
    m = re.search(r'\b(20\d{2})-(20\d{2})\b', query)
    if m:
        return int(m.group(2))
    # Short range "2023-24"
    m = re.search(r'\b(20\d{2})-(\d{2})\b', query)
    if m:
        return int(m.group(1)[:2] + m.group(2))
    # Single year "2024"
    m = re.search(r'\b(20\d{2})\b', query)
    if m:
        return int(m.group(1))
    return None


def extract_team_abbrev(query: str) -> Optional[str]:
    """Extract team abbreviation from query text."""
    try:
        from src.nba_data import TEAM_ABBREVS
        q = query.lower()
        for alias, abbrev in sorted(TEAM_ABBREVS.items(), key=lambda x: -len(x[0])):
            if len(alias) > 2 and alias in q:
                return abbrev
        # Also check raw 3-letter codes
        for code in ["GSW", "LAL", "BOS", "CHI", "MIA", "NYK", "BKN", "PHX",
                     "DEN", "MIL", "PHI", "CLE", "DAL", "MEM", "MIN", "SAC",
                     "NOP", "OKC", "ORL", "POR", "SAS", "TOR", "UTA", "WAS",
                     "ATL", "CHA", "DET", "HOU", "IND", "LAC"]:
            if code.lower() in q.split() or code.lower() + " " in q:
                return code
    except Exception:
        pass
    return None


def extract_draft_year(query: str) -> Optional[int]:
    """Extract NBA draft year from queries like '2003 draft' or 'draft class 2017'."""
    q = query.lower()
    m = re.search(r'(\d{4})\s+(?:nba\s+)?draft', q)
    if m:
        return int(m.group(1))
    m = re.search(r'draft(?:\s+class)?(?:\s+of)?\s+(\d{4})', q)
    if m:
        return int(m.group(1))
    return None


def extract_n_games(query: str) -> Optional[int]:
    """Extract number of games: 'last 10 games' → 10."""
    q = query.lower()
    m = re.search(r'(?:last|past|recent)\s+(\d+)\s+games?', q)
    if m:
        return int(m.group(1))
    return None


def extract_n_seasons(query: str) -> Optional[int]:
    """Extract number of seasons from query (e.g., 2 from 'last two seasons')."""
    q = query.lower()
    m = re.search(r'last\s+(\w+)\s+(?:seasons?|years?)', q)
    if m:
        return WORD_TO_NUM.get(m.group(1), None)
    m = re.search(r'past\s+(\w+)\s+(?:seasons?|years?)', q)
    if m:
        return WORD_TO_NUM.get(m.group(1), None)
    m = re.search(r'previous\s+(\w+)\s+(?:seasons?|years?)', q)
    if m:
        return WORD_TO_NUM.get(m.group(1), None)
    return None


def parse_query(query: str, history: list = None) -> Dict[str, Any]:
    """
    Main entry point. Parses a user query into structured intent + entities.

    Returns dict with:
        intent: str
        confidence: float (0-1)
        player_name: Optional[str]
        season_year: Optional[int] (end year, e.g. 2024 for 2023-24)
        n_seasons: Optional[int]
        team_abbrev: Optional[str]
        draft_year: Optional[int]
        n_games: Optional[int]
        raw_query: str
    """
    intent, confidence = classify_intent(query)
    player_name = extract_player_name(query, history)
    season_year = extract_season_year(query)
    n_seasons = extract_n_seasons(query)
    team_abbrev = extract_team_abbrev(query)
    draft_year = extract_draft_year(query)
    n_games = extract_n_games(query)

    q_lower = query.lower()

    # Hard keyword overrides (highest priority)
    advanced_kws = ["advanced stats", "advanced metric", "per ", "true shooting",
                    "usage rate", "net rating", "off rating", "def rating",
                    "offensive rating", "defensive rating", "pie score", "ts%", "usg%"]
    if any(kw in q_lower for kw in advanced_kws):
        intent = "advanced_stats"

    # Auto-correct intent based on extracted entities
    elif draft_year:
        intent = "draft_class"
    elif n_seasons and intent not in ("player_stats_multi_season",):
        intent = "player_stats_multi_season"
    elif season_year and intent not in (
        "player_stats_single_season", "player_stats_multi_season",
        "team_season_stats", "player_gamelog", "league_stats", "draft_class",
    ):
        intent = "player_stats_single_season"

    # "current season" / "this year" / "this season" → player_stats_current
    if (intent in ("player_stats_general",) and
            any(w in q_lower for w in ["current", "this season", "this year", "now", "today"])):
        intent = "player_stats_current"

    # If team found + stats keyword but no strong player intent → team season stats
    if (team_abbrev and not intent.startswith("player_stats") and
            intent not in ("team_season_stats", "team_schedule", "draft_class", "advanced_stats") and
            any(w in q_lower for w in ["stats", "statistics", "numbers", "roster", "season", "players"])):
        intent = "team_season_stats"

    return {
        "intent": intent,
        "confidence": confidence,
        "player_name": player_name,
        "season_year": season_year,
        "n_seasons": n_seasons,
        "team_abbrev": team_abbrev,
        "draft_year": draft_year,
        "n_games": n_games,
        "raw_query": query,
    }
