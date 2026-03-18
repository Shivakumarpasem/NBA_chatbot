"""
Fetch NBA content for RAG. Run: python scripts/fetch_nba_content.py

Data flow: Wikipedia + news pages -> nba_recaps.txt -> RAG indexes it.
Daily NBA news: RAG goes through pages that post NBA news (configured in NEWS_SOURCES).

To add a new news source: add to NEWS_SOURCES below. See docs/DATA_SOURCES.md.
"""
import re
import time
from pathlib import Path
from typing import List, Optional

import requests

ROOT = Path(__file__).resolve().parents[1]
RAG_DIR = ROOT / "data" / "rag_docs"
RAG_DIR.mkdir(parents=True, exist_ok=True)

USER_AGENT = "NBA-Chatbot/1.0 (Educational; fetching for local RAG knowledge base)"

# Add NBA news pages here. RAG will index their content for "any NBA news today?" queries.
NEWS_SOURCES = [
    {"url": "https://www.nba.com/news", "name": "NBA.com News", "href_pattern": r"/news/"},
    # Add more, e.g.:
    # {"url": "https://www.espn.com/nba/", "name": "ESPN NBA", "href_pattern": r"/nba/"},
    # {"url": "https://bleacherreport.com/nba", "name": "Bleacher Report NBA", "href_pattern": r"/articles/"},
]

def _wikipedia_seed_titles() -> List[str]:
    """
    Build a broader set of NBA pages for RAG accuracy.
    Includes modern Finals pages, star players, teams, awards, and key lists.
    """
    finals_years = list(range(2010, 2025))
    finals_titles = [f"{y}_NBA_Finals" for y in finals_years]

    core_players = [
        "Stephen_Curry", "LeBron_James", "Kevin_Durant", "Kobe_Bryant", "Michael_Jordan",
        "Nikola_Jokic", "Giannis_Antetokounmpo", "Luka_Doncic", "Jayson_Tatum",
        "Joel_Embiid", "Shai_Gilgeous-Alexander", "Damian_Lillard", "James_Harden",
        "Kawhi_Leonard", "Jimmy_Butler", "Russell_Westbrook", "Chris_Paul",
        "Anthony_Davis", "Kyrie_Irving", "Victor_Wembanyama",
    ]

    key_teams = [
        "Boston_Celtics", "Los_Angeles_Lakers", "Golden_State_Warriors",
        "Chicago_Bulls", "San_Antonio_Spurs", "Miami_Heat", "Cleveland_Cavaliers",
        "Milwaukee_Bucks", "Denver_Nuggets", "Philadelphia_76ers", "Phoenix_Suns",
        "Dallas_Mavericks", "Oklahoma_City_Thunder", "New_York_Knicks",
    ]

    nba_reference_topics = [
        "NBA", "NBA_playoffs", "List_of_NBA_champions", "NBA_Most_Valuable_Player_Award",
        "NBA_All-Star_Game", "NBA_salary_cap", "NBA_draft", "List_of_National_Basketball_Association_career_scoring_leaders",
        "List_of_National_Basketball_Association_career_assists_leaders",
        "List_of_National_Basketball_Association_career_rebounds_leaders",
    ]

    # Keep order stable and remove duplicates.
    seen = set()
    ordered = []
    for title in finals_titles + core_players + key_teams + nba_reference_topics:
        if title not in seen:
            seen.add(title)
            ordered.append(title)
    return ordered


def fetch_wikipedia(title: str) -> str:
    """Fetch summary from Wikipedia REST API. Returns extract text or empty string."""
    url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + title.replace(" ", "_")
    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=10)
        r.raise_for_status()
        data = r.json()
        extract = data.get("extract", "")
        return extract.strip() if extract else ""
    except Exception as e:
        print(f"  Wikipedia {title}: {e}")
        return ""


def fetch_news_from_url(url: str, name: str, href_pattern: Optional[str] = None) -> str:
    """Fetch a news page and extract headlines. Used by RAG for daily NBA news."""
    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=15)
        r.raise_for_status()
        html = r.text
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
            headlines = []
            pat = href_pattern if href_pattern else r".+"
            regex = re.compile(pat)
            for a in soup.find_all("a", href=regex)[:50]:
                txt = a.get_text(strip=True)
                if txt and 15 < len(txt) < 200 and txt not in headlines:
                    headlines.append(txt)
            if headlines:
                return f"=== {name} (Latest) ===\n\nSource: {url}\n\n" + "\n\n".join([f"- {h}" for h in headlines[:20]])
        except ImportError:
            pass
        # Fallback: regex
        m = re.findall(r'["\']([^"\']{30,120})["\']', html)
        if m:
            return f"=== {name} ===\n\n" + "\n\n".join([f"- {x}" for x in m[:15]])
        return f"=== {name} ===\n\nVisit {url} for latest headlines."
    except Exception as e:
        print(f"  {name}: {e}")
        return f"=== {name} ===\n\n(Unable to fetch. Visit {url} for latest.)"


def fetch_all_news() -> str:
    """Fetch from all NEWS_SOURCES. RAG indexes this for 'any NBA news today?'"""
    chunks = []
    for src in NEWS_SOURCES:
        url = src["url"]
        name = src.get("name", url)
        pattern = src.get("href_pattern", "")
        chunks.append(fetch_news_from_url(url, name, pattern or None))
        time.sleep(0.5)
    return "\n\n".join(chunks)


def main():
    print("Fetching NBA content from Wikipedia and NBA.com...")

    chunks = []

    # Wikipedia: broader historical + player + team + awards corpus
    wikipedia_titles = _wikipedia_seed_titles()
    print(f"  Wikipedia seed pages: {len(wikipedia_titles)}")
    for title in wikipedia_titles:
        t = title.replace("_", " ")
        print(f"  Wikipedia: {t}")
        text = fetch_wikipedia(title)
        if text:
            # Keep richer summaries to improve retrieval quality.
            text = text[:1400] + "..." if len(text) > 1400 else text
            chunks.append(f"=== {t} (Source: Wikipedia) ===\n\n{text}")
        time.sleep(0.25)

    # NBA news (from pages that post NBA updates; RAG uses this for news queries)
    print("  News sources:", [s["name"] for s in NEWS_SOURCES])
    news_text = fetch_all_news()
    chunks.append(news_text)

    # Write to nba_recaps.txt
    out_path = RAG_DIR / "nba_recaps.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(chunks))

    # Delete Chroma cache so RAG rebuilds index with new content (close Streamlit first)
    chroma_path = RAG_DIR.parent / ".chroma_db"
    if chroma_path.exists():
        try:
            import shutil
            shutil.rmtree(chroma_path)
            print(f"  Cleared {chroma_path}")
        except (PermissionError, OSError) as e:
            print(f"  Could not clear Chroma cache (close the app first): {e}")

    print(f"\nDone. Wrote {len(chunks)} sections to {out_path}")


if __name__ == "__main__":
    main()
