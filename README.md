# NBA Chatbot

Hybrid RAG + tool-calling assistant for NBA stats and history.

## Why I built this

I wanted a chat-first way to explore sports data (starting with the NBA) while keeping the project practical: fast answers for stats/schedules, plus narrative/history questions using retrieval.

Long-term, my goal is to expand this into a broader **sports data platform** where anyone can easily:

- Search across leagues and seasons
- Pull exactly the slice of data they need
- Download results as **CSV / Excel** for analysis and dashboards

## Quick Start

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Create `.env`** in project root with your Gemini API key:
   ```
   GOOGLE_API_KEY=your-gemini-key-here
   ```
   Get a free key at https://aistudio.google.com/apikey

3. **Run the chat app**
   ```bash
   streamlit run app.py
   ```

4. Open the URL (usually http://localhost:8501) and ask questions.

## What You Can Ask

- **Stats**: "Who led in points in 2023-2024?", "Compare Curry and LeBron 2023-24", "Top 10 in steals"
- **History / RAG**: "What happened in the 2016 Finals?", "Tell me about Ray Allen's shot", "Michael Jordan last shot"
- **Schedules / games**: "Who are the Bulls playing next?", "Games today", "Upcoming games"

## Project Structure

- `app.py` — Streamlit chat UI
- `src/orchestrator.py` — Routes queries to tools or RAG, calls Gemini
- `src/rag.py` — RAG: embeddings (sentence-transformers) + Chroma vector store
- `src/nba_data.py` — Stats tools (leaderboards, player summary, etc.)
- `src/live_data.py` — Live endpoints (schedule/scoreboard where applicable)

## Data notes

This repo is designed to be safe to publish publicly:

- Local secrets (like `.env`) are **not committed**
- Large/local datasets under `data/` are **not committed**

If you want the full experience locally, you can either:

- Add your own datasets under `data/` (see file paths referenced in `src/orchestrator.py`), or
- Use the included scripts to fetch/build knowledge sources where available (example below).

## Fetch Fresh Content (Wikipedia + NBA News)

Run regularly (e.g. daily) to refresh the RAG knowledge base with news and verified content:

```bash
python scripts/fetch_nba_content.py
```

- **News queries** ("any NBA news today?") use RAG → populated by this script from NBA news pages
- **Schedule queries** use live API (NBA CDN) or `BALDONTLIE_API_KEY` if set
- **Historical** (Finals, bios) comes from Wikipedia + RAG docs

**Close the Streamlit app first** so the Chroma cache can be cleared. Or delete `data/.chroma_db` after fetching.

**To add news sources:** Edit `NEWS_SOURCES` in `scripts/fetch_nba_content.py`.

## Run Stats Tests Only (No LLM)

```bash
python src/test_metrics.py
```
