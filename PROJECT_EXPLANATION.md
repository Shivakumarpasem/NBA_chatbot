# NBA Chatbot — Full Project Explanation

This document explains everything that was built and how it works.

---

## Is It Done?

**Yes.** The NBA Chatbot is complete with:

1. **Stats tools** — Python functions that compute exact numbers from CSV data  
2. **RAG path** — Retrieval-augmented generation for NBA history/narrative questions  
3. **LLM orchestration** — Google Gemini routes queries and calls tools  
4. **Chat UI** — Streamlit app for interactive use  

What’s **not** done (optional later improvements):

- Multi-turn conversation (history is not kept across messages)
- Evals (Ragas, TruLens)
- Deployment (Docker, cloud hosting)

---

## 1. Big Picture: What Happens When You Ask a Question

```
You type: "Who led in points in 2023-24?"
                ↓
        [Orchestrator checks: stats or RAG?]
                ↓
        Stats question → Gemini picks a tool (top_stat_leaderboard)
                ↓
        Python runs the tool → returns table of top scorers
                ↓
        Gemini summarizes → "Shai Gilgeous-Alexander led with 1808 points..."
```

For a narrative question:

```
You type: "What happened in the 2016 Finals?"
                ↓
        [Orchestrator: RAG triggers (heuristic)]
                ↓
        RAG: embed query → search Chroma → get top 4 chunks
                ↓
        Gemini gets chunks as context → answers only from that
```

---

## 2. File-by-File Explanation

### `app.py` (Streamlit chat UI)

- Entry point for the chat interface
- Keeps message history in `st.session_state.messages`
- Each user input is sent to `run()` in the orchestrator
- Shows user and assistant messages in chat bubbles
- Uses `st.chat_input` and `st.chat_message`

### `src/orchestrator.py` (Core logic)

**Purpose:** Route queries to either stats tools or RAG, then use Gemini to answer.

**Flow:**

1. **Load `.env`** — Uses `python-dotenv` to read `GOOGLE_API_KEY` or `GEMINI_API_KEY` from `.env`
2. **RAG vs stats** — `_is_rag_query()` checks for phrases like “what happened”, “finals”, “tell me about”, etc. If matched → RAG path. Otherwise → stats path.
3. **RAG path:**  
   - `rag.retrieve()` returns top‑k chunks from the knowledge base  
   - Chunks are passed to Gemini: “Answer only from this context”  
   - Reduces hallucinations by grounding the answer in retrieved text
4. **Stats path:**  
   - Gemini is given tool definitions and the user query  
   - It chooses a tool (e.g. `top_stat_leaderboard`, `find_players`) and parameters  
   - `_call_tool()` executes the Python function on your CSV data  
   - The tool result is sent back to Gemini for a natural-language summary

**Tools Gemini can call:**

- `find_players` — search by name (e.g. “curry”)
- `player_summary` — one player’s stats for a season
- `top_stat_leaderboard` — any stat: PTS, AST, TRB, STL, BLK, Misses, 3P
- `top_3pt_pct` — best 3P% shooters
- `compare_players` — side‑by‑side stats for several players
- `player_3pt_by_season` — 3PT stats by season
- `get_last_n_games` — last N games for a player (from box scores)

### `src/rag.py` (RAG pipeline)

**Purpose:** Store NBA text in a vector DB and retrieve relevant chunks for a query.

**Flow:**

1. **Chunking** — `nba_recaps.txt` is split by `=== header ===` sections into chunks
2. **Embeddings** — `sentence-transformers` (all‑MiniLM‑L6‑v2) turns text into vectors locally (no API cost)
3. **Storage** — Chroma stores chunks and vectors in `data/.chroma_db`
4. **Retrieval** — A query is embedded, Chroma returns the most similar chunks
5. **On first use** — If the DB is empty, `build_index()` runs automatically

**Why this setup:**

- `sentence-transformers` + Chroma run entirely on your machine
- No extra cost for embeddings or vector storage
- Only Gemini calls cost money (free tier is usually enough)

### `src/nba_data.py` (Stats tools)

**Purpose:** Read CSV data and compute stats in Python.

- `load_adj_shooting()` — loads `totals_stats.csv`, cleans columns, derives `Misses = FGA - FG`
- `find_players()` — search by name
- `top_stat_leaderboard()` — generic leaderboard for any numeric stat
- `top_3pt_pct()` — best 3P% shooters
- `player_summary()` — one player’s season stats
- `compare_players()` — compare several players
- `player_3pt_by_season()` — 3PT over seasons
- `get_last_n_games()` — last N games (from box scores)
- `load_box_scores()` — load box score CSV

Numbers are computed in Python, not by the LLM.

### `src/call_tools.py` (Safe wrapper)

- `safe_call(fn, *args, **kwargs)` — runs a function and returns `{ok, data, error}`
- Keeps tool errors from crashing the app

### `data/rag_docs/nba_recaps.txt` (Knowledge base)

Text for RAG: recaps and bios about:

- 2016 Finals (Cavaliers comeback)
- 2013 Finals (Ray Allen shot)
- 1998 Finals (Jordan’s last shot)
- Stephen Curry
- LeBron James

You can edit this file to add more NBA content.

---

## 3. Config and Security

### `.env`

- Holds `GOOGLE_API_KEY` or `GEMINI_API_KEY`
- Loaded via `python-dotenv`
- Listed in `.gitignore` so the key is never committed

### `.gitignore`

- `.env` — API key
- `.chroma_db/` — vector DB cache
- `__pycache__/`, `*.pyc`, `venv/` — Python artifacts

---

## 4. How to Run

```bash
# 1. Install
pip install -r requirements.txt

# 2. Ensure .env has your key
# GOOGLE_API_KEY=your-key-here

# 3. Start the chat
streamlit run app.py
```

Open the URL (typically http://localhost:8501) and ask questions.

---

## 5. Cost Overview

| Component              | Cost                          |
|------------------------|--------------------------------|
| Gemini API             | Free tier (e.g. ~60 req/min)  |
| sentence-transformers  | Free (runs locally)           |
| Chroma                 | Free (local storage)          |
| Streamlit              | Free (local)                   |

---

## 6. Architecture Diagram (Conceptual)

```
User query
    ↓
Orchestrator
    ├─ RAG path: query → embed → Chroma → chunks → Gemini → answer
    └─ Stats path: query → Gemini (tools) → pick tool → Python runs it → Gemini summarizes → answer
```

---

## Summary

You have a working NBA chatbot that:

1. Uses **tools** for stats (leaderboards, comparisons) from your CSV data  
2. Uses **RAG** for history/narrative questions from the knowledge base  
3. Uses **Gemini** to route queries, call tools, and generate answers  
4. Exposes a **Streamlit chat UI** for interaction  
5. Keeps your API key in `.env` and out of version control  

The project is complete and ready to run. Add more content to `nba_recaps.txt` or extend the tools in `nba_data.py` as needed.
