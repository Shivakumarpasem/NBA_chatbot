# For Custom GPT: Project Structure & How to Run

## 1. Tree output (structure, ~4 levels)

Paste this to your GPT so it knows the layout:

```
NBA Chatbot/
├── Column.py
├── PROJECT_STRUCTURE_AND_RUN.md
├── data/
│   ├── Readme.md
│   ├── all_time_teams.csv
│   ├── current_teams.csv
│   ├── rookies_by_year.csv
│   ├── schedule.csv
│   ├── player_stats/
│   │   ├── totals_stats.csv          ← main data (single source of truth)
│   │   ├── adj-shooting.csv
│   │   ├── advanced_stats.csv
│   │   ├── per_game_stats.csv
│   │   ├── per_minute_stats.csv
│   │   ├── per_poss_stats.csv
│   │   ├── pbp_stats.csv
│   │   └── shooting_stats.csv
│   └── boxscores_by_year/
│       └── (multiple CSV files)
├── docs/
│   └── ARCHITECTURE_WORKFLOW.md
└── src/
    ├── nba_data.py         ← all tools (load, find_players, player_summary, compare_players, etc.)
    ├── test_metrics.py     ← main file to run (tests all tools)
    ├── test_load.py
    └── test_search.py
```

**Note:** Only `data/player_stats/totals_stats.csv` is used as the single source of truth for the current tools. Other CSVs are not used yet.

---

## 2. File to run + command

**File to run:** `src/test_metrics.py`

**Command (from project root):**
```bash
python src/test_metrics.py
```

**Command (if you are already inside `src/`):**
```bash
python test_metrics.py
```

**Working directory:** Project root = `C:\Users\shiva\NBA Chatbot` (or wherever the unzipped folder is).  
So run from there: `python src/test_metrics.py`

**What it does:** Loads `data/player_stats/totals_stats.csv`, runs the tools (Curry 3PT by season, top 3P%, top PTS, player_summary, compare_players), and prints the results. No UI; this is the current “run” for the project.
