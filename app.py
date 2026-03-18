"""
NBA Chatbot - Streamlit chat UI.
Run from project root: streamlit run app.py

Add GOOGLE_API_KEY or GEMINI_API_KEY to .env for full stats + RAG support.
"""
import streamlit as st
import pandas as pd
from pathlib import Path
import sys

# Ensure project root in path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Preload models and data on first run (cached) — avoids slow first query
@st.cache_resource
def _warmup():
    from src.orchestrator import _load_df
    _load_df()  # load CSV into memory
    RAG_DIR = ROOT / "data" / "rag_docs"
    if (RAG_DIR / "nba_recaps.txt").exists():
        try:
            from src.rag import build_index, retrieve
            build_index(RAG_DIR)  # Chroma index + sentence-transformers model
            retrieve(RAG_DIR, "Finals", top_k=1)  # prime retrieval
        except Exception:
            pass

@st.cache_resource
def _load_stats_df():
    """Load stats dataframe for sidebar searches."""
    from src.orchestrator import _load_df
    df_totals, _, _ = _load_df()
    return df_totals

from src.orchestrator import run


st.set_page_config(page_title="NBA Chatbot", page_icon="🏀", layout="wide")

_warmup()  # run once, cached

st.title("🏀 NBA Chatbot")
st.caption("Chat naturally—ask about schedules, stats, comparisons, or NBA history. I remember our conversation.")

# Sidebar with tabs - no scrolling!
with st.sidebar:
    from datetime import datetime
    
    # Current season for display
    now = datetime.now()
    year = now.year
    current_season = f"{year}-{year+1}" if now.month >= 10 else f"{year-1}-{year}"
    
    # ============ MAIN TABS ============
    tab_games, tab_leaders, tab_search = st.tabs(["🏀 Games", "🏆 Leaderboard", "🔍 Search"])
    
    # ============ TAB 1: GAMES (Live Scores + Upcoming) ============
    with tab_games:
        # Live Scoreboard
        try:
            from src.sidebar_data import get_todays_scoreboard, get_all_upcoming_games
            
            @st.cache_data(ttl=30)  # Refresh every 30 seconds for live scores
            def fetch_scoreboard():
                return get_todays_scoreboard()
            
            scoreboard = fetch_scoreboard()
            
            if scoreboard:
                live_games = [g for g in scoreboard if g["status"] == 2]
                final_games = [g for g in scoreboard if g["status"] == 3]
                scheduled_games = [g for g in scoreboard if g["status"] == 1]
                
                # Live games first
                if live_games:
                    st.markdown("### 🔴 LIVE")
                    for g in live_games:
                        period_txt = f"Q{g['period']}" if g["period"] <= 4 else f"OT{g['period']-4}"
                        clock_txt = g['clock'] if g['clock'] else ""
                        
                        col_a, col_s, col_h = st.columns([3, 2, 3])
                        with col_a:
                            st.markdown(f"**{g['away_team']}**")
                        with col_s:
                            st.markdown(f"**{g['away_score']} - {g['home_score']}**")
                        with col_h:
                            st.markdown(f"**{g['home_team']}**")
                        st.caption(f"{period_txt} {clock_txt}")
                        st.markdown("---")
                
                # Final games
                if final_games:
                    st.markdown("### Final")
                    for g in final_games:
                        col_a, col_s, col_h = st.columns([3, 2, 3])
                        with col_a:
                            away_bold = "**" if g["away_score"] > g["home_score"] else ""
                            st.markdown(f"{away_bold}{g['away_team']} {g['away_score']}{away_bold}")
                        with col_s:
                            st.markdown("-")
                        with col_h:
                            home_bold = "**" if g["home_score"] > g["away_score"] else ""
                            st.markdown(f"{home_bold}{g['home_team']} {g['home_score']}{home_bold}")
                    st.markdown("---")
                
                # Scheduled games
                if scheduled_games:
                    st.markdown("### Upcoming Today")
                    for g in scheduled_games:
                        st.markdown(f"**{g['away_team']}** @ **{g['home_team']}**")
                        st.caption(g["status_text"])
            
            # Also show upcoming games from schedule
            if not scoreboard:
                @st.cache_data(ttl=300)
                def fetch_upcoming():
                    return get_all_upcoming_games(n=8)
                
                games = fetch_upcoming()
                if games:
                    st.caption("Upcoming Games")
                    for g in games:
                        st.markdown(f"**{g['away']}** @ **{g['home']}**")
                        st.caption(f"{g['date']} • {g['status']}")
                else:
                    st.caption("No games today")
                    
        except Exception:
            st.caption("Could not load scoreboard")
    
    # ============ TAB 2: LEADERBOARD (Teams | Players) ============
    with tab_leaders:
        st.caption(f"🔴 LIVE • {current_season}")
        
        # Toggle between Teams and Players
        if "leaderboard_view" not in st.session_state:
            st.session_state["leaderboard_view"] = "teams"
        
        col_t, col_p = st.columns(2)
        with col_t:
            if st.button("🏀 Teams", use_container_width=True, 
                        type="primary" if st.session_state["leaderboard_view"] == "teams" else "secondary"):
                st.session_state["leaderboard_view"] = "teams"
                st.rerun()
        with col_p:
            if st.button("👤 Players", use_container_width=True,
                        type="primary" if st.session_state["leaderboard_view"] == "players" else "secondary"):
                st.session_state["leaderboard_view"] = "players"
                st.rerun()
        
        st.markdown("---")
        
        # ========== TEAMS VIEW (Standings) ==========
        if st.session_state["leaderboard_view"] == "teams":
            try:
                from src.sidebar_data import fetch_conference_standings
                
                @st.cache_data(ttl=600, show_spinner=False)
                def fetch_standings(conf):
                    return fetch_conference_standings(conf)
                
                # Conference toggle
                if "standings_conf" not in st.session_state:
                    st.session_state["standings_conf"] = "East"
                
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("🌅 East", use_container_width=True,
                                type="primary" if st.session_state["standings_conf"] == "East" else "secondary"):
                        st.session_state["standings_conf"] = "East"
                        st.rerun()
                with c2:
                    if st.button("🌄 West", use_container_width=True,
                                type="primary" if st.session_state["standings_conf"] == "West" else "secondary"):
                        st.session_state["standings_conf"] = "West"
                        st.rerun()
                
                conf = st.session_state["standings_conf"]
                
                with st.spinner("Loading..."):
                    standings = fetch_standings(conf)
                
                if standings:
                    st.markdown(f"**{conf}ern Conference**")
                    for t in standings:
                        st.caption(f"{t['rank']}. **{t['team']}** {t['wins']}-{t['losses']} ({t['pct']})")
                else:
                    st.warning("Standings loading slow")
                    st.info("💬 Ask in chat:\n\"NBA East standings\"")
            except Exception:
                st.warning("Standings unavailable")
                st.info("💬 Ask in chat:\n\"NBA standings\"")
        
        # ========== PLAYERS VIEW (Stats) ==========
        else:
            try:
                from src.sidebar_data import fetch_live_leaders
                
                @st.cache_data(ttl=300)
                def fetch_leaders_by_stat(stat, per_game):
                    return fetch_live_leaders(stat, limit=5, per_game=per_game)
                
                # Initialize session state
                if "leader_stat" not in st.session_state:
                    st.session_state["leader_stat"] = "PTS"
                if "leader_mode" not in st.session_state:
                    st.session_state["leader_mode"] = "per_game"
                
                # Mode toggle: Per Game vs Totals
                mode_col1, mode_col2 = st.columns(2)
                with mode_col1:
                    if st.button("📊 Per Game", 
                                key="mode_pergame",
                                use_container_width=True,
                                type="primary" if st.session_state["leader_mode"] == "per_game" else "secondary"):
                        st.session_state["leader_mode"] = "per_game"
                        st.rerun()
                with mode_col2:
                    if st.button("📈 Totals", 
                                key="mode_totals",
                                use_container_width=True,
                                type="primary" if st.session_state["leader_mode"] == "totals" else "secondary"):
                        st.session_state["leader_mode"] = "totals"
                        st.rerun()
                
                st.markdown("")  # Spacing
                
                # Stat selector buttons
                stat_options = {
                    "🏀": "PTS",
                    "🎯": "AST", 
                    "💪": "REB",
                    "🖐️": "STL",
                    "🚫": "BLK",
                }
                stat_names = {
                    "PTS": "Points",
                    "AST": "Assists",
                    "REB": "Rebounds",
                    "STL": "Steals",
                    "BLK": "Blocks",
                }
                
                # Button row
                cols = st.columns(5)
                for i, (emoji, stat) in enumerate(stat_options.items()):
                    with cols[i]:
                        is_selected = st.session_state["leader_stat"] == stat
                        if st.button(emoji, key=f"stat_{stat}", 
                                    help=stat_names[stat],
                                    use_container_width=True,
                                    type="primary" if is_selected else "secondary"):
                            st.session_state["leader_stat"] = stat
                            st.rerun()
                
                # Show selected leaders
                selected_stat = st.session_state["leader_stat"]
                selected_mode = st.session_state["leader_mode"]
                per_game = (selected_mode == "per_game")
                
                leaders = fetch_leaders_by_stat(selected_stat, per_game)
                
                if leaders:
                    mode_label = "per game" if per_game else "total"
                    st.markdown(f"**{stat_names[selected_stat]}** ({mode_label})")
                    for p in leaders:
                        val = p['value']
                        val_str = f"{val:.1f}" if isinstance(val, float) else f"{val}"
                        st.caption(f"{p['rank']}. {p['name']} ({p['team']}) - **{val_str}**")
                else:
                    st.caption("Could not load")
            except Exception:
                st.caption("Could not load leaders")
    
    # ============ TAB 3: SEARCH ============
    with tab_search:
        search_query = st.text_input("Team or player...", placeholder="Warriors, LeBron", key="sidebar_search", label_visibility="collapsed")
        
        if search_query and len(search_query) >= 2:
            from src.sidebar_data import (
                resolve_team_abbrev, TEAM_INFO, get_team_next_games, 
                search_players, get_player_full_info, get_team_standing
            )
            
            team_abbrev = resolve_team_abbrev(search_query)
            
            if team_abbrev and team_abbrev in TEAM_INFO:
                # ========== TEAM CARD ==========
                from src.sidebar_data import get_team_full_info
                
                team_info = TEAM_INFO[team_abbrev]
                st.markdown(f"### 🏀 {team_info['name']}")
                st.caption(f"{team_abbrev} • {team_info['conference']}ern Conference")
                
                # Get full team info with standings
                with st.spinner("Loading..."):
                    full_info = get_team_full_info(team_abbrev)
                
                if full_info:
                    # Current Standing
                    if full_info.get("current_rank"):
                        st.markdown(f"**📊 Current:** #{full_info['current_rank']} ({full_info['current_record']})")
                    
                    # Previous Season
                    if full_info.get("prev_rank"):
                        st.caption(f"📅 Last season: #{full_info['prev_rank']} ({full_info['prev_record']})")
                    
                    # Championships & Best Year
                    champs = full_info.get("championships", 0)
                    best_year = full_info.get("best_year", "-")
                    if champs > 0:
                        st.caption(f"🏆 Championships: **{champs}** (Last: {best_year})")
                    else:
                        st.caption(f"🏆 Championships: 0")
                
                st.markdown("---")
                st.markdown("**📅 Next Games:**")
                next_games = get_team_next_games(team_abbrev, n=3)
                if next_games:
                    for g in next_games:
                        away_team = g.get("visitor") or g.get("away", "")
                        home_team = g.get("home", "")
                        opponent = away_team if home_team == team_abbrev else home_team
                        loc = "vs" if home_team == team_abbrev else "@"
                        st.caption(f"• {g['date']} {loc} **{opponent}**")
                else:
                    st.caption("No upcoming games")
            
            else:
                # ========== PLAYER SEARCH ==========
                df = _load_stats_df()
                players = search_players(df, search_query, limit=5)
                
                if players:
                    for p in players:
                        if st.button(f"📊 {p['name']}", key=f"p_{p['player_id']}", use_container_width=True):
                            st.session_state["selected_player"] = p["player_id"]
                    
                    if "selected_player" in st.session_state:
                        player_info = get_player_full_info(df, st.session_state["selected_player"])
                        if player_info:
                            st.markdown("---")
                            st.markdown(f"### {player_info['name']}")
                            
                            try:
                                st.image(player_info["headshot_url"], width=80)
                            except:
                                pass
                            
                            status = "🟢 Active" if player_info["is_active"] else "⚪ Retired"
                            st.caption(f"{status} • {player_info['current_team']}")
                            st.caption(f"Career: {player_info['first_season']} → {player_info['last_season']}")
                            
                            # Current/Last Season Stats
                            current = player_info.get("current_season_stats") or player_info.get("last_season_stats")
                            if current:
                                st.markdown(f"**📊 {current['season']} ({current['team']})**")
                                c1, c2, c3, c4 = st.columns(4)
                                c1.metric("G", current.get('G', 0))
                                c2.metric("PTS", f"{current.get('PTS', 0):.1f}")
                                c3.metric("AST", f"{current.get('AST', 0):.1f}")
                                c4.metric("REB", f"{current.get('TRB', 0):.1f}")
                                
                                c5, c6, c7 = st.columns(3)
                                c5.metric("STL", f"{current.get('STL', 0):.1f}")
                                c6.metric("BLK", f"{current.get('BLK', 0):.1f}")
                                c7.metric("3P%", f"{current.get('3P%', 0):.3f}")
                            
                            # Career Totals
                            career = player_info.get("career", {})
                            if career:
                                st.markdown("**💼 Career Totals**")
                                c1, c2, c3 = st.columns(3)
                                c1.metric("PTS", f"{career.get('PTS', 0):,}")
                                c2.metric("AST", f"{career.get('AST', 0):,}")
                                c3.metric("REB", f"{career.get('TRB', 0):,}")
                                
                                c4, c5, c6 = st.columns(3)
                                c4.metric("STL", f"{career.get('STL', 0):,}")
                                c5.metric("BLK", f"{career.get('BLK', 0):,}")
                                c6.metric("Games", f"{career.get('G', 0):,}")
                else:
                    st.caption("No results")
        else:
            st.caption("Type to search teams or players")
    
    # ============ BOTTOM OPTIONS (always visible) ============
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.messages = []
            for key in ["selected_player", "leader_stat", "leaderboard_view", "standings_conf"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    with col2:
        if st.button("🔄 Reload", use_container_width=True):
            st.cache_resource.clear()
            st.cache_data.clear()
            st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

if prompt := st.chat_input("Ask about NBA stats or history..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[-10:]]
                response = run(prompt, use_llm=True, history=history)
                
                # Append formatted tables to response if DataFrames exist
                from src.orchestrator import get_export_dataframes
                import re as _re
                export_dfs = get_export_dataframes()
                if export_dfs:
                    # Strip any tables/details Gemini may have generated (we add our own)
                    response = _re.sub(r'<details>.*?</details>', '', response, flags=_re.DOTALL).strip()
                    response = _re.sub(r'\|.*\|.*\n(\|[-: |]+\|\n)?(\|.*\|.*\n)*', '', response).strip()
                    
                    response += "\n\n"
                    for idx, df_info in enumerate(export_dfs):
                        df = df_info["df"]
                        label = df_info.get("label", "")
                        source = df_info.get("source", "Unknown")
                        
                        try:
                            markdown_table = df.to_markdown(index=False)
                            if len(export_dfs) > 1:
                                response += f"\n**{label}**\n\n"
                            response += f"**Source:** `{source}`\n\n"
                            response += f"""<details>
<summary>📊 View Detailed Stats Table</summary>

{markdown_table}

</details>

"""
                        except Exception:
                            pass
                            
            except Exception as e:
                err = str(e)
                response = "Something went wrong. Please try again or rephrase." if err in ("object", "'object'", "") else f"Error: {err}"
        st.markdown(response, unsafe_allow_html=True)
        
        # Export UI - Show download buttons if DataFrames are available
        try:
            from src.orchestrator import get_export_dataframes
            import io
            
            export_dfs = get_export_dataframes()
            if export_dfs:
                st.markdown("---")
                st.markdown("**📥 Export Data:**")
                
                for idx, df_info in enumerate(export_dfs):
                    df = df_info["df"]
                    filename = df_info.get("filename", f"nba_data_{idx}")
                    label = df_info.get("label", f"Dataset {idx+1}")
                    source = df_info.get("source", "Unknown")
                    
                    if len(export_dfs) > 1:
                        st.caption(f"**{label}**")
                    st.caption(f"Source: {source}")
                    
                    col1, col2 = st.columns(2)
                    
                    # CSV download
                    with col1:
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📄 Download CSV",
                            data=csv,
                            file_name=f"{filename}.csv",
                            mime="text/csv",
                            key=f"csv_{idx}_{filename}",
                            use_container_width=True
                        )
                    
                    # Excel download
                    with col2:
                        buffer = io.BytesIO()
                        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                            df.to_excel(writer, index=False, sheet_name='NBA Stats')
                        excel_data = buffer.getvalue()
                        
                        st.download_button(
                            label="📊 Download Excel",
                            data=excel_data,
                            file_name=f"{filename}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key=f"excel_{idx}_{filename}",
                            use_container_width=True
                        )
                    
                    if len(export_dfs) > 1 and idx < len(export_dfs) - 1:
                        st.markdown("")  # Spacing between multiple exports
        except Exception:
            pass  # Silently fail if export not available

    st.session_state.messages.append({"role": "assistant", "content": response})
