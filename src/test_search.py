from nba_data import load_adj_shooting, find_players

PATH = r"C:\Users\shiva\NBA Chatbot\data\player_stats\totals_stats.csv"

df = load_adj_shooting(PATH)

print(find_players(df, "curry", limit=10).to_string(index=False))
