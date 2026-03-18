from nba_data import load_adj_shooting

PATH = r"C:\Users\shiva\NBA Chatbot\data\player_stats\totals_stats.csv"  # change this

df = load_adj_shooting(PATH)
print("shape:", df.shape)
print("columns:", df.columns.tolist())
print(df.head(3).to_string(index=False))
