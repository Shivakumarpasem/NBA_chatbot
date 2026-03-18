import pandas as pd

PATH = "NBA_Dataset/player_stats/adj-shooting.csv"  # change if needed

df = pd.read_csv(PATH)
print("Rows:", len(df))
print("Columns:", list(df.columns))
print(df.head(3).to_string(index=False))