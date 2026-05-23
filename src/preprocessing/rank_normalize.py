from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/luad")
PROTEOMICS = DATA_DIR / "proteomics_raw.csv"
OUTPUT_PATH = DATA_DIR / "rank_normalized_proteomics.csv"

df = pd.read_csv(PROTEOMICS, header=[0,1], index_col=0)

def rank_normalize(df):
    df_ranked = df.rank(axis=1, pct=True, na_option='keep')
    df_ranked.to_csv(OUTPUT_PATH)
    print(df_ranked.shape)
    print(f"NaN count before: {df.isna().sum().sum()}")
    print(f"NaN count after: {df_ranked.isna().sum().sum()}")
    print(f"Value range: {df_ranked.min().min():.3f} to {df_ranked.max().max():.3f}")


rank_normalize(df)