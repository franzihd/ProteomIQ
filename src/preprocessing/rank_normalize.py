from pathlib import Path
import pandas as pd
import argparse

def rank_normalize(cohort):
    data_dir = Path(f"data/cptac/{cohort}")
    df = pd.read_csv(data_dir / "proteomics_raw.csv", header=[0,1], index_col=0)
    df_ranked = df.rank(axis=1, pct=True, na_option='keep')
    df_ranked.to_csv(data_dir / "rank_normalized_proteomics.csv")
    print(df_ranked.shape)
    print(f"NaN count before: {df.isna().sum().sum()}")
    print(f"NaN count after: {df_ranked.isna().sum().sum()}")
    print(f"Value range: {df_ranked.min().min():.3f} to {df_ranked.max().max():.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", required=True)
    args = parser.parse_args()
    rank_normalize(args.cohort)