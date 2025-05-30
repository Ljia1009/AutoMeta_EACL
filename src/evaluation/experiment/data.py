import os
import pandas as pd


os.chdir("LING573_AutoMeta/outputs/evaluation/experiment")

print("cwd:", os.getcwd())

# group for length
df_baseline = pd.read_csv("bart_review_50_baseline_40_90.txt.csv", index_col=False)
df_v1 = pd.read_csv("bart_review_50_baseline_100.txt.csv", index_col=False)
df_v2 = pd.read_csv("bart_review_50_baseline_150.txt.csv", index_col=False)

for col in df_baseline.columns:
    if col == "gold":
        continue
    col_v1 = f"{col}_v1"
    col_v2 = f"{col}_v2"

    df_baseline.rename(columns={col: f"{col}_"}, inplace=True)
    df_baseline[col_v1] = df_v1[col]
    df_baseline[col_v2] = df_v2[col]

df_baseline = df_baseline.drop(columns=[c for c in df_baseline.columns if c.startswith('Unnamed')], errors='ignore')
df_baseline = df_baseline.reset_index(drop=True)

df_baseline.to_csv("length_summary.csv")


# group for individual prompts
df_baseline = pd.read_csv("bart_review_50_baseline_150.txt.csv", index_col=False)
df_individual = pd.read_csv("bart_review_50_individual_baseline.txt.csv", index_col=False)
for col in df_baseline.columns:
    if col == "gold":
        continue
    df_baseline.rename(columns={col: f"{col}_"}, inplace=True)
    col_individual = f"{col}_individual"
    df_baseline[col_individual] = df_individual[col]

df_baseline = df_baseline.drop(columns=[c for c in df_baseline.columns if c.startswith('Unnamed')], errors='ignore')
df_baseline = df_baseline.reset_index(drop=True)

df_baseline.to_csv("individual_summary.csv")


# group for overall prompts (individual prompt not included but dynamic length)
df_baseline = pd.read_csv("bart_review_50_baseline_150.txt.csv", index_col=False)
df_v1 = pd.read_csv("bart_review_50_prompt_v1_150.txt.csv", index_col=False)
df_v2 = pd.read_csv("bart_review_50_prompt_v2_150.txt.csv", index_col=False)
for col in df_baseline.columns:
    if col == "gold":
        continue

    df_baseline.rename(columns={col: f"{col}_v0"}, inplace=True)
    col_v1 = f"{col}_v1"
    col_v2 = f"{col}_v2"

    df_baseline[col_v1] = df_v1[col]
    df_baseline[col_v2] = df_v2[col]

df_baseline = df_baseline.drop(columns=[c for c in df_baseline.columns if c.startswith('Unnamed')], errors='ignore')
df_baseline = df_baseline.reset_index(drop=True)
df_baseline.to_csv("prompt_summary.csv")
