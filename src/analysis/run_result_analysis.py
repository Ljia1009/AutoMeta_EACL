import pandas as pd
import glob
import os
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

file_paths = glob.glob("../evaluation/*.csv")

regular_files = [fp for fp in file_paths if not os.path.basename(fp).startswith('disco')]
main_metrics = []

for file_path in regular_files:
    model_name = os.path.basename(file_path).replace("_out.txt.csv", "")
    df = pd.read_csv(file_path)
    df['prediction_tokens'] = df['prediction'].apply(lambda x: len(word_tokenize(str(x))))

    summary = {
        'model': model_name,
        'Prediction-Tokens-Length': df['prediction_tokens'].mean(),
        'ROUGE': df['rouge_score'].mean(),
        'BERTScore-F1': df['bertscore_f1'].mean(),
        'BERTScore-precision': df['bertscore_precision'].mean(),
        'BERTScore-recall': df['bertscore_recall'].mean(),
        'FactCC-Score': df['factCC_score'].mean(),
        'FactCC-Label-Rate': (df['factCC_label'] == 'CORRECT').mean(),
    }
    main_metrics.append(summary)

df_main_metrics = pd.DataFrame(main_metrics)

disco_files = [fp for fp in file_paths if os.path.basename(fp).startswith("disco")]

disco_metrics = []

for file_path in disco_files:
    model_name = os.path.basename(file_path).replace("disco_", "").replace("_out.txt.csv", "")
    df = pd.read_csv(file_path)

    summary = {
        'model': model_name,
        'Disco_EntityGraph': df['EntityGraph'].mean(),
        'Disco_LexicalChain': df['LexicalChain'].mean(),
        'Disco_RC': df['RC'].mean(),
        'Disco_LC': df['LC'].mean(),
    }
    disco_metrics.append(summary)

df_disco_metrics = pd.DataFrame(disco_metrics)

df_merged = pd.merge(df_main_metrics, df_disco_metrics, on='model', how='outer')  # outer in case some models are missing either part

df_merged = df_merged.round(4)

df_merged = df_merged.sort_values(by="model", ascending=False)
df_merged.to_csv("../evaluation/analysis/evaluation_summary_combined.csv", index=False)

print(df_merged)
