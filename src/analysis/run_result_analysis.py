import pandas as pd
import glob
import os
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

file_paths = glob.glob("outputs/evaluation/finetune/review_bertscore/*.csv")

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
    }
    main_metrics.append(summary)

df_main_metrics = pd.DataFrame(main_metrics)

df_main_metrics.to_csv("outputs/analysis/evaluation_summary_finetuned.csv", index=False)

print(df_main_metrics)
