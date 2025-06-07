import argparse
import pandas as pd
from sklearn.metrics import classification_report

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        default="",
        help="Full path of the file used for input logits. ",
    )
    parser.add_argument(
        "--result_path",
        type=str,
        default="",
        help="Full path of the file used for the result. ",
    )
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # load logits from a CSV file to a dataframe
    arg = get_args()
    input_path = arg.input_path
    df = pd.read_csv(input_path, index_col=0)
    
    ref_labels = df[['ref_logit_0','ref_logit_1']].idxmax(axis=1)
    gen_labels = df[['gen_logit_0','gen_logit_1']].idxmax(axis=1)
    ref_labels = ref_labels.replace({f'ref_logit_{i}': i for i in range(2)})
    gen_labels = gen_labels.replace({f'gen_logit_{i}': i for i in range(2)})

    # save the classification report to a text file
    with open(arg.result_path, 'w') as f:
        f.write(classification_report(ref_labels, gen_labels))    
