#!/bin/bash

DATA_PATH="../data/raw/ORSUM_test.jsonl"
OUTPUT_DIR="../outputs/generated/baseline/comparison/review"
EVAL_DIR="../outputs/evaluation/baseline/review_bertscore"
FILE_OPTION="test"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPT_DIR/../src" || exit

for OUTPUT_PATH in "$OUTPUT_DIR"/*_out.txt.json; do

    BASENAME=$(basename "$OUTPUT_PATH" .json)    
    SUFFIX="${BASENAME#_out.txt}"  

    EVAL_FILE="$EVAL_DIR/${BASENAME}.csv"

    echo "=== handling $OUTPUT_PATH ==="
    KEY_OPTION="review"

    python -m evaluation.run_evaluation_bertscore "$@" \
        --output_json_path $OUTPUT_PATH \
        --orig_data_path $DATA_PATH \
        --save_path $EVAL_DIR \
        --save_file $EVAL_FILE \
        --file_option  $FILE_OPTION \
        --key_option $KEY_OPTION \
        --device "mps"
done