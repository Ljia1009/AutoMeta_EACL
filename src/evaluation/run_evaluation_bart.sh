#!/bin/bash

DATA_PATH="data/raw/ORSUM_test.jsonl"
OUTPUT_DIR="outputs/generated"
EVAL_DIR="outputs/evaluation"



for OUTPUT_PATH in "$OUTPUT_DIR"/*_out.txt.json; do
    BASENAME=$(basename "$OUTPUT_PATH" .json)    
    SUFFIX="${BASENAME#_out.txt}"  

    EVAL_PATH="$EVAL_DIR/${BASENAME}.csv"

    KEY_OPTION="all"

    echo "=== handling $OUTPUT_PATH ==="
    python3 src/run_evaluation_bart.py \
        --data_path "$DATA_PATH" \
        --key_option "$KEY_OPTION" \
        --output_path "$OUTPUT_PATH" \
        --evaluation_result_path "$EVAL_PATH"
    
    KEY_OPTION="review"

    python3 src/run_evaluation_bart.py \
        --data_path "$DATA_PATH" \
        --key_option "$KEY_OPTION" \
        --output_path "$OUTPUT_PATH" \
        --evaluation_result_path "$EVAL_PATH"
done
