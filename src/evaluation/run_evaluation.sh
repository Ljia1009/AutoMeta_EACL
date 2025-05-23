#!/bin/bash

DATA_PATH="/Users/chenxinliu/LING573_AutoMeta/data/ORSUM_test.jsonl"
OUTPUT_DIR="/Users/chenxinliu/LING573_AutoMeta/output"
EVAL_DIR="/Users/chenxinliu/LING573_AutoMeta/evaluation"


for OUTPUT_PATH in "$OUTPUT_DIR"/*_out.txt.json; do
    BASENAME=$(basename "$OUTPUT_PATH" .json)    
    SUFFIX="${BASENAME#_out.txt}"  

    EVAL_PATH="$EVAL_DIR/${BASENAME}.csv"

    KEY_OPTION="all"

    echo "=== handling $OUTPUT_PATH ==="
    python run_evaluation.py \
        --data_path "$DATA_PATH" \
        --key_option "$KEY_OPTION" \
        --output_path "$OUTPUT_PATH" \
        --evaluation_result_path "$EVAL_PATH"
    
    KEY_OPTION="review"

    python run_evaluation.py \
        --data_path "$DATA_PATH" \
        --key_option "$KEY_OPTION" \
        --output_path "$OUTPUT_PATH" \
        --evaluation_result_path "$EVAL_PATH"
done
