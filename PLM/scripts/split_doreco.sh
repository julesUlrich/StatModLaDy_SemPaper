#!/bin/bash

#echo "$(get_data_dir $lang)"

split_doreco() {
  local lang=$1
  python src/split_corpus.py \
    --input_dir "data/raw_doreco/${SHAREDNESS}/doreco_${lang}" \
    --output_dir "$(get_data_dir $lang)" \
    --eval_split \
    --train_ratio $TRAIN_RATIO \
    --eval_ratio $EVAL_RATIO \
    --overview_file "data/raw_doreco/${SHAREDNESS}/doreco_${lang}/doreco_${lang}_sentence_overview.csv" \
    --enhanced_overview_filename "${lang}_${SPLIT_NAME}_reference_overview.csv"
}
