#!/bin/bash
set -euo pipefail

source scripts/config.sh
source scripts/paths.sh
source scripts/split_doreco.sh

for LANG in "${LANGS[@]}"; do
  echo "Processing $LANG."

  # Split
  split_doreco "$LANG"

  DATA_DIR=$(get_data_dir $LANG)
  OUT_DIR=$(get_model_output_dir $LANG)
  RES_DIR=$(get_result_dir $LANG)
  PHON_CSV=$(get_phoneme_info_file $LANG)

  mkdir -p -- "$OUT_DIR" "$RES_DIR"

  # Tokenizer
  python src/train_tokenizer.py \
    --train_path "$DATA_DIR/train.txt" \
    --phoneme_csv "$PHON_CSV" \
    --column phoneme \
    --fixed_vocab \
    --output_path "$OUT_DIR/tokenizer_${LANG}.json" \
    --handle_missing ignore

  # Model training
  python src/train_model.py \
    --train_path "$DATA_DIR/train.txt" \
    --eval_path "$DATA_DIR/eval.txt" \
    --tokenizer_path "$OUT_DIR/tokenizer_${LANG}.json" \
    --output_dir "$OUT_DIR" \
    --n_layers "$N_LAYERS" \
    --heads "$HEADS" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --target_perplexity "$TARGET_PERPLEXITY" \
    --epochs "$EPOCHS" \
    2>&1 | tee "$OUT_DIR/training.log"

  # Evaluation
  python src/evaluate_entropy.py \
    --test_files_dir "$DATA_DIR" \
    --overview_csv "${DATA_DIR}/${LANG}_${SPLIT_NAME}_reference_overview.csv" \
    --output_dir "$RES_DIR" \
    --model_dir "$OUT_DIR/best_model" \
    --csv_path "$PHON_CSV" \
    --csv_column phoneme \
    --tokenizer_path "$OUT_DIR/final_model/tokenizer/tokenizer.json" \
    --exclude_special_from_csv \
    --glottocode "$LANG" \
    --build_fallback \
    --max_context_length 8

  python src/create_result_summary.py \
    --results_dir "$RES_DIR" \
    --out_csv "$RES_DIR/${LANG}_${SPLIT_NAME}_ER.csv"

  # Merge speech rate
  if [[ "$CORPUS" == "VC" ]]; then
    SR_CSV="data/raw_VC/${SHAREDNESS}/VC_${LANG}/VC_${LANG}_phoneme_durations.csv"
  else
    SR_CSV="data/raw_doreco/${SHAREDNESS}/doreco_${LANG}/doreco_${LANG}_phoneme_durations.csv"
  fi

  python src/merge_robust.py \
    --entropy_csv "$RES_DIR/${LANG}_${SPLIT_NAME}_ER.csv" \
    --speechrate_csv "$SR_CSV" \
    --out_csv "$RES_DIR/${LANG}_${SPLIT_NAME}_information_rate.csv" \
    --fallback_entropy "$RES_DIR/ent_train_eval.csv" \
    --impute_sr speaker_mean

  echo "Finished $LANG"
done

echo "All done."