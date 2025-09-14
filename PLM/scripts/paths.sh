#!/bin/bash

timestamp() { date +"%Y%m%d_%H%M"; }

get_raw_dir() {
  if [[ "$CORPUS" == "VC" ]]; then
    echo "data/raw_VC"
  else
    echo "data/raw_doreco"
  fi
}

get_split_dir() {
  echo "data/${SHAREDNESS}/${SPLIT_NAME}"
}

get_data_dir() {
  local lang=$1
  echo "data/${CORPUS}/${SHAREDNESS}/${SPLIT_NAME}/$lang"
}

get_model_output_dir() {
  local lang=$1
  local ts=$(timestamp)
  echo "models/$CORPUS/${lang}/gpt2_phoneme_${lang}_${MODEL_NAME}_${SPLIT_NAME}_${ts}"
}

get_result_dir() {
  local lang=$1
  local ts=$(timestamp)
  echo "results/$CORPUS/${SPLIT_NAME}/${lang}/gpt2_phoneme_${lang}_${MODEL_NAME}_${SPLIT_NAME}_${ts}"
}

get_phoneme_info_file() {
  local lang=$1
  echo "data/phoneme_infos/doreco_${lang}_phoneme_infos.csv"
}