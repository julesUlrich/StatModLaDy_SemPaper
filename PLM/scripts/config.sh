#!/bin/bash

# Corpus: "doreco" or "VC"
export CORPUS="doreco"

# Sharedness level: "shared" or "not_shared"
export SHAREDNESS="not_shared"

# Name of the data split (used in folder and model naming)
export SPLIT_NAME="doreco75"

# Split ratios
export TRAIN_RATIO=0.75
export EVAL_RATIO=0.15
export TEST_RATIO=0.1

# Model tag
export MODEL_NAME="normal"

# Training params
export N_LAYERS=12
export HEADS=8
export BATCH_SIZE=8
export LEARNING_RATE=3e-5
export TARGET_PERPLEXITY=3.0
export EPOCHS=2

# Language list
export LANGS=(
  nisv1234
  nort2875
  orko1234
  port1286
  sout2856
  teop1238
  toto1304
  vera1241
)
