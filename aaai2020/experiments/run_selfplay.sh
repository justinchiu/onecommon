#!/bin/bash

model_path=$1

python -u selfplay.py \
  --alice_model_file $model_path \
  --bob_model_file $model_path \
  --temperature 0.25 \
  --context_file shared_4 \
  | tee expts/${model_path}.selfplay
