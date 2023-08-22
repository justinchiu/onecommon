#!/bin/bash

dir=gpt200_90_logs
name=gpt200_90_selfplay

dir=dbg-gpt-4-0613
name=dbg-gpt-4-0613-selfplay

dir=gpt-4-0613-200
name=gpt-4-0613--200selfplay


dir=baselinegpt4-0822-200
name=baselinegpt4-0822--200selfplay

shared_ctx_count=4

shift
shift
shift

mkdir -p $dir

log_file=${dir}/${name}_ctx-${shared_ctx_count}.selfplay.log
out_file=${dir}/${name}_ctx-${shared_ctx_count}.selfplay.out

rerank_args="--language_rerank --next_mention_reranking --language_beam_keep_all_finished --reranking_confidence"
rerank_args2="--language_beam_size 16 --next_mention_reranking_k 4 --next_mention_reranking_max_mentions 4"

#python -u selfplay.py \
  #--alice_model_file=${model_dir_a}/1_ep-12.th \
  #--bob_model_file=${model_dir_b}/1_ep-12.th \
  #--context_file=shared_${shared_ctx_count} \
  #--cuda \
  #--markable_detector_file=serialized_models/markable_detector_with_dict_1.th \
  #--verbose \
  #--num_contexts 1000 \
  #${rerank_args} ${rerank_args2} \
  #--log_file=${log_file} $@
#exit 0

# tee-ing to out for now, messes up signals

# 50 contexts for now
python gpt_selfplay.py \
  --context_file=shared_${shared_ctx_count} \
  --markable_detector_file=serialized_models/markable_detector_with_dict_1.th \
  --verbose \
  --num_contexts 200 \
  --log_file=${log_file} \
  $@ #\
  #| tee ${out_file}
