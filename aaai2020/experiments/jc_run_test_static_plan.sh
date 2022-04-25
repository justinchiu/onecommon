#!/bin/bash

model_dir_a=expts/rel3_tsel_ref_dial_model_separate/nov-15/plain-hierarchical-structured-recurrence/1
model_dir_b=expts/rel3_tsel_ref_dial_model_separate/nov-15/plain-hierarchical-structured-recurrence/1
name=TEST_STATIC_PLAN
shared_ctx_count=4

shift
shift
shift

log_file=analysis_log/${name}.log
out_file=analysis_log/${name}.out

rerank_args="--language_rerank --next_mention_reranking --language_beam_keep_all_finished \
--reranking_confidence"
#rerank_args2="--language_beam_size 16 --next_mention_reranking_k 4 \
#rerank_args2="--language_beam_size 16 --next_mention_reranking_k 16 \

rerank_args2="--language_beam_size 16 --next_mention_reranking_k 8 \
--next_mention_reranking_max_mentions 4"
rerank_args3=" --reranking_confidence_type keep_best --next_mention_candidate_generation topk_multi_mention"

rerank_args2="--language_beam_size 16 --next_mention_reranking_k 4 \
--next_mention_reranking_max_mentions 4"
rerank_args3=" --reranking_confidence_type keep_best"

rerank_args4="--must_contain S_pGlR0nKz9pQ4ZWsw S_n0ocL412kqOAl9QR S_hxYVpiz9A5jI6fyd S_JnKzqWlH9GP4ajch"
rerank_args4="--must_contain S_pGlR0nKz9pQ4ZWsw"
rerank_args4="--must_contain S_n0ocL412kqOAl9QR"

#rerank_args4="--must_contain S_hxYVpiz9A5jI6fyd"
#rerank_args4="--must_contain S_JnKzqWlH9GP4ajch"

python -u test_planning_static.py \
  --alice_model_file=${model_dir_a}/1_ep-12.th \
  --bob_model_file=${model_dir_b}/1_ep-12.th \
  --context_file=shared_${shared_ctx_count} \
  --cuda \
  --markable_detector_file=serialized_models/markable_detector_with_dict_1.th \
  --verbose \
  --DBG_PLAN analysis_log/${name}.json \
  --DBG_GEN \
  --num_contexts 1000 \
  ${rerank_args} ${rerank_args2} ${rerank_args3} ${rerank_args4} \
  --log_file=${log_file} $@

echo "LOGFILE ${log_file}"
exit 0
