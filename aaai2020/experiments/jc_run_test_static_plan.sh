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

rerank_args2="--language_beam_size 32 --next_mention_reranking_k 8 \
--next_mention_reranking_max_mentions 4"
rerank_args2="--language_beam_size 64 --next_mention_reranking_k 8 \
--next_mention_reranking_max_mentions 4"
rerank_args3=" --reranking_confidence_type keep_best"

#rerank_args4="--must_contain S_pGlR0nKz9pQ4ZWsw S_n0ocL412kqOAl9QR S_hxYVpiz9A5jI6fyd S_JnKzqWlH9GP4ajch"
rerank_args4="--must_contain S_rguqGgNYfrnW7AFz"
rerank_args4="--must_contain \
S_SXTkzYMf65Txohx1 \
S_d1A25BOwQKs9ea96 \
S_CGIVg5Xg4PX8Cb1u \
S_vNh9L6L87GQlSQBd \
S_Hacog9vt6Ezr19YQ \
S_iOLJUUUVcUGzaIMi \
S_sqkWqU9oCt7gPCNj \
S_pqC0O80Ojf5BKLyV \
S_prpoeEF96KCoJSNQ \
S_quuRrcGUJVSDQYau \
S_PYif71iPEFrO4ACc \
S_NUwkRTnWCDZH0dox \
S_hjX5jTcQ73bzhon0 \
S_4HAFEDGI61j0QePK \
S_VN8fzF9YgHtzXDsX \
S_RTwBsuFR8n8ryo3g \
S_TiT6JNB0XSpnAjzP \
S_RWYCZVdTDVyxpjWr \
S_XIvcA4MT8hC0zN9M \
S_MMCMJd56CCUER6gV "

#rerank_args4="--must_contain S_pqC0O80Ojf5BKLyV"
#rerank_args4="--must_contain S_RWYCZVdTDVyxpjWr"
rerank_args4=""

split=train
split=valid
seed=1
context_file="${split}_context_${seed}"
#belief="or"
belief="cost"
#belief="cost_egocentric"
#logdir="analysis_log/${split}_${seed}"
logdir="analysis_log/${split}_${seed}_absolute_${belief}"

mkdir -p ${logdir}

python -u test_planning_static.py \
  --alice_model_file=${model_dir_a}/1_ep-12.th \
  --bob_model_file=${model_dir_b}/1_ep-12.th \
  --belief or \
  --absolute_bucketing \
  --context_file=${context_file} \
  --cuda \
  --markable_detector_file=serialized_models/markable_detector_with_dict_1.th \
  --verbose \
  --DBG_PLAN analysis_log/${name}.json \
  --dialog_log_dir ${logdir} \
  ${rerank_args} ${rerank_args2} ${rerank_args3} ${rerank_args4} \
  --log_file=${log_file} $@

echo "LOGFILE ${log_file}"
exit 0

#switch to num_contexts 1000
