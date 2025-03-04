#!/bin/bash

#prediction_type="collapsed"
#belief="or"
#belief="cost"
#belief="cost_egocentric"

belief=$1

if [[ -z "$1" ]]; then
    echo "Must provide first argument=belief {or,cost,cost_egocentric}" 1>&2
    exit 1
fi


prediction_type=${2-multi_reference}

if [ $prediction_type = "multi_reference" ]
then
    model_dir_a=expts/rel3_tsel_ref_dial_model_separate/nov-15/plain-hierarchical-structured-recurrence/1
    model_dir_b=expts/rel3_tsel_ref_dial_model_separate/nov-15/plain-hierarchical-structured-recurrence/1
    #model_dir_a=expts/rel3_tsel_ref_dial_model_separate/jc-partner/baseline/1
    #model_dir_b=expts/rel3_tsel_ref_dial_model_separate/jc-partner/baseline/1
else
    model_dir_a=expts/rel3_tsel_ref_dial_model_separate/jc-partner/collapsed-mention-baseline/1
    model_dir_b=expts/rel3_tsel_ref_dial_model_separate/jc-partner/collapsed-mention-baseline/1
fi

echo "prediction_type=$prediction_type"

bucket_size=${3-5}
echo "bucket_size=$bucket_size"

shared_ctx_count=4

bart_dir=${4}
echo "bart_dir=${bart_dir}"

shift
shift
shift


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
#rerank_args4="--must_contain S_a5VVGxsBgrYKSijz"
#rerank_args4="--must_contain S_rfDQvM9yjii25RlV"
#rerank_args4="--must_contain S_RUtTM6LOTtufBj20"
#rerank_args4="--must_contain S_jlQrDFDT8leLRppp"
rerank_args4="--must_contain \
S_00O7gM7Rm9SJpVkM \
S_0JRo3K0k6le07BIe \
S_0Oq4CbDhgqrrwNWT \
S_0QKuWZZo5P7Awqdn \
S_0WJK0QNkP20y80pi "
#rerank_args4=""


split=train
split=valid
seed=1
context_file="${split}_context_${seed}"


#logdir="analysis_log/${split}_${seed}_absolute_${belief}"
#logdir="analysis_log/${split}_${seed}_absolute_${belief}_${prediction_type}_b${bucket_size}"
#logdir="analysis_log/MST_${split}_${seed}_absolute_${belief}_${prediction_type}_b${bucket_size}"
#logdir="analysis_log/MST2_${split}_${seed}_absolute_${belief}_${prediction_type}_b${bucket_size}"
logdir="analysis_log/GEN_${split}_${seed}_absolute_${belief}_${prediction_type}_b${bucket_size}_g${bart_dir}"
#logdir="analysis_log/DELETE_${split}_${seed}_absolute_${belief}_${prediction_type}_b${bucket_size}"

#name="STATIC_PLAN_${split}_${seed}_absolute_${belief}_${prediction_type}_b${bucket_size}"
#name="MST_STATIC_PLAN_${split}_${seed}_absolute_${belief}_${prediction_type}_b${bucket_size}"
#name="MST2_STATIC_PLAN_${split}_${seed}_absolute_${belief}_${prediction_type}_b${bucket_size}"
name="GEN_STATIC_PLAN_${split}_${seed}_absolute_${belief}_${prediction_type}_b${bucket_size}_g${bart_dir}"
#name="DELETE_STATIC_PLAN_${split}_${seed}_absolute_${belief}_${prediction_type}_b${bucket_size}"
log_file="analysis_log/${name}.log"
out_file="analysis_log/${name}.out"

mkdir -p ${logdir}

python -u test_planning_static.py \
  --alice_model_file=${model_dir_a}/1_ep-12.th \
  --bob_model_file=${model_dir_b}/1_ep-12.th \
  --language_generator_path=./${bart_dir}/checkpoint-23000 \
  --belief ${belief} \
  --absolute_bucketing 1 \
  --num_size_buckets ${bucket_size} \
  --num_color_buckets ${bucket_size} \
  --context_file=${context_file} \
  --cuda \
  --markable_detector_file=serialized_models/markable_detector_with_dict_1.th \
  --verbose \
  --DBG_PLAN analysis_log/${name}.json \
  --dialog_log_dir ${logdir} \
  ${rerank_args} ${rerank_args2} ${rerank_args3} ${rerank_args4} \
  --log_file=${log_file} #$@

echo "LOGFILE ${log_file}"
exit 0

#switch to num_contexts 1000
