#!/bin/bash

model_dir_a=expts/rel3_tsel_ref_dial_model_separate/nov-15/plain-hierarchical-structured-recurrence/1
model_dir_b=expts/rel3_tsel_ref_dial_model_separate/nov-15/plain-hierarchical-structured-recurrence/1

shared_ctx_count=4

num_contexts=1000
#num_contexts=200

num_buckets=5
num_buckets=3

# belief entropy threshold (deprecated)
#threshold=3
#threshold=2.75
#threshold=2
#threshold=1.5
#threshold=1

# max dot probability belief threshold
threshold=0.8

#belief="or"
#belief="or_egocentric"
belief="cost"
belief="cost_egocentric"


#absolute_bucketing=0
absolute_bucketing=1

#select_config_size=2
select_config_size=3
#select_config_size=4

# best
#length_coef=0.05
length_coef=0
# finished
#length_coef=0.075
#length_coef=0.1
#length_coef=0.125
# too much below here
#length_coef=0.22
#length_coef=0.33

diam_coef=0
# want to run
#diam_coef=4
# running
#diam_coef=0.2
#diam_coef=0.5
# ran, too small
#diam_coef=0.025
#diam_coef=0.05
#diam_coef=0.075
#diam_coef=0.1
# too big
#diam_coef=1
#diam_coef=2

contig_coef=0
#contig_coef=1

name="SS_N${num_contexts}_T${threshold}_B${belief}_AB${absolute_bucketing}_L${length_coef}_S${select_config_size}_D${diam_coef}_C${contig_coef}"
name="DBG_RM"

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

logdir="analysis_log/${name}_1"

mkdir -p ${logdir}

python -u selfplay.py \
  --num_contexts ${num_contexts} \
  --alice_model_file=${model_dir_a}/1_ep-12.th \
  --bob_model_file=${model_dir_b}/1_ep-12.th \
  --context_file=shared_4 \
  --belief_alice --belief_bob \
  --belief ${belief} \
  --belief_threshold ${threshold} \
  --symbolic \
  --absolute_bucketing ${absolute_bucketing} \
  --num_size_buckets ${num_buckets} \
  --num_color_buckets ${num_buckets} \
  --length_coef ${length_coef} \
  --diameter_coef ${diam_coef} \
  --contiguity_coef ${contig_coef} \
  --select_config_size ${select_config_size} \
  --cuda \
  --markable_detector_file=serialized_models/markable_detector_with_dict_1.th \
  --verbose \
  ${rerank_args} ${rerank_args2} ${rerank_args3} \
  --dialog_log_dir ${logdir} \
  --log_file=${log_file} $@

echo "LOGFILE ${log_file}"
exit 0

#switch to num_contexts 1000
