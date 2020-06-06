#!/bin/bash

#script="./sbatch_gpu.sh"
script="./sbatch_1080ti.sh"

# next mention prediction
${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
        hierarchical_only-self_mb-this-mentioned_next-mention_mask-pad_h2o-1_ep-60 \
        --model_type hierarchical_rnn_reference_model \
        --max_epoch 60 \
        --lang_only_self \
        --hid2output 1-hidden-layer  \
        --next_mention_prediction \
        --mention_beliefs this_mentioned 

${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
        hierarchical_only-self_gb-this-mentioned_mask-pad_h2o-1_ep-60 \
        --model_type hierarchical_rnn_reference_model \
        --max_epoch 60 \
        --lang_only_self \
        --hid2output 1-hidden-layer  \
        --generation_beliefs this_mentioned

${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
        hierarchical_only-self_mark-dots-mentioned_mask-pad_h2o-1_ep-60 \
        --model_type hierarchical_rnn_reference_model \
        --max_epoch 60 \
        --lang_only_self \
        --hid2output 1-hidden-layer  \
        --mark_dots_mentioned

${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
        hierarchical_only-self_mb-last-partner-mentioned_next-mention_mask-pad_h2o-1_ep-60 \
        --model_type hierarchical_rnn_reference_model \
        --max_epoch 60 \
        --lang_only_self \
        --hid2output 1-hidden-layer  \
        --next_mention_prediction \
        --mention_beliefs last_partner_mentioned 

${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
        hierarchical_only-self_mb-cumulative-partner-mentioned_next-mention_mask-pad_h2o-1_ep-60 \
        --model_type hierarchical_rnn_reference_model \
        --max_epoch 60 \
        --lang_only_self \
        --hid2output 1-hidden-layer  \
        --next_mention_prediction \
        --mention_beliefs cumulative_partner_mentioned 
