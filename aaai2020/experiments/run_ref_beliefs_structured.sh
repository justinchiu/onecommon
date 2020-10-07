#!/bin/bash

#script="./sbatch_gpu.sh"
# script="./sbatch_1080ti.sh"
script=""

# partner ref prediction

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         hierarchical_only-self_partial-structured-attention_partner-ref_rb-last-partner-mentioned_mask-pad_h2o-1 \
#         --model_type hierarchical_rnn_reference_model \
#         --max_epoch 30 \
#         --lang_only_self \
#         --hid2output 1-hidden-layer  \
#         --partner_reference_prediction \
#         --structured_attention \
#         --ref_beliefs last_partner_mentioned 

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         hierarchical_only-self_partner-ref_rb-last-partner-mentioned_mask-pad_h2o-1 \
#         --model_type hierarchical_rnn_reference_model \
#         --max_epoch 30 \
#         --lang_only_self \
#         --hid2output 1-hidden-layer  \
#         --partner_reference_prediction \
#         --ref_beliefs last_partner_mentioned 

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         hierarchical_only-self_partner-ref_prb-last-mentioned_mask-pad_h2o-1 \
#         --model_type hierarchical_rnn_reference_model \
#         --max_epoch 30 \
#         --lang_only_self \
#         --hid2output 1-hidden-layer  \
#         --partner_reference_prediction \
#         --partner_ref_beliefs last_mentioned

${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
        hierarchical_only-self_partner-ref_mask-pad_h2o-1 \
        --model_type hierarchical_rnn_reference_model \
        --max_epoch 30 \
        --lang_only_self \
        --hid2output 1-hidden-layer  \
        --partner_reference_prediction

${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
        hierarchical_only-self_partner-ref_prb-last-mentioned+last-partner-mentioned_mask-pad_h2o-1 \
        --model_type hierarchical_rnn_reference_model \
        --max_epoch 30 \
        --lang_only_self \
        --hid2output 1-hidden-layer  \
        --partner_reference_prediction \
        --partner_ref_beliefs last_mentioned last_partner_mentioned

${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
        hierarchical_only-self_partner-ref_prb-last-mentioned+last-partner-mentioned+cumulative-mentioned+cumulative-partner-mentioned_mask-pad_h2o-1 \
        --model_type hierarchical_rnn_reference_model \
        --max_epoch 30 \
        --lang_only_self \
        --hid2output 1-hidden-layer  \
        --partner_reference_prediction \
        --partner_ref_beliefs last_mentioned last_partner_mentioned cumulative_mentioned cumulative_partner_mentioned
