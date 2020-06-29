#!/bin/bash

#script="./sbatch_gpu.sh"
# script="./sbatch_1080ti.sh"
script=""

# default
# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         hierarchical_only-self_mask-pad_h2o-1 \
#         --model_type hierarchical_rnn_reference_model \
#         --max_epoch 30 \
#         --lang_only_self \
#         --hid2output 1-hidden-layer 

## generation beliefs
## gb-this-mentioned and mark-dots-mentioned have similar stats (as they probably should)
# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         hierarchical_only-self_gb-this-mentioned_mask-pad_h2o-1 \
#         --model_type hierarchical_rnn_reference_model \
#         --max_epoch 30 \
#         --lang_only_self \
#         --hid2output 1-hidden-layer  \
#         --generation_beliefs this_mentioned

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         hierarchical_only-self_mark-dots-mentioned_mask-pad_h2o-1 \
#         --model_type hierarchical_rnn_reference_model \
#         --max_epoch 30 \
#         --lang_only_self \
#         --hid2output 1-hidden-layer  \
#         --mark_dots_mentioned

# next mention prediction
# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         hierarchical_only-self_mbfix_next-mention_mask-pad_h2o-1 \
#         --model_type hierarchical_rnn_reference_model \
#         --max_epoch 30 \
#         --lang_only_self \
#         --hid2output 1-hidden-layer  \
#         --next_mention_prediction

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         hierarchical_only-self_mbfix-this-mentioned_next-mention_mask-pad_h2o-1 \
#         --model_type hierarchical_rnn_reference_model \
#         --max_epoch 30 \
#         --lang_only_self \
#         --hid2output 1-hidden-layer  \
#         --next_mention_prediction \
#         --mention_beliefs this_mentioned 

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         hierarchical_only-self_mbfix-next-mentioned_next-mention_mask-pad_h2o-1 \
#         --model_type hierarchical_rnn_reference_model \
#         --max_epoch 30 \
#         --lang_only_self \
#         --hid2output 1-hidden-layer  \
#         --next_mention_prediction \
#         --mention_beliefs next_mentioned 

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         hierarchical_only-self_mbfix-this-partner-mentioned_next-mention_mask-pad_h2o-1 \
#         --model_type hierarchical_rnn_reference_model \
#         --max_epoch 30 \
#         --lang_only_self \
#         --hid2output 1-hidden-layer  \
#         --next_mention_prediction \
#         --mention_beliefs this_partner_mentioned 

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         hierarchical_only-self_mbfix-last-partner-mentioned_next-mention_mask-pad_h2o-1 \
#         --model_type hierarchical_rnn_reference_model \
#         --max_epoch 30 \
#         --lang_only_self \
#         --hid2output 1-hidden-layer  \
#         --next_mention_prediction \
#         --mention_beliefs last_partner_mentioned 

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         hierarchical_only-self_mbfix-cumulative-partner-mentioned_next-mention_mask-pad_h2o-1 \
#         --model_type hierarchical_rnn_reference_model \
#         --max_epoch 30 \
#         --lang_only_self \
#         --hid2output 1-hidden-layer  \
#         --next_mention_prediction \
#         --mention_beliefs cumulative_partner_mentioned 

# this-partner-mentioned-predicted
# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         hierarchical_only-self_partner-ref_next-mention_mask-pad_h2o-1 \
#         --model_type hierarchical_rnn_reference_model \
#         --max_epoch 30 \
#         --lang_only_self \
#         --partner_reference_prediction \
#         --hid2output 1-hidden-layer  \
#         --next_mention_prediction

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         hierarchical_only-self_partial-structured-attention_partner-ref_next-mention_mask-pad_h2o-1 \
#         --model_type hierarchical_rnn_reference_model \
#         --max_epoch 30 \
#         --lang_only_self \
#         --partner_reference_prediction \
#         --hid2output 1-hidden-layer  \
#         --next_mention_prediction \
#         --structured_attention

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         hierarchical_only-self_partner-ref_mbfix-this-partner-mentioned-predicted_next-mention_mask-pad_h2o-1 \
#         --model_type hierarchical_rnn_reference_model \
#         --max_epoch 30 \
#         --lang_only_self \
#         --partner_reference_prediction \
#         --hid2output 1-hidden-layer  \
#         --next_mention_prediction \
#         --mention_beliefs this_partner_mentioned_predicted

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         hierarchical_only-self_partner-ref_detach_mbfix-this-partner-mentioned-predicted_next-mention_mask-pad_h2o-1 \
#         --model_type hierarchical_rnn_reference_model \
#         --max_epoch 30 \
#         --lang_only_self \
#         --partner_reference_prediction \
#         --hid2output 1-hidden-layer  \
#         --next_mention_prediction \
#         --mention_beliefs this_partner_mentioned_predicted \
#         --detach_beliefs

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         hierarchical_only-self_partial-structured-attention_partner-ref_detach_mbfix-this-partner-mentioned-predicted_next-mention_mask-pad_h2o-1 \
#         --model_type hierarchical_rnn_reference_model \
#         --structured_attention \
#         --max_epoch 30 \
#         --lang_only_self \
#         --partner_reference_prediction \
#         --hid2output 1-hidden-layer  \
#         --next_mention_prediction \
#         --mention_beliefs this_partner_mentioned_predicted \
#         --detach_beliefs

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         hierarchical_only-self_mbfix-this-partner-mentioned_next-mention_mask-pad_h2o-1_refac \
#         --model_type hierarchical_rnn_reference_model \
#         --max_epoch 30 \
#         --lang_only_self \
#         --hid2output 1-hidden-layer  \
#         --next_mention_prediction \
#         --mention_beliefs this_partner_mentioned 

${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
        hierarchical_only-self_partial-structured-attention_partner-ref_detach_mbstart-10_mbfix-this-partner-mentioned-predicted_next-mention_mask-pad_h2o-1 \
        --model_type hierarchical_rnn_reference_model \
        --structured_attention \
        --max_epoch 40 \
        --next_mention_start_epoch 10 \
        --lang_only_self \
        --partner_reference_prediction \
        --hid2output 1-hidden-layer  \
        --next_mention_prediction \
        --mention_beliefs this_partner_mentioned_predicted \
        --detach_beliefs
