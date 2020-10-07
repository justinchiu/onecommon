#!/bin/bash

#script="./sbatch_gpu.sh"
# script="./sbatch_1080ti.sh"
script=""


# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         hierarchical_only-self_partial-structured-attention_partner-ref_mbfix-this-partner-mentioned-noised-0.5-0.05_next-mention_mask-pad_h2o-1 \
#         --model_type hierarchical_rnn_reference_model \
#         --structured_attention \
#         --max_epoch 30 \
#         --lang_only_self \
#         --partner_reference_prediction \
#         --hid2output 1-hidden-layer  \
#         --next_mention_prediction \
#         --mention_beliefs this_partner_mentioned_noised \
#         --belief_noise_pos_to_neg_probability 0.5 \
#         --belief_noise_neg_to_pos_probability 0.05

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         hierarchical_only-self_partial-structured-attention_partner-ref_mbfix-this-partner-mentioned-noised-0.4-0.04_next-mention_mask-pad_h2o-1 \
#         --model_type hierarchical_rnn_reference_model \
#         --structured_attention \
#         --max_epoch 30 \
#         --lang_only_self \
#         --partner_reference_prediction \
#         --hid2output 1-hidden-layer  \
#         --next_mention_prediction \
#         --mention_beliefs this_partner_mentioned_noised \
#         --belief_noise_pos_to_neg_probability 0.4 \
#         --belief_noise_neg_to_pos_probability 0.04

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         hierarchical_only-self_partial-structured-attention_partner-ref_mbfix-this-partner-mentioned-noised-0.25-0.025_next-mention_mask-pad_h2o-1 \
#         --model_type hierarchical_rnn_reference_model \
#         --structured_attention \
#         --max_epoch 30 \
#         --lang_only_self \
#         --partner_reference_prediction \
#         --hid2output 1-hidden-layer  \
#         --next_mention_prediction \
#         --mention_beliefs this_partner_mentioned_noised \
#         --belief_noise_pos_to_neg_probability 0.25 \
#         --belief_noise_neg_to_pos_probability 0.025

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         hierarchical_only-self_partial-structured-attention_partner-ref_mbfix-this-partner-mentioned-noised-0.1-0.01_next-mention_mask-pad_h2o-1 \
#         --model_type hierarchical_rnn_reference_model \
#         --structured_attention \
#         --max_epoch 30 \
#         --lang_only_self \
#         --partner_reference_prediction \
#         --hid2output 1-hidden-layer  \
#         --next_mention_prediction \
#         --mention_beliefs this_partner_mentioned_noised \
#         --belief_noise_pos_to_neg_probability 0.1 \
#         --belief_noise_neg_to_pos_probability 0.01

${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
        hierarchical_only-self_partial-structured-attention_partner-ref_mbfix-this-partner-mentioned-noised-0.0-0.00_next-mention_mask-pad_h2o-1 \
        --model_type hierarchical_rnn_reference_model \
        --structured_attention \
        --max_epoch 30 \
        --lang_only_self \
        --partner_reference_prediction \
        --hid2output 1-hidden-layer  \
        --next_mention_prediction \
        --mention_beliefs this_partner_mentioned_noised \
        --belief_noise_pos_to_neg_probability 0.0 \
        --belief_noise_neg_to_pos_probability 0.00
