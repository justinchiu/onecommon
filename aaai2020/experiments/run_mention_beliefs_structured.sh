#!/bin/bash

#script="./sbatch_gpu.sh"
# script="./sbatch_1080ti.sh"
script=""

# next mention prediction
${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
        hierarchical_only-self_partial-structured-attention-no-marg_mbfix_next-mention_mask-pad_h2o-1 \
        --model_type hierarchical_rnn_reference_model \
        --max_epoch 30 \
        --lang_only_self \
        --hid2output 1-hidden-layer  \
        --next_mention_prediction \
        --structured_attention \
        --structured_attention_no_marginalize

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         hierarchical_only-self_partial-structured-attention-no-marg_mbfix-this-partner-mentioned_next-mention_mask-pad_h2o-1 \
#         --model_type hierarchical_rnn_reference_model \
#         --max_epoch 30 \
#         --lang_only_self \
#         --hid2output 1-hidden-layer  \
#         --next_mention_prediction \
#         --mention_beliefs this_partner_mentioned  \
#         --structured_attention \
#         --structured_attention_no_marginalize

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         hierarchical_only-self_partial-structured-attention-no-marg_partner-ref_mask-pad_h2o-1 \
#         --model_type hierarchical_rnn_reference_model \
#         --max_epoch 30 \
#         --lang_only_self \
#         --hid2output 1-hidden-layer  \
#         --partner_reference_prediction \
#         --structured_attention \
#         --structured_attention_no_marginalize

${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
        hierarchical_only-self_partial-structured-attention_mbfix_next-mention_mask-pad_h2o-1 \
        --model_type hierarchical_rnn_reference_model \
        --max_epoch 30 \
        --lang_only_self \
        --hid2output 1-hidden-layer  \
        --next_mention_prediction \
        --structured_attention

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         hierarchical_only-self_partial-structured-attention_mbfix-this-partner-mentioned_next-mention_mask-pad_h2o-1 \
#         --model_type hierarchical_rnn_reference_model \
#         --max_epoch 30 \
#         --lang_only_self \
#         --hid2output 1-hidden-layer  \
#         --next_mention_prediction \
#         --mention_beliefs this_partner_mentioned  \
#         --structured_attention

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         hierarchical_only-self_partial-structured-attention_partner-ref_mask-pad_h2o-1 \
#         --model_type hierarchical_rnn_reference_model \
#         --max_epoch 30 \
#         --lang_only_self \
#         --hid2output 1-hidden-layer  \
#         --partner_reference_prediction \
#         --structured_attention
