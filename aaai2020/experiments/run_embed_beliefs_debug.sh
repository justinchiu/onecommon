#!/bin/bash

#script="./sbatch_gpu.sh"
# script="./sbatch_1080ti.sh"
script=""
# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         embed_beliefs_debug/hierarchical_only-self_partner-ref_next-mention_mask-pad_h2o-1_sigmoid_mb-tpm-lm_refac-fix-tying-fix-last \
#         --model_type hierarchical_rnn_reference_model \
#         --max_epoch 30 \
#         --lang_only_self \
#         --hid2output 1-hidden-layer  \
#         --attention_type sigmoid \
#         --partner_reference_prediction \
#         --next_mention_prediction \
#         --mention_beliefs this_partner_mentioned last_mentioned

## the above, but with embedded beliefs
# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         embed_beliefs_debug/hierarchical_only-self_partner-ref_next-mention_mask-pad_h2o-1_sigmoid_mb-tpm-lm_embed-beliefs-refac-fix-tying-fix-last \
#         --model_type hierarchical_rnn_reference_model \
#         --max_epoch 30 \
#         --lang_only_self \
#         --hid2output 1-hidden-layer  \
#         --attention_type sigmoid \
#         --partner_reference_prediction \
#         --next_mention_prediction \
#         --embed_beliefs \
#         --mention_beliefs this_partner_mentioned last_mentioned

        #embed_beliefs_debug/hierarchical_only-self_partial-structured-attention-no-marg_stat-relational_partner-ref_next-mention_mask-pad_h2o-1_sigmoid_mb-tpm-lm_refac \

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         embed_beliefs_debug/hierarchical_only-self_partial-structured-attention-no-marg_stat-relational_partner-ref_next-mention_mask-pad_h2o-1_sigmoid_mb-tpm-lm_refac-fix-tying-fix-last \
#         --model_type hierarchical_rnn_reference_model \
#         --max_epoch 30 \
#         --lang_only_self \
#         --structured_attention \
#         --structured_attention_no_marginalize \
#         --structured_temporal_attention  \
#         --structured_temporal_attention_transitions relational  \
#         --hid2output 1-hidden-layer  \
#         --attention_type sigmoid \
#         --partner_reference_prediction \
#         --next_mention_prediction \
#         --mention_beliefs this_partner_mentioned last_mentioned

## the above, but with embedded beliefs
${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
        mention_beliefs/hierarchical_only-self_partial-structured-attention-no-marg_stat-relational_partner-ref_next-mention_mask-pad_h2o-1_sigmoid_mb-tpm-lm_embed-beliefs-fix-tying \
        --model_type hierarchical_rnn_reference_model \
        --max_epoch 30 \
        --lang_only_self \
        --structured_attention \
        --structured_attention_no_marginalize \
        --structured_temporal_attention  \
        --structured_temporal_attention_transitions relational  \
        --hid2output 1-hidden-layer  \
        --attention_type sigmoid \
        --partner_reference_prediction \
        --next_mention_prediction \
        --embed_beliefs \
        --mention_beliefs this_partner_mentioned last_mentioned
