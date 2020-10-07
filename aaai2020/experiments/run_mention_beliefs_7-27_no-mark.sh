#!/bin/bash

#script="./sbatch_gpu.sh"
# script="./sbatch_1080ti.sh"
script=""

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         mention_beliefs/hierarchical_only-self_partial-structured-attention-no-marg_stat-relational_partner-ref_next-mention_mask-pad_h2o-1_sigmoid \
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
#         --next_mention_prediction

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         mention_beliefs/hierarchical_only-self_partial-structured-attention-no-marg_stat-relational_partner-ref_next-mention_mask-pad_h2o-1_sigmoid_mb-partners \
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
#         --mention_beliefs partners 

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         mention_beliefs/hierarchical_only-self_partial-structured-attention-no-marg_stat-relational_partner-ref_next-mention_mask-pad_h2o-1_sigmoid_mb-lpm \
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
#         --mention_beliefs last_partner_mentioned

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         mention_beliefs/hierarchical_only-self_partial-structured-attention-no-marg_stat-relational_partner-ref_next-mention_mask-pad_h2o-1_sigmoid_mb-cpm \
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
#         --mention_beliefs cumulative_partner_mentioned

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         mention_beliefs/hierarchical_only-self_partial-structured-attention-no-marg_stat-relational_partner-ref_next-mention_mask-pad_h2o-1_sigmoid_mb-lm \
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
#         --mention_beliefs last_mentioned

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         mention_beliefs/hierarchical_only-self_partial-structured-attention-no-marg_stat-relational_partner-ref_next-mention_mask-pad_h2o-1_sigmoid_mb-lm-lpm \
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
#         --mention_beliefs last_mentioned last_partner_mentioned

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         mention_beliefs/hierarchical_only-self_partial-structured-attention-no-marg_stat-relational_partner-ref_next-mention_mask-pad_h2o-1_sigmoid_mb-cm \
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
#         --mention_beliefs cumulative_mentioned

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         mention_beliefs/hierarchical_only-self_partial-structured-attention-no-marg_stat-relational_partner-ref_next-mention_mask-pad_h2o-1_sigmoid_mb-lm-cm-lpm-cpm \
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
#         --mention_beliefs last_mentioned cumulative_mentioned last_partner_mentioned cumulative_partner_mentioned 

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         mention_beliefs/hierarchical_only-self_partial-structured-attention-no-marg_stat-relational_partner-ref_next-mention_mask-pad_h2o-1_sigmoid_mb-lmpred-lpmpred \
#         --model_type hierarchical_rnn_reference_model \
#         --max_epoch 40 \
#         --lang_only_self \
#         --structured_attention \
#         --structured_attention_no_marginalize \
#         --structured_temporal_attention  \
#         --structured_temporal_attention_transitions relational  \
#         --hid2output 1-hidden-layer  \
#         --attention_type sigmoid \
#         --partner_reference_prediction \
#         --next_mention_prediction \
#         --mention_beliefs last_mentioned_predicted last_partner_mentioned_predicted \
#         --detach_beliefs \
#         --next_mention_start_epoch 10 

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         mention_beliefs/hierarchical_only-self_partial-structured-attention-no-marg_stat-relational_partner-ref_next-mention_mask-pad_h2o-1_sigmoid_mb-tpm \
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
#         --mention_beliefs this_partner_mentioned

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         mention_beliefs/hierarchical_only-self_partial-structured-attention-no-marg_stat-relational_partner-ref_next-mention_mask-pad_h2o-1_sigmoid_mb-tpmpred \
#         --model_type hierarchical_rnn_reference_model \
#         --max_epoch 40 \
#         --lang_only_self \
#         --structured_attention \
#         --structured_attention_no_marginalize \
#         --structured_temporal_attention  \
#         --structured_temporal_attention_transitions relational  \
#         --hid2output 1-hidden-layer  \
#         --attention_type sigmoid \
#         --partner_reference_prediction \
#         --next_mention_prediction \
#         --mention_beliefs this_partner_mentioned_predicted \
#         --detach_beliefs \
#         --next_mention_start_epoch 10 

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         mention_beliefs/hierarchical_only-self_partial-structured-attention-no-marg_stat-relational_partner-ref_next-mention_mask-pad_h2o-1_sigmoid_mb-tpm-lm_refac \
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
        mention_beliefs/hierarchical_only-self_partial-structured-attention-no-marg_stat-relational_partner-ref_next-mention_mask-pad_h2o-1_sigmoid_mb-tpm-lm_embed-beliefs \
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

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         mention_beliefs/hierarchical_only-self_partial-structured-attention-no-marg_stat-relational_partner-ref_next-mention_mask-pad_h2o-1_sigmoid_mb-tpmpred-lmpred \
#         --model_type hierarchical_rnn_reference_model \
#         --max_epoch 40 \
#         --lang_only_self \
#         --structured_attention \
#         --structured_attention_no_marginalize \
#         --structured_temporal_attention  \
#         --structured_temporal_attention_transitions relational  \
#         --hid2output 1-hidden-layer  \
#         --attention_type sigmoid \
#         --partner_reference_prediction \
#         --next_mention_prediction \
#         --mention_beliefs this_partner_mentioned_predicted last_mentioned_predicted \
#         --detach_beliefs \
#         --next_mention_start_epoch 10 
