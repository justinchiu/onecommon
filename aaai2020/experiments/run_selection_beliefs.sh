#!/bin/bash
script=""
# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         selection_beliefs/hierarchical_only-self_mark-dots-mentioned_word-attention-constrained_partner-ref_next-mention_mask-pad_h2o-1_sigmoid \
#         --model_type hierarchical_rnn_reference_model \
#         --max_epoch 30 \
#         --lang_only_self \
#         --mark_dots_mentioned \
#         --word_attention_constrained \
#         --hid2output 1-hidden-layer  \
#         --attention_type sigmoid \
#         --partner_reference_prediction \
#         --next_mention_prediction

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         selection_beliefs/hierarchical_only-self_mark-dots-mentioned_word-attention-constrained_partner-ref_next-mention_mask-pad_h2o-1_sigmoid_sb-partners \
#         --model_type hierarchical_rnn_reference_model \
#         --max_epoch 30 \
#         --lang_only_self \
#         --mark_dots_mentioned \
#         --word_attention_constrained \
#         --hid2output 1-hidden-layer  \
#         --attention_type sigmoid \
#         --partner_reference_prediction \
#         --next_mention_prediction \
#         --selection_beliefs partners 

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         selection_beliefs/hierarchical_only-self_mark-dots-mentioned_word-attention-constrained_partner-ref_next-mention_mask-pad_h2o-1_sigmoid_sb-lpm \
#         --model_type hierarchical_rnn_reference_model \
#         --max_epoch 30 \
#         --lang_only_self \
#         --mark_dots_mentioned \
#         --word_attention_constrained \
#         --hid2output 1-hidden-layer  \
#         --attention_type sigmoid \
#         --partner_reference_prediction \
#         --next_mention_prediction \
#         --selection_beliefs last_partner_mentioned

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         selection_beliefs/hierarchical_only-self_mark-dots-mentioned_word-attention-constrained_partner-ref_next-mention_mask-pad_h2o-1_sigmoid_sb-cpm \
#         --model_type hierarchical_rnn_reference_model \
#         --max_epoch 30 \
#         --lang_only_self \
#         --mark_dots_mentioned \
#         --word_attention_constrained \
#         --hid2output 1-hidden-layer  \
#         --attention_type sigmoid \
#         --partner_reference_prediction \
#         --next_mention_prediction \
#         --selection_beliefs cumulative_partner_mentioned

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         selection_beliefs/hierarchical_only-self_mark-dots-mentioned_word-attention-constrained_partner-ref_next-mention_mask-pad_h2o-1_sigmoid_sb-lm \
#         --model_type hierarchical_rnn_reference_model \
#         --max_epoch 30 \
#         --lang_only_self \
#         --mark_dots_mentioned \
#         --word_attention_constrained \
#         --hid2output 1-hidden-layer  \
#         --attention_type sigmoid \
#         --partner_reference_prediction \
#         --next_mention_prediction \
#         --selection_beliefs last_mentioned

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         selection_beliefs/hierarchical_only-self_mark-dots-mentioned_word-attention-constrained_partner-ref_next-mention_mask-pad_h2o-1_sigmoid_sb-lm-lpm \
#         --model_type hierarchical_rnn_reference_model \
#         --max_epoch 30 \
#         --lang_only_self \
#         --mark_dots_mentioned \
#         --word_attention_constrained \
#         --hid2output 1-hidden-layer  \
#         --attention_type sigmoid \
#         --partner_reference_prediction \
#         --next_mention_prediction \
#         --selection_beliefs last_mentioned last_partner_mentioned

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         selection_beliefs/hierarchical_only-self_mark-dots-mentioned_word-attention-constrained_partner-ref_next-mention_mask-pad_h2o-1_sigmoid_sb-cm \
#         --model_type hierarchical_rnn_reference_model \
#         --max_epoch 30 \
#         --lang_only_self \
#         --mark_dots_mentioned \
#         --word_attention_constrained \
#         --hid2output 1-hidden-layer  \
#         --attention_type sigmoid \
#         --partner_reference_prediction \
#         --next_mention_prediction \
#         --selection_beliefs cumulative_mentioned

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         selection_beliefs/hierarchical_only-self_mark-dots-mentioned_word-attention-constrained_partner-ref_next-mention_mask-pad_h2o-1_sigmoid_sb-lm-cm-lpm-cpm \
#         --model_type hierarchical_rnn_reference_model \
#         --max_epoch 30 \
#         --lang_only_self \
#         --mark_dots_mentioned \
#         --word_attention_constrained \
#         --hid2output 1-hidden-layer  \
#         --attention_type sigmoid \
#         --partner_reference_prediction \
#         --next_mention_prediction \
#         --selection_beliefs last_mentioned cumulative_mentioned last_partner_mentioned cumulative_partner_mentioned 

${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
        selection_beliefs/hierarchical_only-self_partial-structured-attention-no-marg_stat-relational_mark-dots-mentioned_word-attention-constrained_partner-ref_next-mention_mask-pad_h2o-1_sigmoid_sb-lmpred-lpmpred \
        --model_type hierarchical_rnn_reference_model \
        --max_epoch 40 \
        --lang_only_self \
        --structured_attention \
        --structured_attention_no_marginalize \
        --structured_temporal_attention  \
        --structured_temporal_attention_transitions relational  \
        --mark_dots_mentioned \
        --word_attention_constrained \
        --hid2output 1-hidden-layer  \
        --attention_type sigmoid \
        --partner_reference_prediction \
        --next_mention_prediction \
        --selection_beliefs last_mentioned_predicted last_partner_mentioned_predicted \
        --detach_beliefs \
        --selection_start_epoch 10 
