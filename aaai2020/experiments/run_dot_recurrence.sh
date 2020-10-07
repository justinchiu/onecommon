#!/bin/bash
script=""

mkdir expts/rel3_tsel_ref_dial_model_separate/dot_recurrence 2>/dev/null

base_name="dot_recurrence/base"
base_args="--model_type hierarchical_rnn_reference_model \
        --max_epoch 20 \
        --lang_only_self \
        --structured_attention \
        --structured_attention_no_marginalize \
        --structured_temporal_attention  \
        --structured_temporal_attention_transitions relational  \
        --structured_attention_language_conditioned \
        --mark_dots_mentioned \
        --word_attention_constrained \
        --hid2output 1-hidden-layer  \
        --attention_type sigmoid \
        --partner_reference_prediction \
        --next_mention_prediction \
        --nhid_lang 512 \
        --encode_relative_to_extremes \
        --learned_pooling \
        --untie_grus \
        --bidirectional_reader \
        --hidden_context \
        --hidden_context_mention_encoder \
        --detach_beliefs \
        --reduce_plateau \
        --ref_beliefs last_mentioned_predicted last_partner_mentioned_predicted last_primary_mentioned_predicted last_partner_primary_mentioned_predicted \
        --ref_beliefs_warmup last_mentioned last_partner_mentioned last_primary_mentioned last_partner_primary_mentioned \
        --ref_beliefs_warmup_epochs 10 \
        --partner_ref_beliefs last_mentioned_predicted last_partner_mentioned_predicted last_primary_mentioned_predicted last_partner_primary_mentioned_predicted \
        --partner_ref_beliefs_warmup last_mentioned last_partner_mentioned last_primary_mentioned last_partner_primary_mentioned \
        --partner_ref_beliefs_warmup_epochs 10 \
        "

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_mb-tpmpred-lmpred_sb-tpmpred-lmpred \
#         $base_args \
#         --mention_beliefs this_partner_mentioned_predicted last_mentioned_predicted \
#         --selection_beliefs this_partner_mentioned_predicted last_mentioned_predicted \
#         --selection_start_epoch 10 \
#         --next_mention_start_epoch 10

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_dr-self+partner-32-mention-selection \
#         $base_args \
#         --selection_start_epoch 10 \
#         --next_mention_start_epoch 10 \
#         --dot_recurrence self partner \
#         --dot_recurrence_in next_mention selection \
#         --dot_recurrence_dim 32

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_dr-self-32-mention-selection \
#         $base_args \
#         --selection_start_epoch 10 \
#         --next_mention_start_epoch 10 \
#         --dot_recurrence self \
#         --dot_recurrence_in next_mention selection \
#         --dot_recurrence_dim 32

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_dr-partner-32-mention-selection \
#         $base_args \
#         --selection_start_epoch 10 \
#         --next_mention_start_epoch 10 \
#         --dot_recurrence partner \
#         --dot_recurrence_in next_mention selection \
#         --dot_recurrence_dim 32

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_dr-self+partner-1-mention-selection \
#         $base_args \
#         --selection_start_epoch 10 \
#         --next_mention_start_epoch 10 \
#         --dot_recurrence self partner \
#         --dot_recurrence_in next_mention selection \
#         --dot_recurrence_dim 1

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_dr-oracle-self+partner-32-mention-selection \
#         $base_args \
#         --selection_start_epoch 10 \
#         --next_mention_start_epoch 10 \
#         --dot_recurrence self partner \
#         --dot_recurrence_oracle \
#         --dot_recurrence_in next_mention selection \
#         --dot_recurrence_dim 32

## other sizes
# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_dr-self+partner-64-mention-selection \
#         $base_args \
#         --selection_start_epoch 10 \
#         --next_mention_start_epoch 10 \
#         --dot_recurrence self partner \
#         --dot_recurrence_in next_mention selection \
#         --dot_recurrence_dim 64

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_dr-self+partner-128-mention-selection \
#         $base_args \
#         --selection_start_epoch 10 \
#         --next_mention_start_epoch 10 \
#         --dot_recurrence self partner \
#         --dot_recurrence_in next_mention selection \
#         --dot_recurrence_dim 128

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_dr-self+partner-64-mention-selection_nmpt-multi-reference \
#         $base_args \
#         --selection_start_epoch 10 \
#         --next_mention_start_epoch 10 \
#         --dot_recurrence self partner \
#         --dot_recurrence_in next_mention selection \
#         --dot_recurrence_dim 64 \
#         --next_mention_prediction_type=multi_reference 

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_dr-self+partner-64-mention-selection_nmpt-multi-reference_expanded \
#         $base_args \
#         --selection_start_epoch 10 \
#         --next_mention_start_epoch 10 \
#         --dot_recurrence self partner \
#         --dot_recurrence_in next_mention selection \
#         --dot_recurrence_dim 64 \
#         --next_mention_prediction_type=multi_reference 

${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
        ${base_name}_dr-self+partner-64-mention-selection-ref-partner-ref \
        --model_type hierarchical_rnn_reference_model \
        --max_epoch 20 \
        --lang_only_self \
        --structured_attention \
        --structured_attention_no_marginalize \
        --structured_temporal_attention  \
        --structured_temporal_attention_transitions relational  \
        --structured_attention_language_conditioned \
        --mark_dots_mentioned \
        --word_attention_constrained \
        --hid2output 1-hidden-layer  \
        --attention_type sigmoid \
        --partner_reference_prediction \
        --next_mention_prediction \
        --nhid_lang 512 \
        --encode_relative_to_extremes \
        --learned_pooling \
        --untie_grus \
        --bidirectional_reader \
        --hidden_context \
        --hidden_context_mention_encoder \
        --detach_beliefs \
        --reduce_plateau \
        --selection_start_epoch 10 \
        --next_mention_start_epoch 10 \
        --dot_recurrence self partner \
        --dot_recurrence_in next_mention selection ref partner_ref \
        --dot_recurrence_dim 64
