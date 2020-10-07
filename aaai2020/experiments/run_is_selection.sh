#!/bin/bash
script=""

mkdir expts/rel3_tsel_ref_dial_model_separate/is_selection 2>/dev/null

base_name="is_selection/base"
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
        --dot_recurrence self partner \
        --dot_recurrence_in next_mention selection ref partner_ref \
        --dot_recurrence_dim 64 \
        --next_mention_prediction_type=multi_reference  \
        --hidden_context_mention_encoder_count_features \
        --is_selection_prediction \
        "
### OLD
# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name} \
#         $base_args

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_1-layer \
#         $base_args \
#         --is_selection_prediction_layers 1

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_1-layer_dr-in \
#         $base_args \
#         --is_selection_prediction_layers 1 \
#         --dot_recurrence_in next_mention selection ref partner_ref is_selection 

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_dr-in \
#         $base_args \
#         --dot_recurrence_in next_mention selection ref partner_ref is_selection 

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_dr-in_turn \
#         $base_args \
#         --dot_recurrence_in next_mention selection ref partner_ref is_selection  \
#         --is_selection_prediction_turn_feature 

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_1-layer_dr-in_turn \
#         $base_args \
#         --is_selection_prediction_layers 1 \
#         --dot_recurrence_in next_mention selection ref partner_ref is_selection \
#         --is_selection_prediction_turn_feature

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_1-layer_dr-in-2_turn \
#         $base_args \
#         --is_selection_prediction_layers 1 \
#         --dot_recurrence_in next_mention selection ref partner_ref is_selection \
#         --is_selection_prediction_turn_feature

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_1-layer_dr-in-2_turn_context \
#         $base_args \
#         --is_selection_prediction_layers 1 \
#         --dot_recurrence_in next_mention selection ref partner_ref is_selection \
#         --is_selection_prediction_dot_context \
#         --is_selection_prediction_turn_feature

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_dr-in-2_turn_context \
#         $base_args \
#         --dot_recurrence_in next_mention selection ref partner_ref is_selection \
#         --is_selection_prediction_dot_context \
#         --is_selection_prediction_turn_feature

# see if this improves perplexity
# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_hcis_1-layer_dr-in_turn \
#         $base_args \
#         --hidden_context_is_selection \
#         --is_selection_prediction_layers 1 \
#         --dot_recurrence_in next_mention selection ref partner_ref is_selection \
#         --is_selection_prediction_turn_feature

## NEW, without language features
# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_1-layer_dr-in_turn \
#         $base_args \
#         --is_selection_prediction_layers 1 \
#         --dot_recurrence_in next_mention selection ref partner_ref is_selection \
#         --is_selection_prediction_features turn

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_dr-in_turn \
#         $base_args \
#         --is_selection_prediction_layers 1 \
#         --dot_recurrence_in next_mention selection ref partner_ref is_selection \
#         --is_selection_prediction_features turn

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_turn \
#         $base_args \
#         --is_selection_prediction_layers 1 \
#         --dot_recurrence_in next_mention selection ref partner_ref \
#         --is_selection_prediction_features turn

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_1-layer_dr-in_turn_dot-context \
#         $base_args \
#         --is_selection_prediction_layers 1 \
#         --dot_recurrence_in next_mention selection ref partner_ref is_selection \
#         --is_selection_prediction_features turn dot_context

${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
        ${base_name}_1-layer_dr-in_turn_dot-context_language \
        $base_args \
        --is_selection_prediction_layers 1 \
        --dot_recurrence_in next_mention selection ref partner_ref is_selection \
        --is_selection_prediction_features turn dot_context language_state
