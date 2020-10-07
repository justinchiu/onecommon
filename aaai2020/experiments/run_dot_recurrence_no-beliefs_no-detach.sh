#!/bin/bash
script=""

mkdir expts/rel3_tsel_ref_dial_model_separate/dot_recurrence_no-beliefs 2>/dev/null

base_name="dot_recurrence_no-beliefs/no-detach"
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
        --reduce_plateau \
        "

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name} \
#         $base_args

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_mention-selection \
#         $base_args \
#         --dot_recurrence self partner \
#         --dot_recurrence_in next_mention selection \
#         --dot_recurrence_dim 64

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_mention-selection-ref-partner \
#         $base_args \
#         --dot_recurrence self partner \
#         --dot_recurrence_in next_mention selection ref partner_ref \
#         --dot_recurrence_dim 64

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_oracle_mention-selection-ref-partner \
#         $base_args \
#         --dot_recurrence self partner \
#         --dot_recurrence_in next_mention selection ref partner_ref \
#         --dot_recurrence_dim 64 \
#         --dot_recurrence_oracle

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_mention-selection-ref-partner_nmpt-multi \
#         $base_args \
#         --dot_recurrence self partner \
#         --dot_recurrence_in next_mention selection ref partner_ref \
#         --dot_recurrence_dim 64 \
#         --next_mention_prediction_type=multi_reference 

# SIZE 64 with NMPT
# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_mention-selection-ref-partner_nmpt-multi_loss-expanded \
#         $base_args \
#         --dot_recurrence self partner \
#         --dot_recurrence_in next_mention selection ref partner_ref \
#         --dot_recurrence_dim 64 \
#         --next_mention_prediction_type=multi_reference 

# latent refs
# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_latent-ref_mention-selection_nmpt-multi_loss-expanded \
#         $base_args \
#         --ref_weight 0.0 \
#         --partner_ref_weight 0.0 \
#         --dot_recurrence self partner \
#         --dot_recurrence_in next_mention selection \
#         --dot_recurrence_dim 64 \
#         --next_mention_prediction_type=multi_reference 

${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
        ${base_name}_latent-ref_mention-selection_nmpt-multi_loss-expanded_structured-0 \
        $base_args \
        --ref_weight 0.0 \
        --partner_ref_weight 0.0 \
        --dot_recurrence self partner \
        --dot_recurrence_in next_mention selection \
        --dot_recurrence_dim 64 \
        --dot_recurrence_structured \
        --next_mention_prediction_type=multi_reference 

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_mention-selection-ref-partner_nmpt-multi_loss-expanded_ss-ms-10 \
#         $base_args \
#         --selection_start_epoch 10 \
#         --next_mention_start_epoch 10 \
#         --dot_recurrence self partner \
#         --dot_recurrence_in next_mention selection ref partner_ref \
#         --dot_recurrence_dim 64 \
#         --next_mention_prediction_type=multi_reference 

## SIZE 1 with NMPT
# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_dr-1_mention-selection-ref-partner_nmpt-multi_loss-expanded \
#         $base_args \
#         --dot_recurrence self partner \
#         --dot_recurrence_in next_mention selection ref partner_ref \
#         --dot_recurrence_dim 1 \
#         --next_mention_prediction_type=multi_reference 

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_dr-1_mention-selection-ref-partner_nmpt-multi_loss-expanded_ss-ms-10 \
#         $base_args \
#         --selection_start_epoch 10 \
#         --next_mention_start_epoch 10 \
#         --dot_recurrence self partner \
#         --dot_recurrence_in next_mention selection ref partner_ref \
#         --dot_recurrence_dim 1 \
#         --next_mention_prediction_type=multi_reference 

# SIZE 64, split with NMPT 
# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_split_mention-selection-ref-partner_nmpt-multi_loss-expanded \
#         $base_args \
#         --dot_recurrence self partner \
#         --dot_recurrence_in next_mention selection ref partner_ref \
#         --dot_recurrence_dim 64 \
#         --dot_recurrence_split \
#         --next_mention_prediction_type=multi_reference 

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_split_mention-ref-partner_nmpt-multi_loss-expanded \
#         $base_args \
#         --dot_recurrence self partner \
#         --dot_recurrence_in next_mention ref partner_ref \
#         --dot_recurrence_dim 64 \
#         --dot_recurrence_split \
#         --next_mention_prediction_type=multi_reference 

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_split_mention_nmpt-multi_loss-expanded \
#         $base_args \
#         --dot_recurrence self partner \
#         --dot_recurrence_in next_mention \
#         --dot_recurrence_dim 64 \
#         --dot_recurrence_split \
#         --next_mention_prediction_type=multi_reference 

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_split_mention-selection-ref-partner_nmpt-multi_loss-expanded_ss-ms-10 \
#         $base_args \
#         --selection_start_epoch 10 \
#         --next_mention_start_epoch 10 \
#         --dot_recurrence self partner \
#         --dot_recurrence_in next_mention selection ref partner_ref \
#         --dot_recurrence_dim 64 \
#         --dot_recurrence_split \
#         --next_mention_prediction_type=multi_reference 

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_split_mention-selection_nmpt-multi_loss-expanded_ss-ms-10 \
#         $base_args \
#         --selection_start_epoch 10 \
#         --next_mention_start_epoch 10 \
#         --dot_recurrence self partner \
#         --dot_recurrence_in next_mention selection \
#         --dot_recurrence_dim 64 \
#         --dot_recurrence_split \
#         --next_mention_prediction_type=multi_reference 

# SIZE 64, split with NMPT and structured

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_split_mention-selection-ref-partner_nmpt-multi_loss-expanded_structured-0 \
#         $base_args \
#         --dot_recurrence self partner \
#         --dot_recurrence_in next_mention selection ref partner_ref \
#         --dot_recurrence_dim 64 \
#         --dot_recurrence_split \
#         --dot_recurrence_structured \
#         --next_mention_prediction_type=multi_reference 

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_split_mention-selection-ref-partner_nmpt-multi_loss-expanded_structured-1 \
#         $base_args \
#         --dot_recurrence self partner \
#         --dot_recurrence_in next_mention selection ref partner_ref \
#         --dot_recurrence_dim 64 \
#         --dot_recurrence_split \
#         --dot_recurrence_structured \
#         --dot_recurrence_structured_layers=1 \
#         --next_mention_prediction_type=multi_reference 

# SIZE 64, no split with NMPT and structured

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_mention-selection-ref-partner_nmpt-multi_loss-expanded_structured-0 \
#         $base_args \
#         --dot_recurrence self partner \
#         --dot_recurrence_in next_mention selection ref partner_ref \
#         --dot_recurrence_dim 64 \
#         --dot_recurrence_structured \
#         --next_mention_prediction_type=multi_reference 

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_mention-selection-ref-partner_nmpt-multi_loss-expanded_structured-1 \
#         $base_args \
#         --dot_recurrence self partner \
#         --dot_recurrence_in next_mention selection ref partner_ref \
#         --dot_recurrence_dim 64 \
#         --dot_recurrence_structured \
#         --dot_recurrence_structured_layers=1 \
#         --next_mention_prediction_type=multi_reference 
