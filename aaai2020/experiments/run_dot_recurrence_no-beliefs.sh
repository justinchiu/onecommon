#!/bin/bash
script=""

mkdir expts/rel3_tsel_ref_dial_model_separate/dot_recurrence_no-beliefs 2>/dev/null

base_name="dot_recurrence_no-beliefs/base"
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
        "

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name} \
#         $base_args

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_nmpt-multi_loss-expanded \
#         $base_args \
#         --next_mention_prediction_type=multi_reference 

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
#         ${base_name}_oracle_mention-selection-ref-partner_refac \
#         $base_args \
#         --dot_recurrence self partner \
#         --dot_recurrence_in next_mention selection ref partner_ref \
#         --dot_recurrence_dim 64 \
#         --dot_recurrence_oracle

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
#         ${base_name}_mention-selection_nmpt-multi_loss-expanded \
#         $base_args \
#         --dot_recurrence self partner \
#         --dot_recurrence_in next_mention selection \
#         --dot_recurrence_dim 64 \
#         --next_mention_prediction_type=multi_reference 

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_mention_nmpt-multi_loss-expanded \
#         $base_args \
#         --dot_recurrence self partner \
#         --dot_recurrence_in next_mention \
#         --dot_recurrence_dim 64 \
#         --next_mention_prediction_type=multi_reference 

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_selection_nmpt-multi_loss-expanded \
#         $base_args \
#         --dot_recurrence self partner \
#         --dot_recurrence_in selection \
#         --dot_recurrence_dim 64 \
#         --next_mention_prediction_type=multi_reference 

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_oracle_mention-selection-ref-partner_nmpt-multi_loss-expanded_rerun \
#         $base_args \
#         --dot_recurrence self partner \
#         --dot_recurrence_in next_mention selection ref partner_ref \
#         --dot_recurrence_dim 64 \
#         --dot_recurrence_oracle \
#         --next_mention_prediction_type=multi_reference 

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_oracle-self_mention-selection-ref-partner_nmpt-multi_loss-expanded \
#         $base_args \
#         --dot_recurrence self partner \
#         --dot_recurrence_in next_mention selection ref partner_ref \
#         --dot_recurrence_dim 64 \
#         --dot_recurrence_oracle \
#         --dot_recurrence_oracle_for self \
#         --next_mention_prediction_type=multi_reference 

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_oracle-partner_mention-selection-ref-partner_nmpt-multi_loss-expanded \
#         $base_args \
#         --dot_recurrence self partner \
#         --dot_recurrence_in next_mention selection ref partner_ref \
#         --dot_recurrence_dim 64 \
#         --dot_recurrence_oracle \
#         --dot_recurrence_oracle_for partner \
#         --next_mention_prediction_type=multi_reference 

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_dr-1_oracle_mention-selection-ref-partner_nmpt-multi_loss-expanded \
#         $base_args \
#         --dot_recurrence self partner \
#         --dot_recurrence_in next_mention selection ref partner_ref \
#         --dot_recurrence_dim 1 \
#         --dot_recurrence_oracle \
#         --next_mention_prediction_type=multi_reference 

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_mention-selection-ref-partner_nmpt-multi_loss-expanded_uniform \
#         $base_args \
#         --dot_recurrence self partner \
#         --dot_recurrence_in next_mention selection ref partner_ref \
#         --dot_recurrence_dim 64 \
#         --dot_recurrence_uniform \
#         --next_mention_prediction_type=multi_reference 

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_ref-partner_nmpt-multi_loss-expanded \
#         $base_args \
#         --dot_recurrence self partner \
#         --dot_recurrence_in ref partner_ref \
#         --dot_recurrence_dim 64 \
#         --next_mention_prediction_type=multi_reference 

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
#         ${base_name}_split_mention-selection-ref-partner_nmpt-multi_loss-expanded_ss-ms-10 \
#         $base_args \
#         --selection_start_epoch 10 \
#         --next_mention_start_epoch 10 \
#         --dot_recurrence self partner \
#         --dot_recurrence_in next_mention selection ref partner_ref \
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

## Mention attention

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_mattn_mention-selection-ref-partner_nmpt-multi_loss-expanded \
#         $base_args \
#         --dot_recurrence_mention_attention \
#         --dot_recurrence self partner \
#         --dot_recurrence_in next_mention selection ref partner_ref \
#         --dot_recurrence_dim 64 \
#         --next_mention_prediction_type=multi_reference 

## hce-encoder-diffs
# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_split_mention-selection-ref-partner_nmpt-multi_loss-expanded_rerun \
#         $base_args \
#         --dot_recurrence self partner \
#         --dot_recurrence_in next_mention selection ref partner_ref \
#         --dot_recurrence_dim 64 \
#         --dot_recurrence_split \
#         --next_mention_prediction_type=multi_reference 

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_split_mention-selection-ref-partner_nmpt-multi_loss-expanded_rerun_hce-dr \
#         $base_args \
#         --dot_recurrence self partner \
#         --dot_recurrence_in next_mention selection ref partner_ref \
#         --dot_recurrence_dim 64 \
#         --dot_recurrence_split \
#         --next_mention_prediction_type=multi_reference  \
#         --hidden_context_mention_encoder_dot_recurrence

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_split_mention-selection-ref-partner_nmpt-multi_loss-expanded_rerun_hce-dr_hce-cf \
#         $base_args \
#         --dot_recurrence self partner \
#         --dot_recurrence_in next_mention selection ref partner_ref \
#         --dot_recurrence_dim 64 \
#         --dot_recurrence_split \
#         --next_mention_prediction_type=multi_reference  \
#         --hidden_context_mention_encoder_dot_recurrence \
#         --hidden_context_mention_encoder_count_features

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_split_mention-selection-ref-partner_nmpt-multi_loss-expanded_rerun_hce-cf \
#         $base_args \
#         --dot_recurrence self partner \
#         --dot_recurrence_in next_mention selection ref partner_ref \
#         --dot_recurrence_dim 64 \
#         --dot_recurrence_split \
#         --next_mention_prediction_type=multi_reference  \
#         --hidden_context_mention_encoder_count_features

## experiment 1 to check if we could plan using a next-mentioned model that doesn't use language; doesn't improve much over baseline
# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_mention-selection-ref-partner_nmpt-multi_loss-expanded_npm-no-lang \
#         $base_args \
#         --dot_recurrence self partner \
#         --dot_recurrence_in next_mention selection ref partner_ref \
#         --dot_recurrence_dim 64 \
#         --next_mention_prediction_type=multi_reference  \
#         --next_mention_prediction_no_lang

## experiment 2 to check if we could plan using a next-mentioned model that doesn't use language; doesn't improve at all over baseline
# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_mention-selection-ref-partner_nmpt-multi_loss-expanded_npm-no-lang_dri-wawm \
#         $base_args \
#         --dot_recurrence self partner \
#         --dot_recurrence_in next_mention selection ref partner_ref \
#         --dot_recurrence_dim 64 \
#         --next_mention_prediction_type=multi_reference  \
#         --next_mention_prediction_no_lang \
#         --dot_recurrence_inputs weights_average weights_max

## experiment 3 to check if we could plan using a selection model that doesn't use language
# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_mention-selection-ref-partner_nmpt-multi_loss-expanded_sel-no-lang_dri-wawm \
#         $base_args \
#         --dot_recurrence self partner \
#         --dot_recurrence_in next_mention selection ref partner_ref \
#         --dot_recurrence_dim 64 \
#         --next_mention_prediction_type=multi_reference \
#         --selection_prediction_no_lang \
#         --dot_recurrence_inputs weights_average weights_max

## experiment 3 had suspiciously high selection scores; check to make sure we don't do well when we're not using language and beliefs
# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_mention-ref-partner_nmpt-multi_loss-expanded_sel-no-lang_dri-wawm \
#         $base_args \
#         --dot_recurrence self partner \
#         --dot_recurrence_in next_mention ref partner_ref \
#         --dot_recurrence_dim 64 \
#         --next_mention_prediction_type=multi_reference \
#         --selection_prediction_no_lang \
#         --dot_recurrence_inputs weights_average weights_max

## version of experiment 3 that uses oracle refs for self
# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_oracle-self_mention-selection-ref-partner_nmpt-multi_loss-expanded_sel-no-lang_dri-wawm \
#         $base_args \
#         --dot_recurrence self partner \
#         --dot_recurrence_in next_mention selection ref partner_ref \
#         --dot_recurrence_oracle \
#         --dot_recurrence_oracle_for self \
#         --dot_recurrence_dim 64 \
#         --next_mention_prediction_type=multi_reference \
#         --selection_prediction_no_lang \
#         --dot_recurrence_inputs weights_average weights_max

## version of experiment 3 that uses an annealed temperature
# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_anneal_mention-selection-ref-partner_nmpt-multi_loss-expanded_sel-no-lang_dri-wawm \
#         $base_args \
#         --dot_recurrence self partner \
#         --dot_recurrence_in next_mention selection ref partner_ref \
#         --anneal_dot_recurrence \
#         --dot_recurrence_dim 64 \
#         --next_mention_prediction_type=multi_reference \
#         --selection_prediction_no_lang \
#         --dot_recurrence_inputs weights_average weights_max

# use predictions
# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_mention-selection-ref-partner_nmpt-multi_loss-expanded_sel-no-lang_dri-papm \
#         $base_args \
#         --dot_recurrence self partner \
#         --dot_recurrence_in next_mention selection ref partner_ref \
#         --dot_recurrence_dim 64 \
#         --next_mention_prediction_type=multi_reference \
#         --selection_prediction_no_lang \
#         --dot_recurrence_inputs predicted_average predicted_max

## version of experiment 3 that uses an annealed temperature, and predictions
# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_anneal_mention-selection-ref-partner_nmpt-multi_loss-expanded_sel-no-lang_dri-wawmpapm \
#         $base_args \
#         --dot_recurrence self partner \
#         --dot_recurrence_in next_mention selection ref partner_ref \
#         --anneal_dot_recurrence \
#         --dot_recurrence_dim 64 \
#         --next_mention_prediction_type=multi_reference \
#         --selection_prediction_no_lang \
#         --dot_recurrence_inputs weights_average weights_max predicted_average predicted_max

## predicted_hidden
# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_mention-selection-ref-partner_nmpt-multi_loss-expanded_sel-no-lang_dri-ph \
#         $base_args \
#         --dot_recurrence self partner \
#         --dot_recurrence_in next_mention selection ref partner_ref \
#         --dot_recurrence_dim 64 \
#         --next_mention_prediction_type=multi_reference \
#         --selection_prediction_no_lang \
#         --dot_recurrence_inputs predicted_hidden

# a good model?
# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_anneal_mention-selection-ref-partner_nmpt-multi_loss-expanded_sel-no-lang_dri-wawmpapm_sacf-count_hcme-filtered-separate_hcme-count_ep-40 \
#         $base_args \
#         --dot_recurrence self partner \
#         --dot_recurrence_in next_mention selection ref partner_ref \
#         --anneal_dot_recurrence \
#         --dot_recurrence_dim 64 \
#         --next_mention_prediction_type=multi_reference \
#         --selection_prediction_no_lang \
#         --dot_recurrence_inputs weights_average weights_max predicted_average predicted_max \
#         --structured_attention_configuration_features count  \
#         --hidden_context_mention_encoder_type=filtered-separate \
#         --hidden_context_mention_encoder_count_features \
#         --max_epoch 40

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_anneal_mention-selection-ref-partner_nmpt-multi_loss-expanded_sel-no-lang_dri-wawm_sacf-count_hcme-filtered-separate_hcme-count \
#         $base_args \
#         --dot_recurrence self partner \
#         --dot_recurrence_in next_mention selection ref partner_ref \
#         --anneal_dot_recurrence \
#         --dot_recurrence_dim 64 \
#         --next_mention_prediction_type=multi_reference \
#         --selection_prediction_no_lang \
#         --dot_recurrence_inputs weights_average weights_max \
#         --structured_attention_configuration_features count  \
#         --hidden_context_mention_encoder_type=filtered-separate \
#         --hidden_context_mention_encoder_count_features

## see if using predicted in the hidden state helps
# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_anneal_mention-selection-ref-partner_nmpt-multi_loss-expanded_dri-wawmwhpapmph \
#         $base_args \
#         --dot_recurrence self partner \
#         --dot_recurrence_in next_mention selection ref partner_ref \
#         --anneal_dot_recurrence \
#         --dot_recurrence_dim 64 \
#         --next_mention_prediction_type=multi_reference \
#         --dot_recurrence_inputs weights_average weights_max weighted_hidden predicted_average predicted_max predicted_hidden

## take out average and max, since there was an earlier experiment where those hurt when using weighted_hidden
# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_anneal_mention-selection-ref-partner_nmpt-multi_loss-expanded_dri-whph \
#         $base_args \
#         --dot_recurrence self partner \
#         --dot_recurrence_in next_mention selection ref partner_ref \
#         --anneal_dot_recurrence \
#         --dot_recurrence_dim 64 \
#         --next_mention_prediction_type=multi_reference \
#         --dot_recurrence_inputs weighted_hidden predicted_hidden

##
# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_anneal_mention-selection-ref-partner_nmpt-multi_loss-expanded_dri-wawmwhpapmph_sacf-count_hcme-filtered-separate_hcme-count \
#         $base_args \
#         --dot_recurrence self partner \
#         --dot_recurrence_in next_mention selection ref partner_ref \
#         --anneal_dot_recurrence \
#         --dot_recurrence_dim 64 \
#         --next_mention_prediction_type=multi_reference \
#         --dot_recurrence_inputs weights_average weights_max weighted_hidden predicted_average predicted_max predicted_hidden \
#         --structured_attention_configuration_features count  \
#         --hidden_context_mention_encoder_type=filtered-separate \
#         --hidden_context_mention_encoder_count_features
