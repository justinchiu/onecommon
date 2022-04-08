#!/bin/bash
script=""
#script="bash"

overall_name="jc-partner"

out_dir="expts/rel3_tsel_ref_dial_model_separate/${overall_name}"
mkdir -p $out_dir 2>/dev/null

base_args="--model_type hierarchical_rnn_reference_model \
  --max_epoch 12 \
  --lang_only_self \
  --nhid_lang 512 \
  --bsz 8 \
  --learned_pooling \
  --untie_grus \
  --bidirectional_reader \
  --detach_beliefs \
  --reduce_plateau \
  --hid2output 1-hidden-layer  \
  --attention_type sigmoid \
  --encode_relative_to_extremes \
  "

base_better_lang_args="--model_type hierarchical_rnn_reference_model \
  --max_epoch 12 \
  --lang_only_self \
  --nhid_lang 512 \
  --bsz 8 \
  --learned_pooling \
  --untie_grus \
  --bidirectional_reader \
  --detach_beliefs \
  --reduce_plateau \
  --encode_relative_to_extremes \
  "

unshared_args="--model_type hierarchical_rnn_reference_model \
  --max_epoch 12 \
  --lang_only_self \
  --nhid_lang 512 \
  --bsz 8 \
  --learned_pooling \
  --detach_beliefs \
  --reduce_plateau \
  --hid2output 1-hidden-layer  \
  --attention_type sigmoid \
  --encode_relative_to_extremes \
  "

ua_arch_args="--model_type hierarchical_rnn_reference_model \
  --max_epoch 12 \
  --bsz 8 \
  --reduce_plateau \
  "

ua_arch_attn_enc_args="--model_type hierarchical_rnn_reference_model \
  --max_epoch 12 \
  --bsz 8 \
  --reduce_plateau \
  --ctx_encoder_type attn_encoder \
  "

ua_arch_attn_enc_same_opt_args="--model_type hierarchical_rnn_reference_model \
  --max_epoch 30 \
  --bsz 16 \
  --ctx_encoder_type attn_encoder \
  "

unshared_part_lang_args="--model_type hierarchical_rnn_reference_model \
  --max_epoch 12 \
  --nhid_lang 512 \
  --bsz 8 \
  --learned_pooling \
  --detach_beliefs \
  --reduce_plateau \
  --hid2output 1-hidden-layer  \
  --attention_type sigmoid \
  --encode_relative_to_extremes \
  "
unshared_part_lang_small_args="--model_type hierarchical_rnn_reference_model \
  --max_epoch 12 \
  --bsz 8 \
  --learned_pooling \
  --detach_beliefs \
  --reduce_plateau \
  --hid2output 1-hidden-layer  \
  --attention_type sigmoid \
  --encode_relative_to_extremes \
  "

hierarchical_args="--mark_dots_mentioned \
  --word_attention_constrained \
  --partner_reference_prediction \
  --next_mention_prediction \
  --next_mention_prediction_type=multi_reference \
  --tie_reference_attn \
  --hidden_context \
  --hidden_context_mention_encoder \
  --hidden_context_confirmations \
  --hidden_context_mention_encoder_bidirectional \
  --hidden_context_mention_encoder_attention \
  --hidden_context_mention_encoder_type=filtered-separate \
  --hidden_context_mention_encoder_count_features \
  --hidden_context_mention_encoder_diffs \
  --confirmations_resolution_strategy=all \
  --hidden_context_confirmations_in generation next_mention \
  "

dot_recurrence_args="--dot_recurrence self partner \
  --dot_recurrence_in next_mention selection ref partner_ref next_mention_latents \
  --anneal_dot_recurrence \
  --dot_recurrence_dim 64 \
  --dot_recurrence_inputs weights_average weights_max predicted_average predicted_max \
  "

dot_recurrence_lang_args="--dot_recurrence self partner \
  --dot_recurrence_in next_mention selection ref partner_ref next_mention_latents \
  --anneal_dot_recurrence \
  --dot_recurrence_dim 64 \
  --dot_recurrence_inputs weights_average weights_max weighted_hidden predicted_average predicted_max predicted_hidden \
  "

rs_weight=0.1
augmentation_args="--relation_swap_augmentation \
  --relation_swapped_ref_and_partner_ref_weight=${rs_weight} \
  "

structured_attention_args="--structured_attention \
  --structured_attention_no_marginalize \
  --structured_temporal_attention  \
  --structured_attention_language_conditioned \
  --structured_attention_asymmetric_pairs \
  --structured_temporal_attention_transitions relational_asymm \
  --structured_attention_configuration_features count centroids \
  --structured_attention_configuration_transition_features centroid_diffs \
  --structured_attention_configuration_transition_max_size 3 \
  "

mbp_indicator_args="--next_partner_reference_prediction \
 --next_partner_reference_condition lang \
 --next_partner_reference_intersect_encoder indicator \
 --relation_include_intersect \
 --relation_include_intersect_both \
 --intersect_encoding_dim 1 \
 "
mbp_blind_args="--next_partner_reference_prediction \
 --next_partner_reference_condition lang \
 --next_partner_reference_blind \
 --intersect_encoding_dim 1 \
 "
mbp_mask_args="--next_partner_reference_prediction \
 --next_partner_reference_condition lang \
 --next_partner_reference_intersect_encoder mask \
 --intersect_encoding_dim 0 \
 "
mbp_deterministic_args="--next_partner_reference_prediction \
 --next_partner_reference_condition lang \
 --next_partner_reference_intersect_encoder deterministic \
 --intersect_encoding_dim 0 \
 "

mbp_indicator_confirm_args="--next_partner_reference_prediction \
 --next_partner_reference_condition lang \
 --next_partner_reference_intersect_encoder indicator \
 --relation_include_intersect \
 --relation_include_intersect_both \
 --intersect_encoding_dim 1 \
 --next_partner_confirm_prediction \
 --next_partner_confirm_agg sum \
 "

mbp_blind_confirm_args="--next_partner_reference_prediction \
 --next_partner_reference_condition lang \
 --next_partner_reference_blind \
 --intersect_encoding_dim 1 \
 --next_partner_confirm_prediction \
 "

mbp_indicator_confirm_mean_args="--next_partner_reference_prediction \
 --next_partner_reference_condition lang \
 --next_partner_reference_intersect_encoder indicator \
 --relation_include_intersect \
 --relation_include_intersect_both \
 --intersect_encoding_dim 1 \
 --next_partner_confirm_prediction \
 --next_partner_confirm_agg mean \
 "

mbp_indicator_confirm_attn_args="--next_partner_reference_prediction \
 --next_partner_reference_condition lang \
 --next_partner_reference_intersect_encoder indicator \
 --relation_include_intersect \
 --relation_include_intersect_both \
 --intersect_encoding_dim 1 \
 --next_partner_confirm_prediction \
 --next_partner_confirm_agg attn \
 "


# baseline model from fried without any next partner reference
function baseline () {
for fold in $@
do
  this_name=baseline
  mkdir -p ${out_dir}/${this_name} 2>/dev/null
  ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
    ${overall_name}/${this_name}/$fold \
    $base_args \
    $hierarchical_args \
    $structured_attention_args \
    $dot_recurrence_args \
    --wandb \
    --fold_nums $fold \
    --train_response_model binary_dots
done
}

# train next partner reference without any true state info
function blind () {
for fold in $@
do
  this_name=blind
  mkdir -p ${out_dir}/${this_name} 2>/dev/null
  ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
    ${overall_name}/${this_name}/$fold \
    $base_args \
    $hierarchical_args \
    $structured_attention_args \
    $mbp_blind_args \
    $dot_recurrence_args \
    --wandb \
    --fold_nums $fold \
    --train_response_model binary_dots
done
}

# train next partner reference by masking dot context
function mask () {
for fold in $@
do
  this_name=mask
  mkdir -p ${out_dir}/${this_name} 2>/dev/null
  ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
    ${overall_name}/${this_name}/$fold \
    $base_args \
    $hierarchical_args \
    $structured_attention_args \
    $mbp_mask_args \
    $dot_recurrence_args \
    --wandb \
    --fold_nums $fold \
    --train_response_model binary_dots
done
}

# train next partner reference by adding dot indicator features
function indicator () {
for fold in $@
do
  this_name=indicator
  mkdir -p ${out_dir}/${this_name} 2>/dev/null
  #${script} ./train_rel3_tsel_ref_dial_model_separate_nocuda.sh \
  ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
    ${overall_name}/${this_name}/$fold \
    $base_args \
    $hierarchical_args \
    $structured_attention_args \
    $mbp_indicator_args \
    $dot_recurrence_args \
    --wandb \
    --fold_nums $fold \
    --train_response_model binary_dots
done
}

# use deterministic mention prediction based on true state
function deterministic () {
for fold in $@
do
  this_name=deterministic
  mkdir -p ${out_dir}/${this_name} 2>/dev/null
  #${script} ./train_rel3_tsel_ref_dial_model_separate_nocuda.sh \
  ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
    ${overall_name}/${this_name}/$fold \
    $base_args \
    $hierarchical_args \
    $structured_attention_args \
    $mbp_deterministic_args \
    $dot_recurrence_args \
    --wandb \
    --fold_nums $fold \
    --train_response_model binary_dots
done
}

# train next partner reference by adding dot indicator features
# additionally predict (dis-)confirmation
function indicator-confirm () {
for fold in $@
do
  this_name=indicator-confirm
  mkdir -p ${out_dir}/${this_name} 2>/dev/null
  #${script} ./train_rel3_tsel_ref_dial_model_separate_nocuda.sh \
  ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
    ${overall_name}/${this_name}/$fold \
    $base_args \
    $hierarchical_args \
    $structured_attention_args \
    $mbp_indicator_confirm_args \
    $dot_recurrence_args \
    --wandb \
    --fold_nums $fold \
    --train_response_model binary_dots
done
}

# train next partner reference by adding dot indicator features
# additionally predict (dis-)confirmation but BLIND to state
function blind-confirm () {
for fold in $@
do
  this_name=blind-confirm
  mkdir -p ${out_dir}/${this_name} 2>/dev/null
  #${script} ./train_rel3_tsel_ref_dial_model_separate_nocuda.sh \
  ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
    ${overall_name}/${this_name}/$fold \
    $base_args \
    $hierarchical_args \
    $structured_attention_args \
    $mbp_blind_confirm_args \
    $dot_recurrence_args \
    --wandb \
    --fold_nums $fold \
    --train_response_model binary_dots
done
}

# train next partner reference by adding dot indicator features
# additionally predict (dis-)confirmation using attn
function indicator-confirm-attn () {
for fold in $@
do
  this_name=indicator-confirm-attn
  mkdir -p ${out_dir}/${this_name} 2>/dev/null
  #${script} ./train_rel3_tsel_ref_dial_model_separate_nocuda.sh \
  ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
    ${overall_name}/${this_name}/$fold \
    $base_args \
    $hierarchical_args \
    $structured_attention_args \
    $mbp_indicator_confirm_attn_args \
    $dot_recurrence_args \
    --wandb \
    --fold_nums $fold \
    --train_response_model binary_dots
done
}

# train next partner reference by adding dot indicator features
# additionally predict (dis-)confirmation using mean pooling
function indicator-confirm-mean () {
for fold in $@
do
  this_name=indicator-confirm-mean
  mkdir -p ${out_dir}/${this_name} 2>/dev/null
  #${script} ./train_rel3_tsel_ref_dial_model_separate_nocuda.sh \
  ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
    ${overall_name}/${this_name}/$fold \
    $base_args \
    $hierarchical_args \
    $structured_attention_args \
    $mbp_indicator_confirm_mean_args \
    $dot_recurrence_args \
    --wandb \
    --fold_nums $fold \
    --train_response_model binary_dots
done
}

