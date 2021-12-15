#!/bin/bash
script=""
#script="bash"

overall_name="nov-15"

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


## **This was not used in the paper -- it introduces data augmentation**
# this_name=full
# mkdir ${out_dir}/${this_name} 2>/dev/null
# for fold in $@
# do
#   ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#     ${overall_name}/${this_name}/$fold \
#     $base_args \
#     $hierarchical_args \
#     $dot_recurrence_args \
#     $augmentation_args \
#     $structured_attention_args \
#     --fold_nums $fold
# done

## **F-Mem model in Table 1 of paper (-Mem means minus dot memory, which is called "recurrence" here)**
# this_name=plain-hierarchical-structured
# mkdir ${out_dir}/${this_name} 2>/dev/null
# for fold in $@
# do
#   # ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#   #   ${overall_name}/${this_name}/$fold \
#   #   $base_args \
#   #   $hierarchical_args \
#   #   $structured_attention_args \
#   #   --fold_nums $fold

#   for ct in 0 1
#   do
#     ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#       ${overall_name}/${this_name}/${fold}_ct-${ct} \
#       $base_args \
#       $hierarchical_args \
#       $structured_attention_args \
#       --fold_nums $fold \
#       --crosstalk_split $ct
#   done
# done

## **FULL model in Table 1 of paper**
this_name=plain-hierarchical-structured-recurrence
mkdir -p ${out_dir}/${this_name} 2>/dev/null
for fold in $@
do
  ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
    ${overall_name}/${this_name}/$fold \
    $base_args \
    $hierarchical_args \
    $structured_attention_args \
    $dot_recurrence_args \
    --fold_nums $fold

  # for ct in 0 1
  # do
  #   ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
  #     ${overall_name}/${this_name}/${fold}_ct-${ct} \
  #     $base_args \
  #     $hierarchical_args \
  #     $structured_attention_args \
  #     $dot_recurrence_args \
  #     --fold_nums $fold \
  #     --crosstalk_split $ct
  # done
done

# this_name=plain-hierarchical-structured-recurrence-lang
# mkdir ${out_dir}/${this_name} 2>/dev/null
# for fold in $@
# do
#   ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#     ${overall_name}/${this_name}/$fold \
#     $base_args \
#     $hierarchical_args \
#     $structured_attention_args \
#     $dot_recurrence_lang_args \
#     --fold_nums $fold

#   # for ct in 0 1
#   # do
#   #   ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#   #     ${overall_name}/${this_name}/${fold}_ct-${ct} \
#   #     $base_args \
#   #     $hierarchical_args \
#   #     $structured_attention_args \
#   #     $dot_recurrence_args \
#   #     --fold_nums $fold \
#   #     --crosstalk_split $ct
#   # done
# done

# this_name=plain
# mkdir ${out_dir}/${this_name} 2>/dev/null
# for fold in $@
# do
#   # ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#     #   ${overall_name}/${this_name}/$fold \
#     #   $base_args \
#     #   --fold_nums $fold

#   # for ct in 0 1
#   # do
#   #   ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#   #     ${overall_name}/${this_name}/${fold}_ct-${ct} \
#   #     $base_args \
#   #     --fold_nums $fold \
#   #     --crosstalk_split $ct
#   # done
# done

# this_name=plain-better-lang
# mkdir ${out_dir}/${this_name} 2>/dev/null
# for fold in $@
# do
#   ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#     ${overall_name}/${this_name}/$fold \
#     $base_better_lang_args \
#     --fold_nums $fold
# done

#** F-Mem-Struc in the paper
# this_name=plain-hierarchical
# mkdir ${out_dir}/${this_name} 2>/dev/null
# for fold in $@
# do
#   # ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#   #   ${overall_name}/${this_name}/$fold \
#   #   $base_args \
#   #   $hierarchical_args \
#   #   --fold_nums $fold

#   for ct in 0 1
#   do
#     ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#       ${overall_name}/${this_name}/${fold}_ct-${ct} \
#       $base_args \
#       $hierarchical_args \
#       --fold_nums $fold \
#       --crosstalk_split $ct
#   done
# done

# this_name=unshared
# mkdir ${out_dir}/${this_name} 2>/dev/null
# for fold in $@
# do
#   ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#     ${overall_name}/${this_name}/$fold \
#     $unshared_args \
#     --fold_nums $fold
# done

# this_name=ua_arch_args
# mkdir ${out_dir}/${this_name} 2>/dev/null
# for fold in $@
# do
#   ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#     ${overall_name}/${this_name}/$fold \
#     $ua_arch_args \
#     --fold_nums $fold
# done

# this_name=ua_arch_attn_enc
# mkdir ${out_dir}/${this_name} 2>/dev/null
# for fold in $@
# do
#   ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#     ${overall_name}/${this_name}/$fold \
#     $ua_arch_attn_enc_args \
#     --fold_nums $fold
# done

# this_name=ua_arch_attn_enc_shared_attn
# mkdir ${out_dir}/${this_name} 2>/dev/null
# for fold in $@
# do
#   ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#     ${overall_name}/${this_name}/$fold \
#     $ua_arch_attn_enc_args \
#     --fold_nums $fold

#   # for ct in 0 1
#   # do
#   #   ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#   #     ${overall_name}/${this_name}/${fold}_ct-${ct} \
#   #     $ua_arch_attn_enc_args \
#   #     --fold_nums $fold \
#   #     --crosstalk_split $ct
#   # done
# done

# this_name=unshared_part_lang
# mkdir ${out_dir}/${this_name} 2>/dev/null
# for fold in $@
# do
#   ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#     ${overall_name}/${this_name}/$fold \
#     $unshared_part_lang_args \
#     --fold_nums $fold
# done

# this_name=unshared_part_lang_small
# mkdir ${out_dir}/${this_name} 2>/dev/null
# for fold in $@
# do
#   ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#     ${overall_name}/${this_name}/$fold \
#     $unshared_part_lang_small_args \
#     --fold_nums $fold
# done

# this_name=ua_arch_attn_enc_shared_attn_same_opt
# mkdir ${out_dir}/${this_name} 2>/dev/null
# for fold in $@
# do
#   # ${script} ./train_rel3_tsel_ref_dial_model_separate-named-shared-actual.sh \
#   #   ${overall_name}/${this_name}/$fold \
#   #   $ua_arch_attn_enc_same_opt_args \
#   #   --fold_nums $fold

#   for ct in 0 1
#   do
#     ${script} ./train_rel3_tsel_ref_dial_model_separate-named-shared-actual.sh \
#       ${overall_name}/${this_name}/${fold}_ct-${ct} \
#       $ua_arch_attn_enc_same_opt_args \
#       --fold_nums $fold \
#       --crosstalk_split $ct
#   done
# done
