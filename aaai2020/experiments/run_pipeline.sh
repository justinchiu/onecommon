#!/bin/bash
script=""

mkdir expts/rel3_tsel_ref_dial_model_separate/pipeline 2>/dev/null

base_name="pipeline/base"
base_args="--model_type hierarchical_rnn_reference_model \
        --max_epoch 30 \
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
        --reduce_plateau "

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_rb-lpm-lm-lPm-lPpm_prb-lpm-lm-lPm-lPpm_mb-tpmpred-lmpred \
#         $base_args \
#         --max_epoch 40 \
#         --ref_beliefs last_mentioned last_partner_mentioned last_primary_mentioned last_partner_primary_mentioned \
#         --partner_ref_beliefs last_mentioned last_partner_mentioned last_primary_mentioned last_partner_primary_mentioned \
#         --mention_beliefs this_partner_mentioned_predicted last_mentioned_predicted \
#         --next_mention_start_epoch 10

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_512_extremes_learned-pooling_untied_bidirectional_hidden-mention-encoder_mb-tpmpred-lmpred \
#         $base_args \
#         --max_epoch 40 \
#         --mention_beliefs this_partner_mentioned_predicted last_mentioned_predicted \

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_rb-allpred-lpm-lm-lPm-lPpm_prb-lpm-lm-lPm-lPpm_mb-tpmpred-lmpred \
#         $base_args \
#         --max_epoch 40 \
#         --ref_beliefs last_mentioned_predicted last_partner_mentioned_predicted last_primary_mentioned_predicted last_partner_primary_mentioned_predicted \
#         --partner_ref_beliefs last_mentioned_predicted last_partner_mentioned_predicted last_primary_mentioned_predicted last_partner_primary_mentioned_predicted \
#         --mention_beliefs this_partner_mentioned_predicted last_mentioned_predicted \
#         --next_mention_start_epoch 10

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_rb-warmuppred-lpm-lm-lPm-lPpm_prb-lpm-lm-lPm-lPpm_mb-tpmpred-lmpred \
#         $base_args \
#         --max_epoch 40 \
#         --ref_beliefs last_mentioned_predicted last_partner_mentioned_predicted last_primary_mentioned_predicted last_partner_primary_mentioned_predicted \
#         --ref_beliefs_warmup last_mentioned last_partner_mentioned last_primary_mentioned last_partner_primary_mentioned \
#         --ref_beliefs_warmup_epochs 10 \
#         --partner_ref_beliefs last_mentioned_predicted last_partner_mentioned_predicted last_primary_mentioned_predicted last_partner_primary_mentioned_predicted \
#         --partner_ref_beliefs_warmup last_mentioned last_partner_mentioned last_primary_mentioned last_partner_primary_mentioned \
#         --partner_ref_beliefs_warmup_epochs 10 \
#         --mention_beliefs this_partner_mentioned_predicted last_mentioned_predicted \
#         --next_mention_start_epoch 10

### with selection beliefs
# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_rb-allpred-lpm-lm-lPm-lPpm_prb-lpm-lm-lPm-lPpm_mb-tpmpred-lmpred_sb-tpmpred-lmpred \
#         $base_args \
#         --max_epoch 40 \
#         --ref_beliefs last_mentioned_predicted last_partner_mentioned_predicted last_primary_mentioned_predicted last_partner_primary_mentioned_predicted \
#         --partner_ref_beliefs last_mentioned_predicted last_partner_mentioned_predicted last_primary_mentioned_predicted last_partner_primary_mentioned_predicted \
#         --mention_beliefs this_partner_mentioned_predicted last_mentioned_predicted \
#         --selection_beliefs this_partner_mentioned_predicted last_mentioned_predicted \
#         --selection_start_epoch 10 \
#         --next_mention_start_epoch 10

## predicted, without either mention
# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_rb-warmuppred-lpm-lm-lPm-lPpm_prb-lpm-lm-lPm-lPpm_mb-tpmpred-lmpred_sb-tpmpred-lmpred \
#         $base_args \
#         --max_epoch 40 \
#         --ref_beliefs last_mentioned_predicted last_partner_mentioned_predicted last_primary_mentioned_predicted last_partner_primary_mentioned_predicted \
#         --ref_beliefs_warmup last_mentioned last_partner_mentioned last_primary_mentioned last_partner_primary_mentioned \
#         --ref_beliefs_warmup_epochs 10 \
#         --partner_ref_beliefs last_mentioned_predicted last_partner_mentioned_predicted last_primary_mentioned_predicted last_partner_primary_mentioned_predicted \
#         --partner_ref_beliefs_warmup last_mentioned last_partner_mentioned last_primary_mentioned last_partner_primary_mentioned \
#         --partner_ref_beliefs_warmup_epochs 10 \
#         --mention_beliefs this_partner_mentioned_predicted last_mentioned_predicted \
#         --selection_beliefs this_partner_mentioned_predicted last_mentioned_predicted \
#         --selection_start_epoch 10 \
#         --next_mention_start_epoch 10

## predicted, with either mention
# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_rb-warmuppred-lpm-lm-lPm-lPpm_prb-lpm-lm-lPm-lPpm_mb-tpmpred-lmpred-tempred_sb-tpmpred-lmpred-tempred_refac \
#         $base_args \
#         --max_epoch 20 \
#         --ref_beliefs last_mentioned_predicted last_partner_mentioned_predicted last_primary_mentioned_predicted last_partner_primary_mentioned_predicted \
#         --ref_beliefs_warmup last_mentioned last_partner_mentioned last_primary_mentioned last_partner_primary_mentioned \
#         --ref_beliefs_warmup_epochs 10 \
#         --partner_ref_beliefs last_mentioned_predicted last_partner_mentioned_predicted last_primary_mentioned_predicted last_partner_primary_mentioned_predicted \
#         --partner_ref_beliefs_warmup last_mentioned last_partner_mentioned last_primary_mentioned last_partner_primary_mentioned \
#         --partner_ref_beliefs_warmup_epochs 10 \
#         --mention_beliefs this_partner_mentioned_predicted last_mentioned_predicted this_either_mentioned_predicted \
#         --selection_beliefs this_partner_mentioned_predicted last_mentioned_predicted this_either_mentioned_predicted \
#         --selection_start_epoch 10 \
#         --next_mention_start_epoch 10

## predicted, with either mention, and multi-ref
# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_rb-warmuppred-lpm-lm-lPm-lPpm_prb-lpm-lm-lPm-lPpm_mb-tpmpred-lmpred-tempred_sb-tpmpred-lmpred-tempred_nmpt-multi_loss-expanded \
#         $base_args \
#         --max_epoch 20 \
#         --ref_beliefs last_mentioned_predicted last_partner_mentioned_predicted last_primary_mentioned_predicted last_partner_primary_mentioned_predicted \
#         --ref_beliefs_warmup last_mentioned last_partner_mentioned last_primary_mentioned last_partner_primary_mentioned \
#         --ref_beliefs_warmup_epochs 10 \
#         --partner_ref_beliefs last_mentioned_predicted last_partner_mentioned_predicted last_primary_mentioned_predicted last_partner_primary_mentioned_predicted \
#         --partner_ref_beliefs_warmup last_mentioned last_partner_mentioned last_primary_mentioned last_partner_primary_mentioned \
#         --partner_ref_beliefs_warmup_epochs 10 \
#         --mention_beliefs this_partner_mentioned_predicted last_mentioned_predicted this_either_mentioned_predicted \
#         --selection_beliefs this_partner_mentioned_predicted last_mentioned_predicted this_either_mentioned_predicted \
#         --selection_start_epoch 10 \
#         --next_mention_start_epoch 10 \
#         --next_mention_prediction_type=multi_reference 

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_rb-warmuppred-lpm-lm-lPm-lPpm_prb-lpm-lm-lPm-lPpm_mb-tpm-lm_sb-tpm-lm_refac \
#         $base_args \
#         --max_epoch 20 \
#         --ref_beliefs last_mentioned_predicted last_partner_mentioned_predicted last_primary_mentioned_predicted last_partner_primary_mentioned_predicted \
#         --ref_beliefs_warmup last_mentioned last_partner_mentioned last_primary_mentioned last_partner_primary_mentioned \
#         --ref_beliefs_warmup_epochs 10 \
#         --partner_ref_beliefs last_mentioned_predicted last_partner_mentioned_predicted last_primary_mentioned_predicted last_partner_primary_mentioned_predicted \
#         --partner_ref_beliefs_warmup last_mentioned last_partner_mentioned last_primary_mentioned last_partner_primary_mentioned \
#         --partner_ref_beliefs_warmup_epochs 10 \
#         --mention_beliefs this_partner_mentioned last_mentioned \
#         --selection_beliefs this_partner_mentioned last_mentioned \
#         --selection_start_epoch 10 \
#         --next_mention_start_epoch 10

## true, without either mention
# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_rb-warmuppred-lpm-lm-lPm-lPpm_prb-lpm-lm-lPm-lPpm_mb-tpm-lm-tem_sb-tpm-lm-tem_refac \
#         $base_args \
#         --max_epoch 20 \
#         --ref_beliefs last_mentioned_predicted last_partner_mentioned_predicted last_primary_mentioned_predicted last_partner_primary_mentioned_predicted \
#         --ref_beliefs_warmup last_mentioned last_partner_mentioned last_primary_mentioned last_partner_primary_mentioned \
#         --ref_beliefs_warmup_epochs 10 \
#         --partner_ref_beliefs last_mentioned_predicted last_partner_mentioned_predicted last_primary_mentioned_predicted last_partner_primary_mentioned_predicted \
#         --partner_ref_beliefs_warmup last_mentioned last_partner_mentioned last_primary_mentioned last_partner_primary_mentioned \
#         --partner_ref_beliefs_warmup_epochs 10 \
#         --mention_beliefs this_partner_mentioned last_mentioned this_either_mentioned \
#         --selection_beliefs this_partner_mentioned last_mentioned this_either_mentioned \
#         --selection_start_epoch 10 \
#         --next_mention_start_epoch 10

## true, without either mention, with multi-ref
${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
        ${base_name}_rb-warmuppred-lpm-lm-lPm-lPpm_prb-lpm-lm-lPm-lPpm_mb-tpm-lm-tem_sb-tpm-lm-tem_nmpt-multi_loss-expanded \
        $base_args \
        --max_epoch 20 \
        --ref_beliefs last_mentioned_predicted last_partner_mentioned_predicted last_primary_mentioned_predicted last_partner_primary_mentioned_predicted \
        --ref_beliefs_warmup last_mentioned last_partner_mentioned last_primary_mentioned last_partner_primary_mentioned \
        --ref_beliefs_warmup_epochs 10 \
        --partner_ref_beliefs last_mentioned_predicted last_partner_mentioned_predicted last_primary_mentioned_predicted last_partner_primary_mentioned_predicted \
        --partner_ref_beliefs_warmup last_mentioned last_partner_mentioned last_primary_mentioned last_partner_primary_mentioned \
        --partner_ref_beliefs_warmup_epochs 10 \
        --mention_beliefs this_partner_mentioned last_mentioned this_either_mentioned \
        --selection_beliefs this_partner_mentioned last_mentioned this_either_mentioned \
        --selection_start_epoch 10 \
        --next_mention_start_epoch 10 \
        --next_mention_prediction_type=multi_reference 

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_rb-warmuppred-lpm-lm-lPm-lPpm_prb-lpm-lm-lPm-lPpm \
#         $base_args \
#         --max_epoch 40 \
#         --ref_beliefs last_mentioned_predicted last_partner_mentioned_predicted last_primary_mentioned_predicted last_partner_primary_mentioned_predicted \
#         --ref_beliefs_warmup last_mentioned last_partner_mentioned last_primary_mentioned last_partner_primary_mentioned \
#         --ref_beliefs_warmup_epochs 10 \
#         --partner_ref_beliefs last_mentioned_predicted last_partner_mentioned_predicted last_primary_mentioned_predicted last_partner_primary_mentioned_predicted \
#         --partner_ref_beliefs_warmup last_mentioned last_partner_mentioned last_primary_mentioned last_partner_primary_mentioned \
#         --partner_ref_beliefs_warmup_epochs 10 \
#         --selection_start_epoch 10 \
#         --next_mention_start_epoch 10

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_rb-none_prb-none \
#         $base_args \
#         --max_epoch 20 \
#         --selection_start_epoch 10 \
#         --next_mention_start_epoch 10

### with selection beliefs and primaries
# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_rb-allpred-lpm-lm-lPm-lPpm_prb-lpm-lm-lPm-lPpm_mb-tpmpred-lmpred-tPpmpred-lPmpred_sb-tpmpred-lmpred-tPpmpred-lPmpred \
#         $base_args \
#         --max_epoch 20 \
#         --ref_beliefs last_mentioned_predicted last_partner_mentioned_predicted last_primary_mentioned_predicted last_partner_primary_mentioned_predicted \
#         --partner_ref_beliefs last_mentioned_predicted last_partner_mentioned_predicted last_primary_mentioned_predicted last_partner_primary_mentioned_predicted \
#         --mention_beliefs this_partner_mentioned_predicted last_mentioned_predicted this_partner_primary_mentioned_predicted last_primary_mentioned_predicted \
#         --selection_beliefs this_partner_mentioned_predicted last_mentioned_predicted this_partner_primary_mentioned_predicted last_primary_mentioned_predicted \
#         --selection_start_epoch 10 \
#         --next_mention_start_epoch 10

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_rb-warmuppred-lpm-lm-lPm-lPpm_prb-lpm-lm-lPm-lPpm_mb-tpmpred-lmpred-tPpmpred-lPmpred_sb-tpmpred-lmpred-tPpmpred-lPmpred \
#         $base_args \
#         --max_epoch 20 \
#         --ref_beliefs last_mentioned_predicted last_partner_mentioned_predicted last_primary_mentioned_predicted last_partner_primary_mentioned_predicted \
#         --ref_beliefs_warmup last_mentioned last_partner_mentioned last_primary_mentioned last_partner_primary_mentioned \
#         --ref_beliefs_warmup_epochs 10 \
#         --partner_ref_beliefs last_mentioned_predicted last_partner_mentioned_predicted last_primary_mentioned_predicted last_partner_primary_mentioned_predicted \
#         --partner_ref_beliefs_warmup last_mentioned last_partner_mentioned last_primary_mentioned last_partner_primary_mentioned \
#         --partner_ref_beliefs_warmup_epochs 10 \
#         --mention_beliefs this_partner_mentioned_predicted last_mentioned_predicted this_partner_primary_mentioned_predicted last_primary_mentioned_predicted \
#         --selection_beliefs this_partner_mentioned_predicted last_mentioned_predicted this_partner_primary_mentioned_predicted last_primary_mentioned_predicted \
#         --selection_start_epoch 10 \
#         --next_mention_start_epoch 10

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_rb-warmuppred-lpm-lm-lPm-lPpm_prb-lpm-lm-lPm-lPpm_mb-tpm-lm-tPpm-lPm_sb-tpm-lm-tPpm-lPm \
#         $base_args \
#         --max_epoch 20 \
#         --ref_beliefs last_mentioned_predicted last_partner_mentioned_predicted last_primary_mentioned_predicted last_partner_primary_mentioned_predicted \
#         --ref_beliefs_warmup last_mentioned last_partner_mentioned last_primary_mentioned last_partner_primary_mentioned \
#         --ref_beliefs_warmup_epochs 10 \
#         --partner_ref_beliefs last_mentioned_predicted last_partner_mentioned_predicted last_primary_mentioned_predicted last_partner_primary_mentioned_predicted \
#         --partner_ref_beliefs_warmup last_mentioned last_partner_mentioned last_primary_mentioned last_partner_primary_mentioned \
#         --partner_ref_beliefs_warmup_epochs 10 \
#         --mention_beliefs this_partner_mentioned last_mentioned this_partner_primary_mentioned last_primary_mentioned \
#         --selection_beliefs this_partner_mentioned last_mentioned this_partner_primary_mentioned last_primary_mentioned \
#         --selection_start_epoch 10 \
#         --next_mention_start_epoch 10

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_rb-warmuppred-lpm-lm-lPm-lPpm_prb-lpm-lm-lPm-lPpm \
#         $base_args \
#         --max_epoch 20 \
#         --ref_beliefs last_mentioned_predicted last_partner_mentioned_predicted last_primary_mentioned_predicted last_partner_primary_mentioned_predicted \
#         --ref_beliefs_warmup last_mentioned last_partner_mentioned last_primary_mentioned last_partner_primary_mentioned \
#         --ref_beliefs_warmup_epochs 10 \
#         --partner_ref_beliefs last_mentioned_predicted last_partner_mentioned_predicted last_primary_mentioned_predicted last_partner_primary_mentioned_predicted \
#         --partner_ref_beliefs_warmup last_mentioned last_partner_mentioned last_primary_mentioned last_partner_primary_mentioned \
#         --partner_ref_beliefs_warmup_epochs 10 \
#         --selection_start_epoch 10 \
#         --next_mention_start_epoch 10
