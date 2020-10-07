#!/bin/bash
script=""

mkdir expts/rel3_tsel_ref_dial_model_separate/directions 2>/dev/null

base_name="directions/base"
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
        --reduce_plateau "

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         directions/no_lang-cond \
#         --model_type hierarchical_rnn_reference_model \
#         --max_epoch 30 \
#         --lang_only_self \
#         --structured_attention \
#         --structured_attention_no_marginalize \
#         --structured_temporal_attention  \
#         --structured_temporal_attention_transitions relational  \
#         --mark_dots_mentioned \
#         --word_attention_constrained \
#         --hid2output 1-hidden-layer  \
#         --attention_type sigmoid \
#         --partner_reference_prediction \
#         --next_mention_prediction \
#         --reduce_plateau

## Line 26 in the "ref pragmatics" part of the sheet
# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         $base_name \
#         $base_args

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_learned-pooling \
#         $base_args \
#         --learned_pooling

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_learned-pooling_untied \
#         $base_args \
#         --learned_pooling \
#         --untie_grus

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_learned-pooling_untied_bidirectional \
#         $base_args \
#         --learned_pooling \
#         --untie_grus \
#         --bidirectional_reader

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_learned-pooling_untied_bidirectional_hidden-mention-encoder \
#         $base_args \
#         --learned_pooling \
#         --untie_grus \
#         --bidirectional_reader \
#         --hidden_context \
#         --hidden_context_mention_encoder

## 512-dim version of the above
## TODO: only 512 for the writer?
# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_512_learned-pooling_untied_bidirectional_hidden-mention-encoder \
#         $base_args \
#         --nhid_lang 512 \
#         --learned_pooling \
#         --untie_grus \
#         --bidirectional_reader \
#         --hidden_context \
#         --hidden_context_mention_encoder

# with reference beliefs
# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_512_learned-pooling_untied_bidirectional_hidden-mention-encoder_rb-lpm-lm_prb-lpm-lm \
#         $base_args \
#         --nhid_lang 512 \
#         --learned_pooling \
#         --untie_grus \
#         --bidirectional_reader \
#         --hidden_context \
#         --hidden_context_mention_encoder \
#         --ref_beliefs last_mentioned last_partner_mentioned \
#         --partner_ref_beliefs last_mentioned last_partner_mentioned \

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_512_learned-pooling_untied_bidirectional_hidden-mention-encoder_rb-lpm_prb-lm \
#         $base_args \
#         --nhid_lang 512 \
#         --learned_pooling \
#         --untie_grus \
#         --bidirectional_reader \
#         --hidden_context \
#         --hidden_context_mention_encoder \
#         --ref_beliefs last_partner_mentioned \
#         --partner_ref_beliefs last_mentioned \

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_512_learned-pooling_untied_bidirectional_hidden-mention-encoder_rb-lm_prb-lpm \
#         $base_args \
#         --nhid_lang 512 \
#         --learned_pooling \
#         --untie_grus \
#         --bidirectional_reader \
#         --hidden_context \
#         --hidden_context_mention_encoder \
#         --ref_beliefs last_mentioned \
#         --partner_ref_beliefs last_partner_mentioned \

## extremes encoder
# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_512_extremes_learned-pooling_untied_bidirectional_hidden-mention-encoder_3 \
#         $base_args \
#         --nhid_lang 512 \
#         --encode_relative_to_extremes \
#         --learned_pooling \
#         --untie_grus \
#         --bidirectional_reader \
#         --hidden_context \
#         --hidden_context_mention_encoder

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_512_extremes_learned-pooling_untied_bidirectional_hidden-mention-encoder-filtered-shared \
#         $base_args \
#         --nhid_lang 512 \
#         --encode_relative_to_extremes \
#         --learned_pooling \
#         --untie_grus \
#         --bidirectional_reader \
#         --hidden_context \
#         --hidden_context_mention_encoder \
#         --hidden_context_mention_encoder_type=filtered-shared

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_512_extremes_learned-pooling_untied_bidirectional_hidden-mention-encoder-filtered-separate \
#         $base_args \
#         --nhid_lang 512 \
#         --encode_relative_to_extremes \
#         --learned_pooling \
#         --untie_grus \
#         --bidirectional_reader \
#         --hidden_context \
#         --hidden_context_mention_encoder \
#         --hidden_context_mention_encoder_type=filtered-separate

${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
        ${base_name}_512_extremes_learned-pooling_untied_bidirectional_hidden-mention-encoder-filtered-separate-diffs \
        $base_args \
        --nhid_lang 512 \
        --encode_relative_to_extremes \
        --learned_pooling \
        --untie_grus \
        --bidirectional_reader \
        --hidden_context \
        --hidden_context_mention_encoder \
        --hidden_context_mention_encoder_diffs \
        --hidden_context_mention_encoder_type=filtered-separate

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_512_rel-2_extremes_learned-pooling_untied_bidirectional_hidden-mention-encoder-filtered-separate \
#         $base_args \
#         --nhid_lang 512 \
#         --encode_relative_to_extremes \
#         --learned_pooling \
#         --untie_grus \
#         --bidirectional_reader \
#         --hidden_context \
#         --hidden_context_mention_encoder \
#         --hidden_context_mention_encoder_type=filtered-separate \
#         --relation_encoder_layers=2

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_512_rel-2_extremes_learned-pooling_untied_bidirectional_hidden-mention-encoder-filtered-separate_unk-2 \
#         $base_args \
#         --nhid_lang 512 \
#         --encode_relative_to_extremes \
#         --learned_pooling \
#         --untie_grus \
#         --bidirectional_reader \
#         --hidden_context \
#         --hidden_context_mention_encoder \
#         --hidden_context_mention_encoder_type=filtered-separate \
#         --relation_encoder_layers=2 \
#         --unk_threshold=2

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_512_rel-2_extremes_learned-pooling_untied_bidirectional_hidden-mention-encoder-filtered-shared \
#         $base_args \
#         --nhid_lang 512 \
#         --encode_relative_to_extremes \
#         --learned_pooling \
#         --untie_grus \
#         --bidirectional_reader \
#         --hidden_context \
#         --hidden_context_mention_encoder \
#         --hidden_context_mention_encoder_type=filtered-shared \
#         --relation_encoder_layers=2

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_512_bsz-64_rel-2_extremes_learned-pooling_untied_bidirectional_hidden-mention-encoder-filtered-shared \
#         $base_args \
#         --bsz 64 \
#         --nhid_lang 512 \
#         --encode_relative_to_extremes \
#         --learned_pooling \
#         --untie_grus \
#         --bidirectional_reader \
#         --hidden_context \
#         --hidden_context_mention_encoder \
#         --hidden_context_mention_encoder_type=filtered-shared \
#         --relation_encoder_layers=2

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_512_bsz-64_lr-5e-3_rel-2_extremes_learned-pooling_untied_bidirectional_hidden-mention-encoder-filtered-shared \
#         $base_args \
#         --bsz 64 \
#         --lr 5e-3 \
#         --nhid_lang 512 \
#         --encode_relative_to_extremes \
#         --learned_pooling \
#         --untie_grus \
#         --bidirectional_reader \
#         --hidden_context \
#         --hidden_context_mention_encoder \
#         --hidden_context_mention_encoder_type=filtered-shared \
#         --relation_encoder_layers=2

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_512_extremes_learned-pooling_untied_bidirectional_hidden-mention-encoder_rb-lpm-lm_prb-lpm-lm \
#         $base_args \
#         --nhid_lang 512 \
#         --encode_relative_to_extremes \
#         --learned_pooling \
#         --untie_grus \
#         --bidirectional_reader \
#         --hidden_context \
#         --hidden_context_mention_encoder \
#         --ref_beliefs last_mentioned last_partner_mentioned \
#         --partner_ref_beliefs last_mentioned last_partner_mentioned \

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_512_extremes_asymmetric_learned-pooling_untied_bidirectional_hidden-mention-encoder_rb-lpm-lm_prb-lpm-lm \
#         $base_args \
#         --nhid_lang 512 \
#         --encode_relative_to_extremes \
#         --structured_attention_asymmetric_pairs \
#         --learned_pooling \
#         --untie_grus \
#         --bidirectional_reader \
#         --hidden_context \
#         --hidden_context_mention_encoder \
#         --ref_beliefs last_mentioned last_partner_mentioned \
#         --partner_ref_beliefs last_mentioned last_partner_mentioned \

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_512_extremes_asymmetric_learned-pooling_untied_bidirectional_hidden-mention-encoder_rb-lpm-lm-lPm-lPpm_prb-lpm-lm-lPm-lPpm \
#         $base_args \
#         --nhid_lang 512 \
#         --encode_relative_to_extremes \
#         --structured_attention_asymmetric_pairs \
#         --learned_pooling \
#         --untie_grus \
#         --bidirectional_reader \
#         --hidden_context \
#         --hidden_context_mention_encoder \
#         --ref_beliefs last_mentioned last_partner_mentioned last_primary_mentioned last_partner_primary_mentioned \
#         --partner_ref_beliefs last_mentioned last_partner_mentioned last_primary_mentioned last_partner_primary_mentioned \

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_512_extremes_learned-pooling_untied_bidirectional_hidden-mention-encoder_rb-lpm-lm-lPm-lPpm_prb-lpm-lm-lPm-lPpm \
#         $base_args \
#         --nhid_lang 512 \
#         --encode_relative_to_extremes \
#         --learned_pooling \
#         --untie_grus \
#         --bidirectional_reader \
#         --hidden_context \
#         --hidden_context_mention_encoder \
#         --ref_beliefs last_mentioned last_partner_mentioned last_primary_mentioned last_partner_primary_mentioned \
#         --partner_ref_beliefs last_mentioned last_partner_mentioned last_primary_mentioned last_partner_primary_mentioned \

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_512_extremes_learned-pooling_untied_bidirectional_hidden-mention-encoder_rb-lpm-lm-lPm-lPpm_prb-lpm-lm-lPm-lPpm_mb-tpmpred-lmpred \
#         $base_args \
#         --max_epoch 40 \
#         --nhid_lang 512 \
#         --encode_relative_to_extremes \
#         --learned_pooling \
#         --untie_grus \
#         --bidirectional_reader \
#         --hidden_context \
#         --hidden_context_mention_encoder \
#         --ref_beliefs last_mentioned last_partner_mentioned last_primary_mentioned last_partner_primary_mentioned \
#         --partner_ref_beliefs last_mentioned last_partner_mentioned last_primary_mentioned last_partner_primary_mentioned \
#         --mention_beliefs this_partner_mentioned_predicted last_mentioned_predicted \
#         --detach_beliefs \
#         --next_mention_start_epoch 10

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_512_extremes_learned-pooling_untied_bidirectional_hidden-mention-encoder_mb-tpmpred-lmpred \
#         $base_args \
#         --max_epoch 40 \
#         --nhid_lang 512 \
#         --encode_relative_to_extremes \
#         --learned_pooling \
#         --untie_grus \
#         --bidirectional_reader \
#         --hidden_context \
#         --hidden_context_mention_encoder \
#         --mention_beliefs this_partner_mentioned_predicted last_mentioned_predicted \
#         --detach_beliefs \
#         --next_mention_start_epoch 10

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_512_extremes_learned-pooling_untied_bidirectional_hidden-mention-encoder_mention-start-10 \
#         $base_args \
#         --max_epoch 40 \
#         --nhid_lang 512 \
#         --encode_relative_to_extremes \
#         --learned_pooling \
#         --untie_grus \
#         --bidirectional_reader \
#         --hidden_context \
#         --hidden_context_mention_encoder \
#         --detach_beliefs \
#         --next_mention_start_epoch 10
