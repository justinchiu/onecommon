#!/bin/bash
script=""

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         hierarchical_only-self_mark-dots-mentioned_word-attention-constrained_mask-pad_h2o-1_sigmoid \
#         --model_type hierarchical_rnn_reference_model \
#         --max_epoch 30 \
#         --lang_only_self \
#         --mark_dots_mentioned \
#         --word_attention_constrained \
#         --hid2output 1-hidden-layer  \
#         --attention_type sigmoid

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         hierarchical_only-self_mark-dots-mentioned_word-attention-constrained_partner-ref_next-mention_mask-pad_h2o-1_sigmoid \
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
#         hierarchical_only-self_mark-dots-mentioned_miwp_word-attention-constrained_partner-ref_next-mention_mask-pad_h2o-1_sigmoid \
#         --model_type hierarchical_rnn_reference_model \
#         --max_epoch 30 \
#         --lang_only_self \
#         --mark_dots_mentioned \
#         --marks_in_word_prediction \
#         --word_attention_constrained \
#         --hid2output 1-hidden-layer  \
#         --attention_type sigmoid \
#         --partner_reference_prediction \
#         --next_mention_prediction

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         hierarchical_only-self_mark-dots-mentioned_feed-attention-constrained_partner-ref_next-mention_mask-pad_h2o-1_sigmoid \
#         --model_type hierarchical_rnn_reference_model \
#         --max_epoch 30 \
#         --lang_only_self \
#         --mark_dots_mentioned \
#         --marks_in_word_prediction \
#         --no_word_attention \
#         --feed_context \
#         --feed_context_attend \
#         --feed_attention_constrained \
#         --hid2output 1-hidden-layer  \
#         --attention_type sigmoid \
#         --partner_reference_prediction \
#         --next_mention_prediction

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         hierarchical_only-self_mark-dots-mentioned_miwp_word-attention-constrained_feed-attention-constrained_partner-ref_next-mention_mask-pad_h2o-1_sigmoid \
#         --model_type hierarchical_rnn_reference_model \
#         --max_epoch 30 \
#         --lang_only_self \
#         --mark_dots_mentioned \
#         --marks_in_word_prediction \
#         --word_attention_constrained \
#         --feed_context \
#         --feed_context_attend \
#         --feed_attention_constrained \
#         --hid2output 1-hidden-layer  \
#         --attention_type sigmoid \
#         --partner_reference_prediction \
#         --next_mention_prediction

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         hierarchical_only-self_nhid-lang-512_mark-dots-mentioned_word-attention-constrained_hidden-attention-constrained_mask-pad_h2o-1_sigmoid \
#         --model_type hierarchical_rnn_reference_model \
#         --max_epoch 30 \
#         --lang_only_self \
#         --nhid_lang 512 \
#         --mark_dots_mentioned \
#         --word_attention_constrained \
#         --hidden_context \
#         --hidden_attention_constrained \
#         --hid2output 1-hidden-layer  \
#         --attention_type sigmoid

### < HIDDEN MENTION ENCODER >
# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         hierarchical_only-self_nhid-lang-512_mark-dots-mentioned_word-attention-constrained_hidden-mention-encoder_mask-pad_h2o-1_sigmoid \
#         --model_type hierarchical_rnn_reference_model \
#         --max_epoch 30 \
#         --lang_only_self \
#         --nhid_lang 512 \
#         --mark_dots_mentioned \
#         --word_attention_constrained \
#         --hidden_context \
#         --hidden_context_mention_encoder \
#         --hid2output 1-hidden-layer  \
#         --attention_type sigmoid
# # !!! LANG ONLY !!!
# ${script} ./train_rel3_dial_model_separate.sh \
#         hierarchical_only-self_nhid-lang-512_mark-dots-mentioned_word-attention-constrained_hidden-mention-encoder_mask-pad_h2o-1_sigmoid \
#         --model_type hierarchical_rnn_reference_model \
#         --max_epoch 30 \
#         --lang_only_self \
#         --nhid_lang 512 \
#         --mark_dots_mentioned \
#         --word_attention_constrained \
#         --hidden_context \
#         --hidden_context_mention_encoder \
#         --hid2output 1-hidden-layer  \
#         --attention_type sigmoid

# !!! LANG ONLY !!!
# ${script} ./train_rel3_dial_model_separate.sh \
#         hierarchical_only-self_mark-dots-mentioned_word-attention-constrained_hidden-mention-encoder_mask-pad_h2o-1_sigmoid \
#         --model_type hierarchical_rnn_reference_model \
#         --max_epoch 30 \
#         --lang_only_self \
#         --mark_dots_mentioned \
#         --word_attention_constrained \
#         --hidden_context \
#         --hidden_context_mention_encoder \
#         --hid2output 1-hidden-layer  \
#         --attention_type sigmoid

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         hierarchical_only-self_mark-dots-mentioned_word-attention-constrained_hidden-mention-encoder_mask-pad_h2o-1_sigmoid \
#         --model_type hierarchical_rnn_reference_model \
#         --max_epoch 30 \
#         --lang_only_self \
#         --mark_dots_mentioned \
#         --word_attention_constrained \
#         --hidden_context \
#         --hidden_context_mention_encoder \
#         --hid2output 1-hidden-layer  \
#         --attention_type sigmoid

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         hierarchical_only-self_untied_nhid-lang-512_mark-dots-mentioned_word-attention-constrained_hidden-mention-encoder_mask-pad_h2o-1_sigmoid \
#         --model_type hierarchical_rnn_reference_model \
#         --max_epoch 30 \
#         --lang_only_self \
#         --untie_grus \
#         --nhid_lang 512 \
#         --mark_dots_mentioned \
#         --word_attention_constrained \
#         --hidden_context \
#         --hidden_context_mention_encoder \
#         --hid2output 1-hidden-layer  \
#         --attention_type sigmoid
# !!! LANG ONLY !!!
# ${script} ./train_rel3_dial_model_separate.sh \
#         hierarchical_only-self_untied_nhid-lang-512_mark-dots-mentioned_word-attention-constrained_hidden-mention-encoder_mask-pad_h2o-1_sigmoid \
#         --model_type hierarchical_rnn_reference_model \
#         --max_epoch 30 \
#         --lang_only_self \
#         --untie_grus \
#         --nhid_lang 512 \
#         --mark_dots_mentioned \
#         --word_attention_constrained \
#         --hidden_context \
#         --hidden_context_mention_encoder \
#         --hid2output 1-hidden-layer  \
#         --attention_type sigmoid

### </ HIDDEN MENTION ENCODER >

## !!! LANG ONLY !!!
# ${script} ./train_rel3_dial_model_separate.sh \
#         hierarchical_only-self_mark-dots-mentioned_word-attention-constrained_mask-pad_h2o-1_sigmoid \
#         --model_type hierarchical_rnn_reference_model \
#         --max_epoch 30 \
#         --lang_only_self \
#         --mark_dots_mentioned \
#         --word_attention_constrained \
#         --hid2output 1-hidden-layer  \
#         --attention_type sigmoid
## !!! LANG ONLY !!!
# ${script} ./train_rel3_dial_model_separate.sh \
#         hierarchical_only-self_nhid-lang-512_mark-dots-mentioned_word-attention-constrained_mask-pad_h2o-1_sigmoid \
#         --model_type hierarchical_rnn_reference_model \
#         --max_epoch 30 \
#         --lang_only_self \
#         --nhid_lang 512 \
#         --mark_dots_mentioned \
#         --word_attention_constrained \
#         --hid2output 1-hidden-layer  \
#         --attention_type sigmoid

# ## !!! LANG ONLY !!!
# ${script} ./train_rel3_dial_model_separate.sh \
#         hierarchical_only-self_mark-dots-mentioned_miwp_word-attention-constrained_mask-pad_h2o-1_sigmoid \
#         --model_type hierarchical_rnn_reference_model \
#         --max_epoch 30 \
#         --lang_only_self \
#         --mark_dots_mentioned \
#         --marks_in_word_prediction \
#         --word_attention_constrained \
#         --hid2output 1-hidden-layer  \
#         --attention_type sigmoid
# ## !!! LANG ONLY !!!
# ${script} ./train_rel3_dial_model_separate.sh \
#         hierarchical_only-self_nhid-lang-512_mark-dots-mentioned_miwp_word-attention-constrained_mask-pad_h2o-1_sigmoid \
#         --model_type hierarchical_rnn_reference_model \
#         --max_epoch 30 \
#         --lang_only_self \
#         --nhid_lang 512 \
#         --mark_dots_mentioned \
#         --marks_in_word_prediction \
#         --word_attention_constrained \
#         --hid2output 1-hidden-layer  \
#         --attention_type sigmoid

# ## !!! LANG ONLY !!!
# ${script} ./train_rel3_dial_model_separate.sh \
#         hierarchical_only-self_mark-dots-mentioned_no-word-attention_feed-attention-constrained_mask-pad_h2o-1_sigmoid \
#         --model_type hierarchical_rnn_reference_model \
#         --max_epoch 30 \
#         --lang_only_self \
#         --mark_dots_mentioned \
#         --no_word_attention \
#         --feed_context \
#         --feed_context_attend \
#         --feed_attention_constrained \
#         --hid2output 1-hidden-layer  \
#         --attention_type sigmoid
# ${script} ./train_rel3_dial_model_separate.sh \
#         hierarchical_only-self_nhid-lang-512_mark-dots-mentioned_no-word-attention_feed-attention-constrained_mask-pad_h2o-1_sigmoid \
#         --model_type hierarchical_rnn_reference_model \
#         --max_epoch 30 \
#         --lang_only_self \
#         --nhid_lang 512 \
#         --mark_dots_mentioned \
#         --no_word_attention \
#         --feed_context \
#         --feed_context_attend \
#         --feed_attention_constrained \
#         --hid2output 1-hidden-layer  \
#         --attention_type sigmoid

## !!! LANG ONLY !!!
# ${script} ./train_rel3_dial_model_separate.sh \
#         hierarchical_only-self_mark-dots-mentioned_miwp_word-attention-constrained_feed-attention-constrained_mask-pad_h2o-1_sigmoid \
#         --model_type hierarchical_rnn_reference_model \
#         --max_epoch 30 \
#         --lang_only_self \
#         --mark_dots_mentioned \
#         --marks_in_word_prediction \
#         --word_attention_constrained \
#         --feed_context \
#         --feed_context_attend \
#         --feed_attention_constrained \
#         --hid2output 1-hidden-layer  \
#         --attention_type sigmoid
# ${script} ./train_rel3_dial_model_separate.sh \
#         hierarchical_only-self_nhid-lang-512_mark-dots-mentioned_miwp_word-attention-constrained_feed-attention-constrained_mask-pad_h2o-1_sigmoid \
#         --model_type hierarchical_rnn_reference_model \
#         --max_epoch 30 \
#         --lang_only_self \
#         --nhid_lang 512 \
#         --mark_dots_mentioned \
#         --marks_in_word_prediction \
#         --word_attention_constrained \
#         --feed_context \
#         --feed_context_attend \
#         --feed_attention_constrained \
#         --hid2output 1-hidden-layer  \
#         --attention_type sigmoid

# ${script} ./train_rel3_dial_model_separate.sh \
#         hierarchical_only-self_nhid-lang-512_mark-dots-mentioned_word-attention-constrained_hidden-attention-constrained_mask-pad_h2o-1_sigmoid \
#         --model_type hierarchical_rnn_reference_model \
#         --max_epoch 30 \
#         --lang_only_self \
#         --nhid_lang 512 \
#         --mark_dots_mentioned \
#         --word_attention_constrained \
#         --hidden_context \
#         --hidden_attention_constrained \
#         --hid2output 1-hidden-layer  \
#         --attention_type sigmoid

# ${script} ./train_rel3_dial_model_separate.sh \
#         hierarchical_only-self_nhid-lang-512_mark-dots-mentioned_no-word-attention_hidden-attention-constrained_mask-pad_h2o-1_sigmoid \
#         --model_type hierarchical_rnn_reference_model \
#         --max_epoch 30 \
#         --lang_only_self \
#         --nhid_lang 512 \
#         --mark_dots_mentioned \
#         --no_word_attention \
#         --hidden_context \
#         --hidden_attention_constrained \
#         --hid2output 1-hidden-layer  \
#         --attention_type sigmoid

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         hierarchical_only-self_partial-structured-temporal-attention-no-marg_stat-relational_mark-dots-mentioned_miwp_word-attention-constrained_feed-attention-constrained_partner-ref_next-mention_mask-pad_h2o-1_sigmoid \
#         --model_type hierarchical_rnn_reference_model \
#         --max_epoch 30 \
#         --lang_only_self \
#         --structured_attention \
#         --structured_attention_no_marginalize \
#         --structured_temporal_attention  \
#         --structured_temporal_attention_transitions relational  \
#         --mark_dots_mentioned \
#         --marks_in_word_prediction \
#         --word_attention_constrained \
#         --feed_context \
#         --feed_context_attend \
#         --feed_attention_constrained \
#         --hid2output 1-hidden-layer  \
#         --attention_type sigmoid \
#         --partner_reference_prediction \
#         --next_mention_prediction

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         hierarchical_only-self_partial-structured-attention-no-marg_mark-dots-mentioned_word-attention-constrained_partner-ref_next-mention_mask-pad_h2o-1_sigmoid \
#         --model_type hierarchical_rnn_reference_model \
#         --max_epoch 30 \
#         --lang_only_self \
#         --structured_attention \
#         --structured_attention_no_marginalize \
#         --mark_dots_mentioned \
#         --word_attention_constrained \
#         --hid2output 1-hidden-layer  \
#         --attention_type sigmoid \
#         --partner_reference_prediction \
#         --next_mention_prediction

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         hierarchical_only-self_partial-structured-temporal-attention-no-marg_mark-dots-mentioned_word-attention-constrained_partner-ref_next-mention_mask-pad_h2o-1_sigmoid \
#         --model_type hierarchical_rnn_reference_model \
#         --max_epoch 30 \
#         --lang_only_self \
#         --structured_attention \
#         --structured_attention_no_marginalize \
#         --structured_temporal_attention  \
#         --mark_dots_mentioned \
#         --word_attention_constrained \
#         --hid2output 1-hidden-layer  \
#         --attention_type sigmoid \
#         --partner_reference_prediction \
#         --next_mention_prediction

## Line 26 in the "ref pragmatics" part of the sheet
# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         hierarchical_only-self_partial-structured-temporal-attention-no-marg_stat-relational_mark-dots-mentioned_word-attention-constrained_partner-ref_next-mention_mask-pad_h2o-1_sigmoid \
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
#         --next_mention_prediction

## untied gru version of the above
# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         hierarchical_only-self_untied_partial-structured-temporal-attention-no-marg_stat-relational_mark-dots-mentioned_word-attention-constrained_partner-ref_next-mention_mask-pad_h2o-1_sigmoid \
#         --model_type hierarchical_rnn_reference_model \
#         --max_epoch 30 \
#         --lang_only_self \
#         --untie_grus \
#         --structured_attention \
#         --structured_attention_no_marginalize \
#         --structured_temporal_attention  \
#         --structured_temporal_attention_transitions relational  \
#         --mark_dots_mentioned \
#         --word_attention_constrained \
#         --hid2output 1-hidden-layer  \
#         --attention_type sigmoid \
#         --partner_reference_prediction \
#         --next_mention_prediction

## hidden mention encoder version of the above
# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         hierarchical_only-self_untied_partial-structured-temporal-attention-no-marg_stat-relational_mark-dots-mentioned_hidden-mention-encoder_word-attention-constrained_partner-ref_next-mention_mask-pad_h2o-1_sigmoid \
#         --model_type hierarchical_rnn_reference_model \
#         --max_epoch 30 \
#         --lang_only_self \
#         --untie_grus \
#         --structured_attention \
#         --structured_attention_no_marginalize \
#         --structured_temporal_attention  \
#         --structured_temporal_attention_transitions relational  \
#         --mark_dots_mentioned \
#         --word_attention_constrained \
#         --hidden_context \
#         --hidden_context_mention_encoder \
#         --hid2output 1-hidden-layer  \
#         --attention_type sigmoid \
#         --partner_reference_prediction \
#         --next_mention_prediction

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         hierarchical_only-self_untied_partial-structured-temporal-attention-no-marg_stat-relational-lang-cond_mark-dots-mentioned_hidden-mention-encoder_word-attention-constrained_partner-ref_next-mention_mask-pad_h2o-1_sigmoid_plateau \
#         --model_type hierarchical_rnn_reference_model \
#         --max_epoch 30 \
#         --lang_only_self \
#         --untie_grus \
#         --structured_attention \
#         --structured_attention_no_marginalize \
#         --structured_temporal_attention  \
#         --structured_temporal_attention_transitions relational  \
#         --structured_attention_language_conditioned \
#         --mark_dots_mentioned \
#         --word_attention_constrained \
#         --hidden_context \
#         --hidden_context_mention_encoder \
#         --hid2output 1-hidden-layer  \
#         --attention_type sigmoid \
#         --partner_reference_prediction \
#         --next_mention_prediction \
#         --reduce_plateau

${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
        hierarchical_only-self_partial-structured-temporal-attention-no-marg_stat-relational-lang-cond_mark-dots-mentioned_hidden-mention-encoder_word-attention-constrained_partner-ref_next-mention_mask-pad_h2o-1_sigmoid_plateau \
        --model_type hierarchical_rnn_reference_model \
        --max_epoch 30 \
        --lang_only_self \
        --structured_attention \
        --structured_attention_no_marginalize \
        --structured_temporal_attention  \
        --structured_temporal_attention_transitions relational  \
        --structured_attention_language_conditioned \
        --mark_dots_mentioned \
        --word_attention_constrained \
        --hidden_context \
        --hidden_context_mention_encoder \
        --hid2output 1-hidden-layer  \
        --attention_type sigmoid \
        --partner_reference_prediction \
        --next_mention_prediction \
        --reduce_plateau


## 512-dim version of the above
# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         hierarchical_only-self_nhid-lang-512_untied_partial-structured-temporal-attention-no-marg_stat-relational_mark-dots-mentioned_hidden-mention-encoder_word-attention-constrained_partner-ref_next-mention_mask-pad_h2o-1_sigmoid \
#         --model_type hierarchical_rnn_reference_model \
#         --max_epoch 30 \
#         --lang_only_self \
#         --nhid_lang 512 \
#         --untie_grus \
#         --structured_attention \
#         --structured_attention_no_marginalize \
#         --structured_temporal_attention  \
#         --structured_temporal_attention_transitions relational  \
#         --mark_dots_mentioned \
#         --word_attention_constrained \
#         --hidden_context \
#         --hidden_context_mention_encoder \
#         --hid2output 1-hidden-layer  \
#         --attention_type sigmoid \
#         --partner_reference_prediction \
#         --next_mention_prediction

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         hierarchical_only-self_nhid-lang-512_untied_partial-structured-temporal-attention-no-marg_stat-relational-lang-cond_mark-dots-mentioned_hidden-mention-encoder_word-attention-constrained_partner-ref_next-mention_mask-pad_h2o-1_sigmoid \
#         --model_type hierarchical_rnn_reference_model \
#         --max_epoch 30 \
#         --lang_only_self \
#         --nhid_lang 512 \
#         --untie_grus \
#         --structured_attention \
#         --structured_attention_no_marginalize \
#         --structured_temporal_attention \
#         --structured_temporal_attention_transitions relational \
#         --structured_attention_language_conditioned \
#         --mark_dots_mentioned \
#         --word_attention_constrained \
#         --hidden_context \
#         --hidden_context_mention_encoder \
#         --hid2output 1-hidden-layer \
#         --attention_type sigmoid \
#         --partner_reference_prediction \
#         --next_mention_prediction

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         hierarchical_only-self_partial-structured-attention_mark-dots-mentioned_word-attention-constrained_next-mention_mask-pad_h2o-1_sigmoid \
#         --model_type hierarchical_rnn_reference_model \
#         --max_epoch 30 \
#         --lang_only_self \
#         --structured_attention \
#         --mark_dots_mentioned \
#         --word_attention_constrained \
#         --next_mention_prediction \
#         --hid2output 1-hidden-layer  \
#         --attention_type sigmoid

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         hierarchical_only-self_partial-structured-attention-no-marg_mark-dots-mentioned_word-attention-constrained_next-mention_mask-pad_h2o-1_sigmoid \
#         --model_type hierarchical_rnn_reference_model \
#         --max_epoch 30 \
#         --lang_only_self \
#         --structured_attention \
#         --structured_attention_no_marginalize \
#         --mark_dots_mentioned \
#         --word_attention_constrained \
#         --next_mention_prediction \
#         --hid2output 1-hidden-layer  \
#         --attention_type sigmoid

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         hierarchical_only-self_mark-dots-mentioned_only-first_word-attention-constrained_mask-pad_h2o-1_sigmoid \
#         --model_type hierarchical_rnn_reference_model \
#         --max_epoch 30 \
#         --lang_only_self \
#         --mark_dots_mentioned \
#         --only_first_mention \
#         --word_attention_constrained \
#         --hid2output 1-hidden-layer  \
#         --attention_type sigmoid

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         hierarchical_only-self_partial-structured-attention_mark-dots-mentioned_only-first_word-attention-constrained_next-mention_mask-pad_h2o-1_sigmoid \
#         --model_type hierarchical_rnn_reference_model \
#         --max_epoch 30 \
#         --lang_only_self \
#         --structured_attention \
#         --mark_dots_mentioned \
#         --only_first_mention \
#         --word_attention_constrained \
#         --next_mention_prediction \
#         --hid2output 1-hidden-layer  \
#         --attention_type sigmoid

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         hierarchical_only-self_partial-structured-attention-no-marg_mark-dots-mentioned_only-first_word-attention-constrained_next-mention_mask-pad_h2o-1_sigmoid \
#         --model_type hierarchical_rnn_reference_model \
#         --max_epoch 30 \
#         --lang_only_self \
#         --structured_attention \
#         --structured_attention_no_marginalize \
#         --mark_dots_mentioned \
#         --only_first_mention \
#         --word_attention_constrained \
#         --next_mention_prediction \
#         --hid2output 1-hidden-layer  \
#         --attention_type sigmoid
