#!/bin/bash

#script="./sbatch_gpu.sh"
script="./sbatch_1080ti.sh"

## these have already been run

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         hierarchical_only-self_sigmoid_mask-pad_mark-dots-mentioned \
#         --model_type hierarchical_rnn_reference_model \
#         --lang_only_self \
#         --attention_type sigmoid \
#         --mark_dots_mentioned

# ${script} ./train_rel3_dial_model_separate.sh \
#         hierarchical_only-self_sigmoid_mask-pad_mark-dots-mentioned \
#         --model_type hierarchical_rnn_reference_model \
#         --lang_only_self \
#         --attention_type sigmoid \
#         --mark_dots_mentioned

# ${script} ./train_rel3_dial_model_separate.sh \
#         hierarchical_only-self_mask-pad_mark-dots-mentioned \
#         --model_type hierarchical_rnn_reference_model \
#         --lang_only_self \
#         --mark_dots_mentioned

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         hierarchical_only-self_mask-pad_mark-dots-mentioned \
#         --model_type hierarchical_rnn_reference_model \
#         --lang_only_self \
#         --mark_dots_mentioned

# ${script} ./train_rel3_dial_model_separate.sh \
#         hierarchical_only-self_mask-pad_mark-dots-mentioned_rel-2 \
#         --model_type hierarchical_rnn_reference_model \
#         --lang_only_self \
#         --relation_encoder_layers 2 \
#         --mark_dots_mentioned

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         hierarchical_only-self_mask-pad_mark-dots-mentioned_rel-2 \
#         --model_type hierarchical_rnn_reference_model \
#         --lang_only_self \
#         --relation_encoder_layers 2 \
#         --mark_dots_mentioned

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         hierarchical_only-self_sigmoid_mask-pad_mark-dots-mentioned_rel-2 \
#         --model_type hierarchical_rnn_reference_model \
#         --lang_only_self \
#         --attention_type sigmoid \
#         --relation_encoder_layers 2 \
#         --mark_dots_mentioned

# ${script} ./train_rel3_dial_model_separate.sh \
#         hierarchical_only-self_sigmoid_mask-pad_mark-dots-mentioned_rel-2 \
#         --model_type hierarchical_rnn_reference_model \
#         --lang_only_self \
#         --attention_type sigmoid \
#         --relation_encoder_layers 2 \
#         --mark_dots_mentioned

## generation beliefs for the best model of the above

# ${script} ./train_rel3_dial_model_separate.sh \
#         hierarchical_only-self_mask-pad_generation-beliefs-selected_rel-2 \
#         --model_type hierarchical_rnn_reference_model \
#         --lang_only_self \
#         --relation_encoder_layers 2 \
#         --generation_beliefs selected

# ${script} ./train_rel3_dial_model_separate.sh \
#         hierarchical_only-self_mask-pad_generation-beliefs-partners_rel-2 \
#         --model_type hierarchical_rnn_reference_model \
#         --lang_only_self \
#         --relation_encoder_layers 2 \
#         --generation_beliefs partners

# ${script} ./train_rel3_dial_model_separate.sh \
#         hierarchical_only-self_mask-pad_generation-beliefs-last-partner-mentioned_rel-2 \
#         --model_type hierarchical_rnn_reference_model \
#         --lang_only_self \
#         --relation_encoder_layers 2 \
#         --generation_beliefs last_partner_mentioned

# ${script} ./train_rel3_dial_model_separate.sh \
#         hierarchical_only-self_mask-pad_generation-beliefs-cumulative-partner-mentioned_rel-2 \
#         --model_type hierarchical_rnn_reference_model \
#         --lang_only_self \
#         --relation_encoder_layers 2 \
#         --generation_beliefs cumulative_partner_mentioned

## better hid2output conditioning for mark_dots_mentioned

# ${script} ./train_rel3_dial_model_separate.sh \
#         hierarchical_only-self_mask-pad_mark-dots-mentioned_h2o-1 \
#         --model_type hierarchical_rnn_reference_model \
#         --lang_only_self \
#         --hid2output 1-hidden-layer \
#         --mark_dots_mentioned

# ${script} ./train_rel3_dial_model_separate.sh \
#         hierarchical_only-self_mask-pad_mark-dots-mentioned_h2o-2 \
#         --model_type hierarchical_rnn_reference_model \
#         --lang_only_self \
#         --hid2output 2-hidden-layer \
#         --mark_dots_mentioned

# ${script} ./train_rel3_dial_model_separate.sh \
#         hierarchical_only-self_mask-pad_mark-dots-mentioned_miwp \
#         --model_type hierarchical_rnn_reference_model \
#         --lang_only_self \
#         --marks_in_word_prediction \
#         --mark_dots_mentioned

# ${script} ./train_rel3_dial_model_separate.sh \
#         hierarchical_only-self_mask-pad_mark-dots-mentioned_miwp_h2o-1 \
#         --model_type hierarchical_rnn_reference_model \
#         --lang_only_self \
#         --hid2output 1-hidden-layer \
#         --marks_in_word_prediction \
#         --mark_dots_mentioned

# ${script} ./train_rel3_dial_model_separate.sh \
#         hierarchical_only-self_mask-pad_mark-dots-mentioned_miwp_h2o-2 \
#         --model_type hierarchical_rnn_reference_model \
#         --lang_only_self \
#         --hid2output 2-hidden-layer \
#         --marks_in_word_prediction \
#         --mark_dots_mentioned

## test best hid2output conditioning with various belief states

# ${script} ./train_rel3_dial_model_separate.sh \
#         hierarchical_only-self_mask-pad_generation-beliefs-selected_miwp_h2o-1 \
#         --model_type hierarchical_rnn_reference_model \
#         --lang_only_self \
#         --hid2output 1-hidden-layer \
#         --marks_in_word_prediction \
#         --generation_beliefs selected

# ${script} ./train_rel3_dial_model_separate.sh \
#         hierarchical_only-self_mask-pad_generation-beliefs-partners_miwp_h2o-1 \
#         --model_type hierarchical_rnn_reference_model \
#         --lang_only_self \
#         --hid2output 1-hidden-layer \
#         --marks_in_word_prediction \
#         --generation_beliefs partners

# ${script} ./train_rel3_dial_model_separate.sh \
#         hierarchical_only-self_mask-pad_generation-beliefs-last-partner-mentioned_miwp_h2o-1 \
#         --model_type hierarchical_rnn_reference_model \
#         --lang_only_self \
#         --hid2output 1-hidden-layer \
#         --marks_in_word_prediction \
#         --generation_beliefs last_partner_mentioned

# ${script} ./train_rel3_dial_model_separate.sh \
#         hierarchical_only-self_mask-pad_generation-beliefs-cumulative-partner-mentioned_miwp_h2o-1 \
#         --model_type hierarchical_rnn_reference_model \
#         --lang_only_self \
#         --hid2output 1-hidden-layer \
#         --marks_in_word_prediction \
#         --generation_beliefs cumulative_partner_mentioned

# ${script} ./train_rel3_dial_model_separate.sh \
#         hierarchical_only-self_mask-pad_generation-beliefs-partners+last-partner-mentioned_miwp_h2o-1 \
#         --model_type hierarchical_rnn_reference_model \
#         --lang_only_self \
#         --hid2output 1-hidden-layer \
#         --marks_in_word_prediction \
#         --generation_beliefs partners last_partner_mentioned

# ${script} ./train_rel3_dial_model_separate.sh \
#         hierarchical_only-self_mask-pad_generation-beliefs-partners+cumulative-partner-mentioned_miwp_h2o-1 \
#         --model_type hierarchical_rnn_reference_model \
#         --lang_only_self \
#         --hid2output 1-hidden-layer \
#         --marks_in_word_prediction \
#         --generation_beliefs partners cumulative_partner_mentioned

## test best hid2output conditioning with context feeding

# ${script} ./train_rel3_dial_model_separate.sh \
#         hierarchical_only-self_mask-pad_mark-dots-mentioned_miwp_h2o-1_feed-context-attend \
#         --model_type hierarchical_rnn_reference_model \
#         --feed_context \
#         --feed_context_attend \
#         --lang_only_self \
#         --hid2output 1-hidden-layer \
#         --marks_in_word_prediction \
#         --mark_dots_mentioned

# ${script} ./train_rel3_dial_model_separate.sh \
#         hierarchical_only-self_mask-pad_mark-dots-mentioned_miwp_h2o-1_feed-context-attend_no-word-attention \
#         --model_type hierarchical_rnn_reference_model \
#         --feed_context \
#         --feed_context_attend \
#         --no_word_attention \
#         --lang_only_self \
#         --hid2output 1-hidden-layer \
#         --marks_in_word_prediction \
#         --mark_dots_mentioned

## mark first mention
## these don't seem to have any positive effect
# ${script} ./train_rel3_dial_model_separate.sh \
#         hierarchical_only-self_mask-pad_mark-dots-mentioned-first_miwp_h2o-1 \
#         --model_type hierarchical_rnn_reference_model \
#         --lang_only_self \
#         --hid2output 1-hidden-layer \
#         --marks_in_word_prediction \
#         --mark_dots_mentioned \
#         --only_first_mention

# ${script} ./train_rel3_dial_model_separate.sh \
#         hierarchical_only-self_mask-pad_generation-beliefs-last-partner-mentioned-first_miwp_h2o-1 \
#         --model_type hierarchical_rnn_reference_model \
#         --lang_only_self \
#         --hid2output 1-hidden-layer \
#         --marks_in_word_prediction \
#         --generation_beliefs last_partner_mentioned \
#         --only_first_mention

# ${script} ./train_rel3_dial_model_separate.sh \
#         hierarchical_only-self_mask-pad_generation-beliefs-cumulative-partner-mentioned-first_miwp_h2o-1 \
#         --model_type hierarchical_rnn_reference_model \
#         --lang_only_self \
#         --hid2output 1-hidden-layer \
#         --marks_in_word_prediction \
#         --generation_beliefs cumulative_partner_mentioned \
#         --only_first_mention
