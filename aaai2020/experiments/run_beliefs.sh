#!/bin/bash

#script="./sbatch_gpu.sh"
script="./sbatch_1080ti.sh"

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         hierarchical_only-self_sigmoid_mask-pad_selection-beliefs-selected \
#         --model_type hierarchical_rnn_reference_model \
#         --lang_only_self \
#         --attention_type sigmoid \
#         --selection_beliefs selected

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         hierarchical_only-self_sigmoid_mask-pad_generation-beliefs-selected \
#         --model_type hierarchical_rnn_reference_model \
#         --lang_only_self \
#         --attention_type sigmoid \
#         --generation_beliefs selected

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         hierarchical_only-self_sigmoid_mask-pad_generation-beliefs-partners \
#         --model_type hierarchical_rnn_reference_model \
#         --lang_only_self \
#         --attention_type sigmoid \
#         --generation_beliefs partners

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
#         hierarchical_only-self_sigmoid_mask-pad \
#         --model_type hierarchical_rnn_reference_model \
#         --lang_only_self \
#         --attention_type sigmoid

## non-sigmoid replications: does marking help? seems to, when not using sigmoid attention

# ${script} ./train_rel3_dial_model_separate.sh \
#         hierarchical_only-self_mask-pad_mark-dots-mentioned \
#         --model_type hierarchical_rnn_reference_model \
#         --lang_only_self \
#         --mark_dots_mentioned

# ${script} ./train_rel3_dial_model_separate.sh \
#         hierarchical_only-self_mask-pad \
#         --model_type hierarchical_rnn_reference_model \
#         --lang_only_self

# ${script} ./train_rel2_dial_model_separate.sh \
#         hierarchical_only-self_mask-pad_mark-dots-mentioned_seed-2 \
#         --model_type hierarchical_rnn_reference_model \
#         --lang_only_self \
#         --mark_dots_mentioned \
#         --seed 2

# ${script} ./train_rel2_dial_model_separate.sh \
#         hierarchical_only-self_mask-pad_seed-2 \
#         --model_type hierarchical_rnn_reference_model \
#         --lang_only_self \
#         --seed 2

## non-sigmoid beliefs:

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         hierarchical_only-self_mask-pad_generation-beliefs-selected \
#         --model_type hierarchical_rnn_reference_model \
#         --lang_only_self \
#         --generation_beliefs selected

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         hierarchical_only-self_mask-pad_generation-beliefs-partners \
#         --model_type hierarchical_rnn_reference_model \
#         --lang_only_self \
#         --generation_beliefs partners

# ${script} ./train_rel3_dial_model_separate.sh \
#         hierarchical_only-self_mask-pad_generation-beliefs-selected \
#         --model_type hierarchical_rnn_reference_model \
#         --lang_only_self \
#         --generation_beliefs selected

# ${script} ./train_rel3_dial_model_separate.sh \
#         hierarchical_only-self_mask-pad_generation-beliefs-partners \
#         --model_type hierarchical_rnn_reference_model \
#         --lang_only_self \
#         --generation_beliefs partners

## non-sigmoid selected beliefs:

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         hierarchical_only-self_mask-pad_check \
#         --model_type hierarchical_rnn_reference_model \
#         --lang_only_self

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         hierarchical_only-self_mask-pad_generation-beliefs-last-partner-mentioned \
#         --model_type hierarchical_rnn_reference_model \
#         --lang_only_self \
#         --generation_beliefs last_partner_mentioned

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         hierarchical_only-self_mask-pad_generation-beliefs-cumulative-partner-mentioned \
#         --model_type hierarchical_rnn_reference_model \
#         --lang_only_self \
#         --generation_beliefs cumulative_partner_mentioned

# ${script} ./train_rel3_dial_model_separate.sh \
#         hierarchical_only-self_mask-pad_generation-beliefs-last-partner-mentioned \
#         --model_type hierarchical_rnn_reference_model \
#         --lang_only_self \
#         --generation_beliefs last_partner_mentioned

# ${script} ./train_rel3_dial_model_separate.sh \
#         hierarchical_only-self_mask-pad_generation-beliefs-cumulative-partner-mentioned \
#         --model_type hierarchical_rnn_reference_model \
#         --lang_only_self \
#         --generation_beliefs cumulative_partner_mentioned

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         hierarchical_only-self_mask-pad_selection-beliefs-last-partner-mentioned \
#         --model_type hierarchical_rnn_reference_model \
#         --lang_only_self \
#         --selection_beliefs last_partner_mentioned

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         hierarchical_only-self_mask-pad_selection-beliefs-cumulative-partner-mentioned \
#         --model_type hierarchical_rnn_reference_model \
#         --lang_only_self \
#         --selection_beliefs cumulative_partner_mentioned

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         hierarchical_only-self_mask-pad_selection-beliefs-selected \
#         --model_type hierarchical_rnn_reference_model \
#         --lang_only_self \
#         --selection_beliefs selected

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         hierarchical_only-self_mask-pad_selection-beliefs-partners \
#         --model_type hierarchical_rnn_reference_model \
#         --lang_only_self \
#         --selection_beliefs partners

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         hierarchical_only-self_mask-pad_mark-dots-mentioned \
#         --model_type hierarchical_rnn_reference_model \
#         --lang_only_self \
#         --mark_dots_mentioned

# sigmoid vs non-sigmoid

# ${script} ./train_rel3_dial_model_separate.sh \
#         hierarchical_only-self_mask-pad_mark-dots-mentioned_2 \
#         --model_type hierarchical_rnn_reference_model \
#         --lang_only_self \
#         --mark_dots_mentioned

# ${script} ./train_rel3_dial_model_separate.sh \
#         hierarchical_only-self_sigmoid_mask-pad_mark-dots-mentioned_2 \
#         --model_type hierarchical_rnn_reference_model \
#         --lang_only_self \
#         --attention_type sigmoid \
#         --mark_dots_mentioned

# ${script} ./train_rel3_dial_model_separate.sh \
#         hierarchical_only-self_sigmoid_mask-pad_generation-beliefs-selected \
#         --model_type hierarchical_rnn_reference_model \
#         --lang_only_self \
#         --attention_type sigmoid \
#         --generation_beliefs selected

# ${script} ./train_rel3_dial_model_separate.sh \
#         hierarchical_only-self_sigmoid_mask-pad_generation-beliefs-partners \
#         --model_type hierarchical_rnn_reference_model \
#         --lang_only_self \
#         --attention_type sigmoid \
#         --generation_beliefs partners

# ${script} ./train_rel3_dial_model_separate.sh \
#         hierarchical_only-self_sigmoid_mask-pad_generation-beliefs-last-partner-mentioned \
#         --model_type hierarchical_rnn_reference_model \
#         --lang_only_self \
#         --attention_type sigmoid \
#         --generation_beliefs last_partner_mentioned

# ${script} ./train_rel3_dial_model_separate.sh \
#         hierarchical_only-self_sigmoid_mask-pad_generation-beliefs-cumulative-partner-mentioned \
#         --model_type hierarchical_rnn_reference_model \
#         --lang_only_self \
#         --attention_type sigmoid \
#         --generation_beliefs cumulative_partner_mentioned

# testing whether we get better performance on the savio2_1080 cluster than on the savio2_gpu cluster
# ${script} ./train_rel3_dial_model_separate.sh \
#         hierarchical_only-self_mask-pad_mark-dots-mentioned_3-1080 \
#         --model_type hierarchical_rnn_reference_model \
#         --lang_only_self \
#         --mark_dots_mentioned

# ${script} ./train_rel3_dial_model_separate.sh \
#         hierarchical_only-self_sigmoid_mask-pad_mark-dots-mentioned_3-1080 \
#         --model_type hierarchical_rnn_reference_model \
#         --lang_only_self \
#         --attention_type sigmoid \
#         --mark_dots_mentioned

## RERUN, to verify after refactoring
# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         hierarchical_only-self_partner-ref_mask-pad_h2o-1_check \
#         --model_type hierarchical_rnn_reference_model \
#         --lang_only_self \
#         --hid2output 1-hidden-layer  \
#         --partner_reference_prediction 

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         hierarchical_only-self_partner-ref_mask-pad_h2o-1_selection-beliefs-last-partner-mentioned_check \
#         --model_type hierarchical_rnn_reference_model \
#         --lang_only_self \
#         --hid2output 1-hidden-layer  \
#         --partner_reference_prediction \
#         --selection_beliefs last_partner_mentioned

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         hierarchical_only-self_partner-ref_mask-pad_h2o-1_selection-beliefs-last-partner-mentioned-predicted \
#         --model_type hierarchical_rnn_reference_model \
#         --lang_only_self \
#         --hid2output 1-hidden-layer  \
#         --partner_reference_prediction \
#         --selection_beliefs last_partner_mentioned_predicted

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         hierarchical_only-self_partner-ref_mask-pad_h2o-1_selection-beliefs-last-partner-mentioned-predicted_detach-beliefs \
#         --model_type hierarchical_rnn_reference_model \
#         --lang_only_self \
#         --hid2output 1-hidden-layer  \
#         --partner_reference_prediction \
#         --selection_beliefs last_partner_mentioned_predicted \
#         --detach_beliefs

## 60 epochs
# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         hierarchical_only-self_partner-ref_mask-pad_h2o-1_ep-60 \
#         --model_type hierarchical_rnn_reference_model \
#         --max_epoch 60 \
#         --lang_only_self \
#         --hid2output 1-hidden-layer  \
#         --partner_reference_prediction 

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         hierarchical_only-self_partner-ref_mask-pad_h2o-1_selection-beliefs-last-partner-mentioned_ep-60 \
#         --model_type hierarchical_rnn_reference_model \
#         --max_epoch 60 \
#         --lang_only_self \
#         --hid2output 1-hidden-layer  \
#         --partner_reference_prediction \
#         --selection_beliefs last_partner_mentioned

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         hierarchical_only-self_partner-ref_mask-pad_h2o-1_selection-beliefs-last-partner-mentioned-predicted_ep-60 \
#         --model_type hierarchical_rnn_reference_model \
#         --max_epoch 60 \
#         --lang_only_self \
#         --hid2output 1-hidden-layer  \
#         --partner_reference_prediction \
#         --selection_beliefs last_partner_mentioned_predicted

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         hierarchical_only-self_partner-ref_mask-pad_h2o-1_selection-beliefs-last-partner-mentioned-predicted_detach-beliefs_ep-60 \
#         --model_type hierarchical_rnn_reference_model \
#         --max_epoch 60 \
#         --lang_only_self \
#         --hid2output 1-hidden-layer  \
#         --partner_reference_prediction \
#         --selection_beliefs last_partner_mentioned_predicted \
#         --detach_beliefs

# next mention prediction
${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
        hierarchical_only-self_next-mention_mask-pad_h2o-1_ep-60 \
        --model_type hierarchical_rnn_reference_model \
        --max_epoch 60 \
        --lang_only_self \
        --hid2output 1-hidden-layer  \
        --next_mention_prediction

${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
        hierarchical_only-self_partner-ref_next-mention_mask-pad_h2o-1_ep-60 \
        --model_type hierarchical_rnn_reference_model \
        --max_epoch 60 \
        --lang_only_self \
        --hid2output 1-hidden-layer  \
        --partner_reference_prediction  \
        --next_mention_prediction
