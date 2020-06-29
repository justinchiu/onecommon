#!/bin/bash

#script="./sbatch_gpu.sh"
# script="./sbatch_1080ti.sh"
script=""

# next mention prediction
${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
        hierarchical_only-self_mbfix-last_mentioned+this-partner-mentioned+cumulative-mentioned+cumulative-partner-mentioned_next-mention_mask-pad_h2o-1_sigmoid \
        --model_type hierarchical_rnn_reference_model \
        --max_epoch 30 \
        --lang_only_self \
        --hid2output 1-hidden-layer  \
        --next_mention_prediction \
        --mention_beliefs last_mentioned this_partner_mentioned cumulative_mentioned cumulative_partner_mentioned \
        --attention_type sigmoid

${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
        hierarchical_only-self_mbfix-last_mentioned+this-partner-mentioned_next-mention_mask-pad_h2o-1_sigmoid \
        --model_type hierarchical_rnn_reference_model \
        --max_epoch 30 \
        --lang_only_self \
        --hid2output 1-hidden-layer  \
        --next_mention_prediction \
        --mention_beliefs last_mentioned this_partner_mentioned  \
        --attention_type sigmoid

${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
        hierarchical_only-self_mbfix-last_mentioned_next-mention_mask-pad_h2o-1_sigmoid \
        --model_type hierarchical_rnn_reference_model \
        --max_epoch 30 \
        --lang_only_self \
        --hid2output 1-hidden-layer  \
        --next_mention_prediction \
        --mention_beliefs last_mentioned \
        --attention_type sigmoid

${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
        hierarchical_only-self_mbfix-cumulative-mentioned_next-mention_mask-pad_h2o-1_sigmoid \
        --model_type hierarchical_rnn_reference_model \
        --max_epoch 30 \
        --lang_only_self \
        --hid2output 1-hidden-layer  \
        --next_mention_prediction \
        --mention_beliefs cumulative_mentioned \
        --attention_type sigmoid

${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
        hierarchical_only-self_mbfix-last_mentioned+this-partner-mentioned+cumulative-mentioned_next-mention_mask-pad_h2o-1_sigmoid \
        --model_type hierarchical_rnn_reference_model \
        --max_epoch 30 \
        --lang_only_self \
        --hid2output 1-hidden-layer  \
        --next_mention_prediction \
        --mention_beliefs last_mentioned this_partner_mentioned cumulative_mentioned \
        --attention_type sigmoid
