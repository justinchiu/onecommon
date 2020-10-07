#!/bin/bash
script=""

mkdir expts/rel3_tsel_ref_dial_model_separate/directions_debug 2>/dev/null

branch=master_fixed

${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
        directions_debug/unstructured_${branch} \
        --model_type hierarchical_rnn_reference_model \
        --max_epoch 30 \
        --lang_only_self \
        --mark_dots_mentioned \
        --word_attention_constrained \
        --hid2output 1-hidden-layer  \
        --attention_type sigmoid \
        --reduce_plateau

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         directions_debug/structured_${branch} \
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
