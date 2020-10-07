#!/bin/bash
script=""

mkdir expts/rel3_tsel_ref_dial_model_separate/nm_multi_ref 2>/dev/null

base_name="nm_multi_ref/base"
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

${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
        ${base_name}_nm-multi-ref_loss-expanded \
        $base_args  \
        --next_mention_prediction_type=multi_reference \

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         no-mention \
#         --model_type hierarchical_rnn_reference_model \
#         --max_epoch 20 \
#         --lang_only_self \
#         --structured_attention \
#         --structured_attention_no_marginalize \
#         --structured_temporal_attention  \
#         --structured_temporal_attention_transitions relational  \
#         --structured_attention_language_conditioned \
#         --mark_dots_mentioned \
#         --word_attention_constrained \
#         --hid2output 1-hidden-layer  \
#         --attention_type sigmoid \
#         --partner_reference_prediction \
#         --nhid_lang 512 \
#         --encode_relative_to_extremes \
#         --learned_pooling \
#         --untie_grus \
#         --bidirectional_reader \
#         --hidden_context \
#         --hidden_context_mention_encoder \
#         --detach_beliefs \
#         --reduce_plateau \
