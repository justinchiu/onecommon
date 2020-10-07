#!/bin/bash
script=""

mkdir expts/rel3_tsel_ref_dial_model_separate/pipeline_debug_9-20 2>/dev/null

# d26f263f8: either self/partner beliefs and oracle beliefs in agent
# 0fb5f3758: dot_recurrence: add mention attention (doesn't help) and uniform weighting (hurts)
# TODO?: ed4705c5a: refactor and update agent to use beliefs
# 2effd2d7f: add dot_recurrence oracle, using true mentions
# f9465e59d: fix import and store top words

base_name="pipeline_debug_9-20/base"
base_args="--model_type hierarchical_rnn_reference_model \
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
        --max_epoch 20 \
        --ref_beliefs last_mentioned_predicted last_partner_mentioned_predicted last_primary_mentioned_predicted last_partner_primary_mentioned_predicted \
        --ref_beliefs_warmup last_mentioned last_partner_mentioned last_primary_mentioned last_partner_primary_mentioned \
        --ref_beliefs_warmup_epochs 2 \
        --partner_ref_beliefs last_mentioned_predicted last_partner_mentioned_predicted last_primary_mentioned_predicted last_partner_primary_mentioned_predicted \
        --partner_ref_beliefs_warmup last_mentioned last_partner_mentioned last_primary_mentioned last_partner_primary_mentioned \
        --partner_ref_beliefs_warmup_epochs 2 \
        --mention_beliefs this_partner_mentioned last_mentioned \
        --selection_beliefs this_partner_mentioned last_mentioned \
        --selection_start_epoch 2 \
        --next_mention_start_epoch 2  \
        --max_instances_per_split 1000 "

#hash=`git rev-parse HEAD`
hash="either_beliefs_mod"

${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
        ${base_name}_${hash} \
        $base_args \
