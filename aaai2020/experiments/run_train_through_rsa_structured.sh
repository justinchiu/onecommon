#!/bin/bash
script=""

mkdir expts/rel3_tsel_ref_dial_model_separate/train_through_rsa 2>/dev/null

BSZ=4

base_name="train_through_rsa/structured_bsz-${BSZ}"
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
        --hidden_context_mention_encoder_type=filtered-separate \
        --bsz $BSZ \
        --max_mentions_per_utterance 8 \
        --reduce_plateau "

# base_args="--model_type hierarchical_rnn_reference_model \
#         --max_epoch 30 \
#         --lang_only_self \
#         --mark_dots_mentioned \
#         --word_attention_constrained \
#         --hid2output 1-hidden-layer  \
#         --attention_type sigmoid \
#         --partner_reference_prediction \
#         --next_mention_prediction \
#         --nhid_lang 512 \
#         --encode_relative_to_extremes \
#         --learned_pooling \
#         --untie_grus \
#         --bidirectional_reader \
#         --hidden_context \
#         --hidden_context_mention_encoder \
#         --hidden_context_mention_encoder_type=filtered-separate \
#         --bsz $BSZ \
#         --max_mentions_per_utterance 8 \
#         --reduce_plateau "

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_mmigt-1 \
#         $base_args \
#         --max_mentions_in_generation_training=1

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_mmigt-1_l1-1.0 \
#         $base_args \
#         --max_mentions_in_generation_training=1 \
#         --l1_loss_weight 1.0 \

# temporal, to check that refactoring didn't break things
# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_mmigt-1_l1-1.0_l1-prior-next-mention_temporal \
#         $base_args \
#         --max_mentions_in_generation_training=1 \
#         --l1_loss_weight=1.0 \
#         --l1_prior=next_mention 

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_mmigt-1_l1-1.0_temporal_l1-norm-sampling-uniform-128 \
#         $base_args \
#         --max_mentions_in_generation_training=1 \
#         --l1_loss_weight=1.0 \
#         --l1_normalizer_sampling=uniform \
#         --l1_normalizer_sampling_candidates=128 

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_mmigt-1_l1-1.0_temporal_l1-norm-sampling-noised-128 \
#         $base_args \
#         --max_mentions_in_generation_training=1 \
#         --l1_loss_weight=1.0 \
#         --l1_normalizer_sampling=noised \
#         --l1_normalizer_sampling_candidates=128 

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_mmigt-1_l1-1.0_l1-prior-next-mention_temporal_l1-norm-sampling-noised-40 \
#         $base_args \
#         --max_mentions_in_generation_training=1 \
#         --l1_loss_weight=1.0 \
#         --l1_prior=next_mention  \
#         --l1_normalizer_sampling=noised \
#         --l1_normalizer_sampling_candidates=40 

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_mmigt-1_l1-1.0_l1-prior-next-mention_temporal_l1-norm-sampling-uniform-40 \
#         $base_args \
#         --max_mentions_in_generation_training=1 \
#         --l1_loss_weight=1.0 \
#         --l1_prior=next_mention  \
#         --l1_normalizer_sampling=uniform \
#         --l1_normalizer_sampling_candidates=40 

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_mmigt-1_l1-1.0_l1-prior-next-mention_temporal_l1-norm-sampling-noised-128 \
#         $base_args \
#         --max_mentions_in_generation_training=1 \
#         --l1_loss_weight=1.0 \
#         --l1_prior=next_mention  \
#         --l1_normalizer_sampling=noised \
#         --l1_normalizer_sampling_candidates=128 

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_mmigt-1_l1-1.0_l1-prior-next-mention_temporal_l1-norm-sampling-uniform-128 \
#         $base_args \
#         --max_mentions_in_generation_training=1 \
#         --l1_loss_weight=1.0 \
#         --l1_prior=next_mention  \
#         --l1_normalizer_sampling=uniform \
#         --l1_normalizer_sampling_candidates=128 

${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
        ${base_name}_l1-1.0_temporal_l1-norm-sampling-noised-128 \
        $base_args \
        --l1_loss_weight=1.0 \
        --l1_normalizer_sampling=noised \
        --l1_normalizer_sampling_candidates=128 

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_l1-1.0_temporal_l1-norm-sampling-uniform-128 \
#         $base_args \
#         --l1_loss_weight=1.0 \
#         --l1_normalizer_sampling=uniform \
#         --l1_normalizer_sampling_candidates=128 

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         ${base_name}_mmigt-1_l1-1.0_lang-0.0 \
#         $base_args \
#         --max_mentions_in_generation_training=1 \
#         --l1_loss_weight 1.0 \
#         --lang_weight 0.0 \
