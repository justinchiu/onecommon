#!/bin/bash
script=""

for clip in 0.5 1.0 5.0
do
  for lr in 1e-3 5e-3 1e-4
  do
    # ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
    #         optimization/lr-${lr}_clip-${clip}_plateau \
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
    #         --lr=${lr} \
    #         --clip=${clip} \
    #         --reduce_plateau 

    ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
            optimization/lr-${lr}_clip-${clip} \
            --model_type hierarchical_rnn_reference_model \
            --max_epoch 30 \
            --lang_only_self \
            --untie_grus \
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
            --lr=${lr} \
            --clip=${clip}
  done
done

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
