#!/bin/bash

#script="./sbatch_gpu.sh"
script="./sbatch_1080ti.sh"

# better hid2output conditioning for mark_dots_mentioned

${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
        hierarchical_only-self_mask-pad_h2o-1 \
        --model_type hierarchical_rnn_reference_model \
        --lang_only_self \
        --hid2output 1-hidden-layer

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         hierarchical_only-self_partner-ref_mask-pad_h2o-1 \
#         --model_type hierarchical_rnn_reference_model \
#         --lang_only_self \
#         --hid2output 1-hidden-layer \
#         --partner_reference_prediction 

# ${script} ./train_rel3_tsel_ref_dial_model_separate.sh \
#         hierarchical_partner-ref_mask-pad_h2o-1 \
#         --model_type hierarchical_rnn_reference_model \
#         --hid2output 1-hidden-layer \
#         --partner_reference_prediction 
