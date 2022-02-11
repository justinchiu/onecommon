#!/bin/bash

model_dir_a=$1
model_dir_b=$2
name=$3
shared_ctx_count=4

shift
shift
shift

log_file=${model_dir_a}/${name}_ctx-${shared_ctx_count}.selfplay.log
out_file=${model_dir_a}/${name}_ctx-${shared_ctx_count}.selfplay.out

python -u -m pdb selfplay.py \
  --alice_model_file=${model_dir_a}/1_ep-12.th \
  --bob_model_file=${model_dir_b}/1_ep-12.th \
  --context_file=shared_${shared_ctx_count} \
  --cuda \
  --markable_detector_file=serialized_models/markable_detector_with_dict_1.th \
  --verbose \
  --log_file=${log_file} $@

exit 0

# tee-ing to out for now, messes up signals

python -u selfplay.py \
  --alice_model_file=${model_dir_a}/1_ep-12.th \
  --bob_model_file=${model_dir_b}/1_ep-12.th \
  --context_file=shared_${shared_ctx_count} \
  --cuda \
  --markable_detector_file=serialized_models/markable_detector_with_dict_1.th \
  --verbose \
  --log_file=${log_file} \
  $@ \
  | tee ${out_file}
