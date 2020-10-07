#!/bin/bash

model_dir=$1
epoch=$2

if [[ -z $model_dir ]]
then
  echo "pass a model dir"
  exit 1;
fi

if [[ -z $epoch ]]
then
  epoch=30
  epoch_str=""
else
  epoch_str="_epoch-${epoch}"
fi

model_file=${model_dir}/1_ep-${epoch}.th
#model_file=${model_dir}

split=dev

# python test_reference.py \
#   --model_file=$model_file \
#   --lang_only_self \
#   --allow_belief_cheating \
#   --reference_prediction=l0 \
#   --eval_split=${split} \
#   --model_referent_annotation_output_path=${model_dir}/eval_${split}_l0_ref_annotations.json \
#   --cuda \
#   | tee ${model_dir}/eval_${split}_l0.out_temporal${epoch_str}

#for candidates in 20 40
#for candidates in 20 
#for candidates in 40  20 10
#for candidates in 5 1
for candidates in 40 
do
  # python -u test_reference.py \
  #   --model_file=$model_file \
  #   --lang_only_self \
  #   --allow_belief_cheating \
  #   --reference_prediction=l1 \
  #   --partner_reference_prediction=l1 \
  #   --l1_speaker_weight=0.0 \
  #   --l1_candidates=$candidates \
  #   --l1_oracle \
  #   --eval_split=${split} \
  #   --cuda \
  #   | tee ${model_dir}/eval_${split}_cand=${candidates}_oracle.out

  #for speaker_weight in 0.05 0.15 0.25 
  #for speaker_weight in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
  # for speaker_weight in 0 0.5 1.0 0.2 0.4 0.6 0.8 0.3 0.7 0.9
  # do
  #   echo "candidates=${candidates}; speaker_weight=${speaker_weight}"
  # python -u test_reference.py \
  #   --model_file=$model_file \
  #   --lang_only_self \
  #   --allow_belief_cheating \
  #   --reference_prediction=l1 \
  #   --l1_speaker_weight=$speaker_weight \
  #   --l1_candidates=$candidates \
  #   --l1_renormalize \
  #   --eval_split=${split} \
  #   --cuda \
  #   | tee ${model_dir}/eval_${split}_cand=${candidates}_renorm_speaker-weight=${speaker_weight}.out_temporal${epoch_str}
  # done

  python -u test_reference.py \
    --model_file=$model_file \
    --lang_only_self \
    --allow_belief_cheating \
    --reference_prediction=l1 \
    --l1_speaker_weight=1.0 \
    --l1_speaker_weights 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --l1_candidates=$candidates \
    --l1_renormalize \
    --eval_split=${split} \
    --cuda \
    | tee ${model_dir}/eval_${split}_cand=${candidates}_renorm_speaker-weights.out_temporal${epoch_str}
done
