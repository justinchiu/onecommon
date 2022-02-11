#!/bin/bash

model_dir=$1

if [[ -z $model_dir ]]
then
  echo "pass a model dir"
  exit 1;
fi

model_file=${model_dir}/1_ep-30.th
model_file=${model_dir}/1_ep-12.th
#model_file=${model_dir}

split=dev

python test_reference.py \
  --model_file=$model_file \
  --lang_only_self \
  --allow_belief_cheating \
  --reference_prediction=l0 \
  --eval_split=${split} \
  --model_referent_annotation_output_path=${model_dir}/eval_${split}_l0_ref_annotations.json \
  --cuda \
  | tee ${model_dir}/eval_${split}_l0.out_temporal

#for candidates in 20 40
#for candidates in 20 
#for candidates in 40  20 10
#for candidates in 5 1
for candidates in 40 128
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
  for speaker_weight in 0 0.5 1.0
  do
    echo "candidates=${candidates}; speaker_weight=${speaker_weight}"
  python -u test_reference.py \
    --model_file=$model_file \
    --lang_only_self \
    --allow_belief_cheating \
    --reference_prediction=l1 \
    --l1_speaker_weight=$speaker_weight \
    --l1_candidates=$candidates \
    --eval_split=${split} \
    --cuda \
    | tee ${model_dir}/eval_${split}_cand=${candidates}_speaker-weight=${speaker_weight}.out_temporal
  done
done
