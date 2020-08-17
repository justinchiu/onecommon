#!/bin/bash

name=$1
if [[ -z $name ]]
then
  name="default"
else
  shift
fi

script_name=rel3_dial_model_separate

model_dir="${script_name}/${name}"
output_dir="expts/${model_dir}"

#mkdir -p serialized_models/$script_name
mkdir -p $output_dir

python -u train_reference.py \
	--ctx_encoder_type rel_attn_encoder_3 \
	--max_epoch 30 \
	--optimizer adam \
	--model_file $model_dir \
	--nembed_word 256 \
	--nembed_ctx 256 \
	--nhid_lang 256 \
	--nhid_attn 256 \
	--nhid_sel 256 \
	--lang_weight 1.0 \
	--ref_weight 0.0 \
  --partner_ref_weight 0.0 \
	--sel_weight 0.0 \
  --next_mention_weight 0.0 \
	--clip 0.5 \
	--dropout 0.5 \
	--unk_threshold 10 \
  --separate_attn \
  --cuda \
  $@ \
  | tee ${output_dir}/train_fold-1.out


#	--repeat_train \
