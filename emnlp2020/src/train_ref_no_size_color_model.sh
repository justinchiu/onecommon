python train_reference.py \
	--ctx_encoder_type attn_encoder \
	--max_epoch 30 \
	--optimizer adam \
	--model_file ref_no_size_color_model \
	--cuda \
	--tensorboard_log \
	--nembed_word 256 \
	--nembed_ctx 256 \
	--nhid_lang 256 \
	--nhid_attn 256 \
	--nhid_sel 256 \
	--lang_weight 1.0 \
	--ref_weight 1.0 \
	--num_ref_weight 0 \
	--sel_weight 0.03125 \
	--clip 0.5 \
	--dropout 0.5 \
	--unk_threshold 10 \
	--repeat_train \
	--share_attn \
	--remove_size_color \