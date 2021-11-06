export CUDA_VISIBLE_DEVICES=0
python -m paddle.distributed.launch run_qnli.py --do_train --recover_on True --amp --num_workers 0 --bert_model '{PATH_TO_PRETRAINED_MODEL}/bert-large-cased-paddle.tar.gz' --vocab "{PATH_TO_VOCAB}/bert-large-cased-vocab.txt" --new_segment_ids --tokenized_input --data_dir 'DATA_PATH' --src_file 'train.src.10k' --tgt_file 'train.tgt.10k' --output_dir '{PATH_TO_UNILM}/giga_finetune_out/bert_save' --log_dir '{PATH_TO_UNILM}/giga_finetune_out_3/bert_log' --model_recover_path '{PATH_TO_PRETRAINED_MODEL}/unilm1-large-cased.pdparams' --max_seq_length 512 --max_position_embeddings 512 --trunc_seg a --always_truncate_tail --max_len_b 64 --mask_prob 0.7 --max_pred 64 --train_batch_size 16 --gradient_accumulation_steps 1 --learning_rate 0.000001 --warmup_proportion 0.1 --num_train_epochs 30 --local_rank -1