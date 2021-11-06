# unilmv1-Paddle
**Reproduction of "Unified pre-training for natural language understanding (NLU) and generation (NLG)"**


**```Update: ```** **Check out the latest information and models of UniLM at [https://github.com/microsoft/unilm/tree/master/unilm](https://github.com/microsoft/unilm/tree/master/unilm)**


**\*\*\*\*\* November 6th, 2021: Reproduction of UniLM v1 in QNLI Tasks\*\*\*\*\***

## Environment

* Hardware: NVIDIA V100 or V100S is recommended
* python == 3.6.8
* Cuda 10.2 + cudnn 7.6.5
* PaddlePaddle == 2.2.0rc0

## Pre-trained Models

Pre-trained Models is available here:

link: https://pan.baidu.com/s/143Lb12BS_36ztjXTywBZJg
password: n0it




## Quick Start
We train the model on QNLI Dataset and GLUE Score achieves 92.8 on average. 
```bash
git clone https://github.com/fuqiang-git-hub/unilmv1-Paddle.git
bash src_paddle/qnli.sh
```

## Evaluate
Trained Model can be downloaded from 

```bash

```

The size of full training data (3.8M) is quite large. We can stop the fine-tuning procedure after 10 epochs.

We provide a fine-tuned checkpoint (downloaded from [here](https://drive.google.com/open?id=1jOI2nO16Uz4a0OWZ7Ro-jnD54MHMDlsv)) used for decoding. The inference and evaluation process is conducted as follows:
```bash
DATA_DIR=/{path_of_data}/gigaword
MODEL_RECOVER_PATH=/{path_of_fine-tuned_model}/ggw38m_model.bin
EVAL_SPLIT=test
export PYTORCH_PRETRAINED_BERT_CACHE=/{tmp_folder}/bert-cased-pretrained-cache
# run decoding
python biunilm/decode_seq2seq.py --fp16 --amp --bert_model bert-large-cased --new_segment_ids --mode s2s --need_score_traces \
  --input_file ${DATA_DIR}/${EVAL_SPLIT}.src --split ${EVAL_SPLIT} --tokenized_input \
  --model_recover_path ${MODEL_RECOVER_PATH} \
  --max_seq_length 192 --max_tgt_length 32 \
  --batch_size 64 --beam_size 5 --length_penalty 0 \
  --forbid_duplicate_ngrams --forbid_ignore_word "."
# apply length penalty
python biunilm/gen_seq_from_trace.py --bert_model bert-large-cased --alpha 0.6 \
  --input ${MODEL_RECOVER_PATH}.${EVAL_SPLIT}
# run evaluation
python gigaword/eval.py --pred ${MODEL_RECOVER_PATH}.${EVAL_SPLIT}.alp0.6 \
  --gold ${DATA_DIR}/org_data/${EVAL_SPLIT}.tgt.txt --perl
```

The program `eval.py` generates a post-processed output file `${MODEL_RECOVER_PATH}.${EVAL_SPLIT}.alp0.6.post` (downloaded from [here](https://drive.google.com/open?id=1oycvzMC6ZoWZV7BOt5OlZ7q0SxM_0Zc9)).

### Abstractive Summarization - [CNN / Daily Mail](https://github.com/harvardnlp/sent-summary)
                                                                                          | ROUGE-1   | ROUGE-2   | ROUGE-L   |
| -----------------------------------------------------------------------------
| Model                                                           ---------------------------------------------------------------------------- | --------- | --------- | --------- |
| [PGNet (See et al., 2017)](https://www.aclweb.org/anthology/P17-1099)                                                                                     | 39.53     | 17.28     | 36.38     |
| [Bottom-Up (Gehrmann et al., 2018)](https://www.aclweb.org/anthology/D18-1443)                                                                            | 41.22     | 18.68     | 38.34     |
| [GPT-2 TL;DR: (Radford et al., 2019)](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) | 29.34     | 8.27      | 26.58     |
| [MASS (Song et al., 2019)](https://github.com/microsoft/MASS#results-on-abstractive-summarization-9272019)                                                | 42.12     | 19.50     | 39.01     |
| [BertShare (Rothe et al., 2019)](https://arxiv.org/pdf/1907.12461.pdf)                                                                                    | 39.25     | 18.09     | 36.45     |
| [BertSumAbs (Liu and Lapata, 2019)](https://arxiv.org/pdf/1908.08345.pdf)                                                                                 | 41.72     | 19.39     | 38.76     |
| **UniLM**                                                                                                                                                 | **43.08** | **20.43** | **40.34** |

The data can be downloaded from [here](https://drive.google.com/open?id=1jiDbDbAsqy_5BM79SmX6aSu5DQVCAZq1).

```bash
# run fine-tuning
DATA_DIR=/{path_of_data}/cnn_dailymail
OUTPUT_DIR=/{path_of_fine-tuned_model}/
MODEL_RECOVER_PATH=/{path_of_pre-trained_model}/unilmv1-large-cased.bin
export PYTORCH_PRETRAINED_BERT_CACHE=/{tmp_folder}/bert-cased-pretrained-cache
export CUDA_VISIBLE_DEVICES=0,1,2,3
python biunilm/run_seq2seq.py --do_train --fp16 --amp --num_workers 0 \
  --bert_model bert-large-cased --new_segment_ids --tokenized_input \
  --data_dir ${DATA_DIR} \
  --output_dir ${OUTPUT_DIR}/bert_save \
  --log_dir ${OUTPUT_DIR}/bert_log \
  --model_recover_path ${MODEL_RECOVER_PATH} \
  --max_seq_length 768 --max_position_embeddings 768 \
  --trunc_seg a --always_truncate_tail \
  --max_len_a 568 --max_len_b 200 \
  --mask_prob 0.7 --max_pred 140 \
  --train_batch_size 48 --gradient_accumulation_steps 2 \
  --learning_rate 0.00003 --warmup_proportion 0.1 --label_smoothing 0.1 \
  --num_train_epochs 30  
```

We provide a fine-tuned checkpoint (downloaded from [here](https://drive.google.com/open?id=1RyJxShxC9tDYVAyZwUwqkSoQ3l5DfjuE)) used for decoding. The inference and evaluation process is conducted as follows:

```bash
DATA_DIR=/{path_of_data}/cnn_dailymail
MODEL_RECOVER_PATH=/{path_of_fine-tuned_model}/cnndm_model.bin
EVAL_SPLIT=test
export PYTORCH_PRETRAINED_BERT_CACHE=/{tmp_folder}/bert-cased-pretrained-cache
# run decoding
python biunilm/decode_seq2seq.py --fp16 --amp --bert_model bert-large-cased --new_segment_ids --mode s2s --need_score_traces \
  --input_file ${DATA_DIR}/${EVAL_SPLIT}.src --split ${EVAL_SPLIT} --tokenized_input \
  --model_recover_path ${MODEL_RECOVER_PATH} \
  --max_seq_length 768 --max_tgt_length 128 \
  --batch_size 64 --beam_size 5 --length_penalty 0 \
  --forbid_duplicate_ngrams --forbid_ignore_word ".|[X_SEP]"
# apply length penalty
python biunilm/gen_seq_from_trace.py --bert_model bert-large-cased --alpha 1.0 \
  --input ${MODEL_RECOVER_PATH}.${EVAL_SPLIT}
# run evaluation
python cnndm/eval.py --pred ${MODEL_RECOVER_PATH}.${EVAL_SPLIT}.alp1.0 \
  --gold ${DATA_DIR}/org_data/${EVAL_SPLIT}.summary --trunc_len 70 --perl
```

The program `eval.py` generates a post-processed output file `${MODEL_RECOVER_PATH}.${EVAL_SPLIT}.alp1.0.post` (downloaded from [here](https://drive.google.com/open?id=1p93XD0wo3YvyxZnNYywujtnQoNCDiTF7)).

### Question Generation - [SQuAD](https://arxiv.org/abs/1806.03822)

We present the results following the same [data split](https://github.com/xinyadu/nqg/tree/master/data) and [evaluation scripts](https://github.com/xinyadu/nqg/tree/master/qgevalcap) as in [(Du et al., 2017)](https://arxiv.org/pdf/1705.00106.pdf).

| Model                                                              | BLEU-4    | METEOR    | ROUGE-L   |
| ------------------------------------------------------------------ | --------- | --------- | --------- |
| [(Du and Cardie, 2018)](https://www.aclweb.org/anthology/P18-1177) | 15.16     | 19.12     | -         |
| [(Zhang and Bansal, 2019)](https://arxiv.org/pdf/1909.06356.pdf)   | 18.37     | 22.65     | 46.68     |
| **UniLM**                                                          | **22.78** | **25.49** | **51.57** |

We also report the results following the data split as in [(Zhao et al., 2018)](https://aclweb.org/anthology/D18-1424), which uses the reversed dev-test setup.

| Model                                                            | BLEU-4    | METEOR    | ROUGE-L   |
| ---------------------------------------------------------------- | --------- | --------- | --------- |
| [(Zhao et al., 2018)](https://aclweb.org/anthology/D18-1424)     | 16.38     | 20.25     | 44.48     |
| [(Zhang and Bansal, 2019)](https://arxiv.org/pdf/1909.06356.pdf) | 20.76     | 24.20     | 48.91     |
| **UniLM**                                                        | **24.32** | **26.10** | **52.69** |

Note: If we directly use the tokenized references provided by [Du et al. (2017)](https://arxiv.org/pdf/1705.00106.pdf), the results are (22.17 BLEU-4 / 25.47 METEOR / 51.53 ROUGE-L) on the [raw data split](https://github.com/xinyadu/nqg/tree/master/data), and (23.69 BLEU-4 / 26.08 METEOR / 52.70 ROUGE-L) in the reversed dev-test setup.

Our processed data can be downloaded from [here](https://drive.google.com/open?id=11E3Ij-ctbRUTIQjueresZpoVzLMPlVUZ).

```bash
# run fine-tuning
DATA_DIR=/{path_of_data}/qg/train
OUTPUT_DIR=/{path_of_fine-tuned_model}/
MODEL_RECOVER_PATH=/{path_of_pre-trained_model}/unilmv1-large-cased.bin
export PYTORCH_PRETRAINED_BERT_CACHE=/{tmp_folder}/bert-cased-pretrained-cache
export CUDA_VISIBLE_DEVICES=0,1,2,3
python biunilm/run_seq2seq.py --do_train --num_workers 0 \
  --bert_model bert-large-cased --new_segment_ids --tokenized_input \
  --data_dir ${DATA_DIR} --src_file train.pa.tok.txt --tgt_file train.q.tok.txt \
  --output_dir ${OUTPUT_DIR}/bert_save \
  --log_dir ${OUTPUT_DIR}/bert_log \
  --model_recover_path ${MODEL_RECOVER_PATH} \
  --max_seq_length 512 --max_position_embeddings 512 \
  --mask_prob 0.7 --max_pred 48 \
  --train_batch_size 32 --gradient_accumulation_steps 2 \
  --learning_rate 0.00002 --warmup_proportion 0.1 --label_smoothing 0.1 \
  --num_train_epochs 10
```

We provide a fine-tuned checkpoint (downloaded from [here](https://drive.google.com/open?id=1JN2wnkSRotwUnJ_Z-AbWwoPdP53Gcfsn)) used for decoding. The inference and evaluation process is conducted as follows:

```bash
DATA_DIR=/{path_of_data}/qg/test
MODEL_RECOVER_PATH=/{path_of_fine-tuned_model}/qg_model.bin
EVAL_SPLIT=test
export PYTORCH_PRETRAINED_BERT_CACHE=/{tmp_folder}/bert-cased-pretrained-cache
# run decoding
python biunilm/decode_seq2seq.py --bert_model bert-large-cased --new_segment_ids --mode s2s \
  --input_file ${DATA_DIR}/$test.pa.tok.txt --split ${EVAL_SPLIT} --tokenized_input \
  --model_recover_path ${MODEL_RECOVER_PATH} \
  --max_seq_length 512 --max_tgt_length 48 \
  --batch_size 16 --beam_size 1 --length_penalty 0
# run evaluation using our tokenized data as reference
python qg/eval_on_unilm_tokenized_ref.py --out_file qg/output/qg.test.output.txt
# run evaluation using tokenized data of Du et al. (2017) as reference
python qg/eval.py --out_file qg/output/qg.test.output.txt
```

The files `qg/eval_on_unilm_tokenized_ref.py` and `qg/eval.py` are in Python 2.\*, because they are dependent on the [evaluation scripts](https://github.com/xinyadu/nqg/tree/master/qgevalcap) of [Du et al., (2017)](https://arxiv.org/pdf/1705.00106.pdf). The output files can be downloaded from [here](https://drive.google.com/open?id=1MdaRftgl_HMqN7DLvYmw-zKkvOBZCP6U). Notice that our model predictions are cased, while the gold outputs provided by Du et al., (2017) are uncased. So the predicted results need to be converted to lowercase before computing the evaluation metrics.

## FAQ

- Install ROUGE-1.5.5
  - If we would like to use the Perl script of ROUGE, it can be installed by following [instruction-1](https://gist.github.com/donglixp/d7eea02d57ba2e099746f8463c2f6597) and [instruction-2](https://github.com/bheinzerling/pyrouge#installation). The ROUGE-1.5.5 package (written in Perl) can be downloaded from [here](https://github.com/andersjo/pyrouge/tree/master/tools/ROUGE-1.5.5). We can also use the Python-version evaluation script by removing the flag `--perl` when running `eval.py`. Notice that there would be slight number difference between them due to the implementation details.
  
- [Run inference using CPUs](https://github.com/microsoft/unilm/issues/23#issuecomment-549788510)
  - Run `decode_seq2seq.py` without the flags `--amp` and `--fp16`, and uninstall the python package `nvidia/apex`.

## Citation

If you find UniLM useful in your work, you can cite the following paper:
```
@inproceedings{unilm,
    title={Unified Language Model Pre-training for Natural Language Understanding and Generation},
    author={Dong, Li and Yang, Nan and Wang, Wenhui and Wei, Furu and Liu, Xiaodong and Wang, Yu and Gao, Jianfeng and Zhou, Ming and Hon, Hsiao-Wuen},
    year={2019},
    booktitle = "33rd Conference on Neural Information Processing Systems (NeurIPS 2019)"
}
```

## Related Projects/Codebase

- Vision-Language Pre-training: https://github.com/LuoweiZhou/VLP
- MT-DNN: https://github.com/namisan/mt-dnn
- Response Generation Pre-training: https://github.com/microsoft/DialoGPT

## Acknowledgments
Our code is based on [pytorch-transformers v0.4.0](https://github.com/huggingface/pytorch-transformers/tree/v0.4.0). We thank the authors for their wonderful open-source efforts.

## License
This project is licensed under the license found in the LICENSE file in the root directory of this source tree.
Portions of the source code are based on the [pytorch-transformers v0.4.0](https://github.com/huggingface/pytorch-transformers/tree/v0.4.0) project.

[Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct)

### Contact Information

For help or issues using UniLM, please submit a GitHub issue.

For other communications related to UniLM, please contact Li Dong (`lidong1@microsoft.com`), Furu Wei (`fuwei@microsoft.com`).
