# unilmv1-Paddle
**Reproduction of "Unified pre-training for natural language understanding (NLU) and generation (NLG)"**


**```Update: ```** **Check out the latest information and models of UniLM at [https://github.com/microsoft/unilm/tree/master/unilm](https://github.com/microsoft/unilm/tree/master/unilm)**


**\*\*\*\*\* November 6th, 2021: Reproduction of UniLM v1 on QNLI Tasks\*\*\*\*\***

## Environment

* Hardware: NVIDIA V100 or V100S is recommended
* python == 3.6.8
* Cuda 10.2 + cudnn 7.6.5
* PaddlePaddle == 2.2.0rc0

## Pre-trained Model

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
Trained Model can be downloaded here:

link: https://pan.baidu.com/s/1yMBo9AjIsjVzM4GU6IOwIg
password: b0uf
```bash
# run evaluate
bash qnli_eval.sh
```

## Reference
```
@inproceedings{unilm,
    title={Unified Language Model Pre-training for Natural Language Understanding and Generation},
    author={Dong, Li and Yang, Nan and Wang, Wenhui and Wei, Furu and Liu, Xiaodong and Wang, Yu and Gao, Jianfeng and Zhou, Ming and Hon, Hsiao-Wuen},
    year={2019},
    booktitle = "33rd Conference on Neural Information Processing Systems (NeurIPS 2019)"
}
```