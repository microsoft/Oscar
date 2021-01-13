# Oscar: Object-Semantics Aligned Pre-training for Vision-and-Language Tasks    <img src="docs/oscar_logo.png" width="200" align="right"> 
## Updates
05/28/2020: Released finetuned models on downstream tasks, please check [MODEL_ZOO.md](MODEL_ZOO.md). <br/>
05/15/2020: Released pretrained models, datasets, and code for downstream tasks finetuning. <br/>
01/13/2021: our new work [VinVL](https://arxiv.org/abs/2101.00529) proposed OSCAR+, an improved version of OSCAR, and provided a better object-attribute detection model to extract features for V+L tasks. The VinVL work achieved SOTA performance on all seven V+L tasks here. Please stay tuned for the model and code release. 

## Introduction
This repository contains source code necessary to reproduce the results presented in the paper [Oscar: Object-Semantics Aligned Pre-training for Vision-Language Tasks](https://arxiv.org/abs/2004.06165).
We propose a new cross-modal pre-training method **Oscar** (Object-Semantics Aligned Pre-training). It leverages **object tags** detected in images as anchor points to significantly ease the learning of image-text alignments. We pre-train Oscar on the public corpus of 6.5 million text-image pairs, and fine-tune it on downstream tasks, creating new state-of-the-arts on six well-established vision-language understanding and generation tasks. For more on this project, see the [Microsoft Research Blog post](https://www.microsoft.com/en-us/research/blog/objects-are-the-secret-key-to-revealing-the-world-between-vision-and-language/).


<img src="docs/oscar.PNG" width="650"> 

## Performance
Task    | t2i | t2i | i2t | i2t | IC  | IC  |  IC  |  IC  | NoCaps | NoCaps |   VQA    |  NLVR2  |
--------|-----|-----|-----|-----|-----|-----|------|------|--------|--------|----------|---------|
Metric	| R@1 | R@5 | R@1 | R@5 | B@4 |  M  |  C   |   S  |    C   |    S   | test-std | test-P  |
SoTA_S  |39.2 | 68.0|56.6 | 84.5|38.9 |29.2 |129.8 | 22.4 |   61.5 |  9.2   |  70.90   | 53.50   |
SoTA_B  |48.4 | 76.7|63.3 | 87.0|39.5 |29.3 |129.3 | 23.2 |   73.1 | 11.2   |  72.54   | 78.87   |
SoTA_L  |51.7 | 78.4|66.6 | 89.4|  -  |  -  |   -  |   -  |     -  |   -    |  73.40   | 79.50   |
-----   |---  |---  |---  |---  |---  |---  |---   |---   |---     |---     |---       |---      |
Oscar_B |54.0 | 80.8|70.0 | 91.1|40.5 |29.7 |137.6 | 22.8 |   78.8 | 11.7   |  73.44   | 78.36   |
Oscar_L |57.5 | 82.8|73.5 | 92.2|41.7 |30.6 |140.0 | 24.5 |   80.9 | 11.3   |  73.82   | 80.05   |
gain    | 5.8 |  4.4| 6.9 |  2.8| 2.2 | 1.3 | 10.7 | 1.3  |    7.8 |  0.5   |   0.42   |  0.55   |

t2i: text-to-image retrieval; i2t: image-to-text retrieval; IC: image captioning on COCO. 


## Download
We released pre-trained models and datasets for downstream tasks. Please check [DOWNLOAD.md](DOWNLOAD.md) for details. 

## Installation
Check [INSTALL.md](INSTALL.md) for installation instructions.

## Model Zoo
Check [MODEL_ZOO.md](MODEL_ZOO.md) for scripts to run each downstream tasks and the expected performance.

## Citations
Please consider citing this paper if you use the code:
```
@article{li2020oscar,
  title={Oscar: Object-Semantics Aligned Pre-training for Vision-Language Tasks},
  author={Li, Xiujun and Yin, Xi and Li, Chunyuan and Hu, Xiaowei and Zhang, Pengchuan and Zhang, Lei and Wang, Lijuan and Hu, Houdong and Dong, Li and Wei, Furu and Choi, Yejin and Gao, Jianfeng},
  journal={arXiv preprint arXiv:2004.06165},
  year={2020}
}

@article{zhang2021vinvl,
  title={VinVL: Making Visual Representations Matter in Vision-Language Models},
  author={Zhang, Pengchuan and Li, Xiujun and Hu, Xiaowei and Yang, Jianwei and Zhang, Lei and Wang, Lijuan and Choi, Yejin and Gao, Jianfeng},
  journal={arXiv preprint arXiv:2101.00529},
  year={2021}
}
```

## License
Oscar is released under the MIT license. See [LICENSE](LICENSE) for details. 

