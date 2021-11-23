# Download

## Datasets
We provide the extracted image region features, object tags, and the original text annotations for each downstream tasks.
```bash
path/to/azcopy copy 'https://biglmdiag.blob.core.windows.net/vinvl/datasets/TASK_NAME' <target folder> --recursive
```
`TASK_NAME` could be `coco_caption`, `nocaps`, `coco_ir`, `vqa`, `gqa`, `nlvr2`.

## Pre-trained Models
We provide pre-trained *Oscar+* models of Bert-base and Bert-large structures, with the name starting with `base` and `large`, respectively.
```bash
path/to/azcopy copy 'https://biglmdiag.blob.core.windows.net/vinvl/model_ckpts/TASK_NAME' <target folder> --recursive
```
`TASK_NAME` could be `image_captioning` (including `nocaps`), `coco_ir`, `vqa`, `gqa`, `nlvr2`, `od_models`.

The models are trained with both image region features and object tags. The image region features are extracted by the Faster R-CNN with
ResNet-101, using object and attribute annotations from [Visual Genome](http://visualgenome.org/).
The object tags are from:
    1) the same VisualGenome model, named as `-vg-labels`. Or,
    2) the model trained on object annotations from [Open Images V5](https://storage.googleapis.com/openimages/web/index.html). named as `-oid-labels`. Or,
    3) no object tags provied, serving as baseline, named as `-no-labels`.

## Pre-exacted Image Features
For ease-of-use, we make pretrained features available for all pretraining datasets and downstream tasks. 
Features are stored in tsv (tab-separated-values) format that can be used in [pretraining](oscar/datasets/oscar_tsv.py) and dowstream tasks like [COCO Image-Text Retrieval](oscar/run_retrieval.py).

Notice that all the links below are links to a folder. We recommend using the following AzCopy command to download.
```
path/to/azcopy copy <folder-link> <target-address> --recursive
```

[COCO 2014 Train/Val Image Features (~50G)](https://biglmdiag.blob.core.windows.net/vinvl/image_features/coco_X152C4_frcnnbig2_exp168model_0060000model.roi_heads.nm_filter_2_model.roi_heads.score_thresh_0.2/model_0060000/)

[COCO 2014 Test Image Features (~16G)](https://biglmdiag.blob.core.windows.net/vinvl/image_features/coco_X152C4_frcnnbig2_exp168model_0060000model.roi_heads.nm_filter_2_model.roi_heads.score_thresh_0.2/model_0060000/coco2014test/)

[COCO 2015 Test Image Features (~32G)](https://biglmdiag.blob.core.windows.net/vinvl/image_features/coco_X152C4_frcnnbig2_exp168model_0060000model.roi_heads.nm_filter_2_model.roi_heads.score_thresh_0.2/model_0060000/coco2015test/)

[GQA All Image Features (~62G)](https://biglmdiag.blob.core.windows.net/vinvl/image_features/gqa_X152C4_frcnnbig2_exp168model_0060000model.roi_heads.nm_filter_2_model.roi_heads.score_thresh_0.2/model_0060000/)

[NVLR2 Train/Del/Test Image Features (~28G)](https://biglmdiag.blob.core.windows.net/vinvl/image_features/nlvr2_X152C4_frcnnbig2_exp168model_0060000model.roi_heads.nm_filter_2_model.roi_heads.score_thresh_0.2/)

[Flickr30k All Image Features (~14G)](https://biglmdiag.blob.core.windows.net/vinvl/image_features/flickr30k_X152C4_frcnnbig2_exp168model_0060000model.roi_heads.nm_filter_2_model.roi_heads.score_thresh_0.2/model_0060000/)

[Google Conceptual Captions Image Features (Huge, ~960G, splitted into 12 chunks)](https://biglmdiag.blob.core.windows.net/vinvl/image_features/googlecc_X152C4_frcnnbig2_exp168model_0060000model.roi_heads.nm_filter_2_model.roi_heads.score_thresh_0.2/)

[SBU Image Features (Huge, ~280G, splitted into 4 chunks)](https://biglmdiag.blob.core.windows.net/vinvl/image_features/sbu_X152C4_frcnnbig2_exp168model_0060000model.roi_heads.nm_filter_2_model.roi_heads.score_thresh_0.2/model_0060000/)

[Open Images Detection Image Features (Huge, ~530G, splitted into 8 chunks)](https://biglmdiag.blob.core.windows.net/vinvl/image_features/oi_X152C4_frcnnbig2_exp168model_0060000model.roi_heads.nm_filter_2_model.roi_heads.score_thresh_0.2/model_0060000/)


## Oscar+ pretraining corpus
<img src="docs/pretrain_corpus.PNG" width="650"> 

[Small corpus](https://biglmdiag.blob.core.windows.net/vinvl/pretrain_corpus/coco_flickr30k_gqa.tsv)

[Medium corpus](https://biglmdiag.blob.core.windows.net/vinvl/pretrain_corpus/coco_flickr30k_gqa_oi.tsv)

[Large corpus](https://biglmdiag.blob.core.windows.net/vinvl/pretrain_corpus/coco_flickr30k_googlecc_gqa_sbu_oi.tsv)

We have tried our best to make sure that there is no data contamination between pretraining corpus and test sets for downstream tasks. 
More specifically, we use two methods to achieve this. 
(1) We use the COCO Image ID of Visual Genome and Flickr30k images.
(2) For COCO, Visual Genome and Flickr30k, we calucate the pair-wise l2 norm between two images after resizing them into the same size.


### Note
It is recommended to download large files with **AzCopy** for faster speed.
AzCopy executable tools can be downloaded [here](https://docs.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10#download-azcopy).
Decompress the tar file and put the executable in any path. To download from
any URL above, the command is:
```bash
path/to/azcopy copy <URL> <local_path>

# for example, downloading coco_caption.zip
path/to/azcopy copy https://biglmdiag.blob.core.windows.net/oscar/datasets/coco_caption.zip <local_path>
```

