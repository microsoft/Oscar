# Download

Note:  The data is on Azure Storage Blob, a SAS with Read permission is provided. Please append the following SAS at the end of each link to download: 
```bash
?sp=r&st=2023-08-28T01:12:41Z&se=3023-08-28T09:12:41Z&sv=2022-11-02&sr=c&sig=6R1YmWluiXmPLsdVn1rDUpeBp2SYBMxDjc6KoKNlY8Q%3D
```

## Datasets
We provide the extracted image region features, object tags, and the original text annotations for each downstream tasks.
```bash
wget https://biglmdiag.blob.core.windows.net/oscar/datasets/$TASK_NAME.zip
unzip $TASK_NAME.zip -d $DATA_DIR
```
`TASK_NAME` could be `coco_caption`, `coco_ir`, `vqa`, `GQA`, `nlvr2`.

## Pre-trained Models
We provide pre-trained *Oscar* models of Bert-base and Bert-large structures, with the name starting with `base` and `large`, respectively.
```bash
wget https://biglmdiag.blob.core.windows.net/oscar/pretrained_models/$MODEL_NAME.zip
unzip $MODEL_NAME.zip -d $MODEL_DIR
```
`MODEL_NAME` could be `base-vg-labels`, `large-vg-labels`, `base-oid-labels`, `base-no-labels`.

The models are trained with both image region features and object tags. The image region features are extracted by the Faster R-CNN with
ResNet-101, using object and attribute annotations from [Visual Genome](http://visualgenome.org/).
The object tags are from:
    1) the same VisualGenome model, named as `-vg-labels`. Or,
    2) the model trained on object annotations from [Open Images V5](https://storage.googleapis.com/openimages/web/index.html). named as `-oid-labels`. Or,
    3) no object tags provied, serving as baseline, named as `-no-labels`.


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

