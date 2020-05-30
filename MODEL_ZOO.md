## Table of Contents
- <a href='#VQA'>VQA</a>
- <a href='#GQA'>GQA</a>
- <a href='#NLVR2'>NLVR2</a>
- <a href='#Image-Text-Retrieval'>Image/Text Retrieval</a>
- <a href='#Image-Captioning-on-COCO'>Image Captioning on COCO</a>


## Performance
Task    | t2i | t2i | i2t | i2t | IC  | IC  |  IC  |  IC  | NoCaps | NoCaps |   VQA    |  NLVR2  |
--------|-----|-----|-----|-----|-----|-----|------|------|--------|--------|----------|---------|
Metric  | R@1 | R@5 | R@1 | R@5 | B@4 |  M  |  C   |   S  |    C   |    S   | test-std | test-P  |
SoTA_S  |39.2 | 68.0|56.6 | 84.5|38.9 |29.2 |129.8 | 22.4 |   61.5 |  9.2   |  70.90   | 53.50   |
SoTA_B  |48.4 | 76.7|63.3 | 87.0|39.5 |29.3 |129.3 | 23.2 |   73.1 | 11.2   |  72.54   | 78.87   |
SoTA_L  |51.7 | 78.4|66.6 | 89.4|  -  |  -  |   -  |   -  |     -  |   -    |  73.40   | 79.50   |
-----   |---  |---  |---  |---  |---  |---  |---   |---   |---     |---     |---       |---      |
Oscar_B |54.0 | 80.8|70.0 | 91.1|40.5 |29.7 |137.6 | 22.8 |   78.8 | 11.7   |  73.44   | 78.44   |
Oscar_L |57.5 | 82.8|73.5 | 92.2|41.7 |30.6 |140.0 | 24.5 |   80.9 | 11.3   |  73.82   | 80.37   |
gain    | 5.8 |  4.4| 6.9 |  2.8| 2.2 | 1.3 | 10.7 | 1.3  |    7.8 |  0.5   |   0.42   |  0.87   |

t2i: text-to-image retrieval; i2t: image-to-text retrieval; IC: image captioning on COCO. 

For reference, we also release the training logs and output.


## VQA
Script to finetune for Oscar base model.
Base model is trained on train split and evaluated on the val split. Good for later comparison.

Training logs: [eval_logs.json](https://biglmdiag.blob.core.windows.net/oscar/exp/vqa/base/base_9m_ep107_1192k_eu1/application_1575931286052_40649/results/eval_logs.json), [output.txt](https://biglmdiag.blob.core.windows.net/oscar/exp/vqa/base/base_9m_ep107_1192k_eu1/application_1575931286052_40649/results/stdout.txt).<br />
Final server results: [results.txt](https://biglmdiag.blob.core.windows.net/oscar/exp/vqa/base/results.txt).<br />
Model checkpoint: [.zip](https://biglmdiag.blob.core.windows.net/oscar/exp/vqa/base/vqa_base_best.zip).
```bash
python oscar/run_vqa.py -j 4 --img_feature_dim 2054 --max_img_seq_length
    50 --data_label_type mask --img_feature_type faster_r-cnn --data_dir datasets/vqa/2k
    --model_type bert --model_name_or_path pretrained_models/base-vg-labels/ep_107_1192087
    --task_name vqa_text --do_train --do_lower_case --max_seq_length 128 --per_gpu_eval_batch_size
    256 --per_gpu_train_batch_size 32 --learning_rate 5e-05 --num_train_epochs 25
    --output_dir results --label_file datasets/vqa/cache/trainval_ans2label.pkl
    --save_epoch 1 --seed 88 --evaluate_during_training --logging_steps 4000 --drop_out
    0.3 --weight_decay 0.05 --warmup_steps 0 --loss_type bce --img_feat_format pt 
    --classifier linear --cls_hidden_scale 3 --txt_data_dir datasets/vqa/2k
```

Script to finetune for Oscar large model.
Large model is trained on train+val split and evaluated on the val split, for reproduce the paper's best result.

Training logs: [eval_logs.json](https://biglmdiag.blob.core.windows.net/oscar/exp/vqa/large/ab128_img_large_rr1_ep20_590k_tv_done_good/exp_ab128_img_large_rr1_ep20_590k_tv_0.00003_128_50_dp_0.3_wd_0.05_bce_3linear_s88_abcd/results/eval_logs.json), [output.txt](https://biglmdiag.blob.core.windows.net/oscar/exp/vqa/large/ab128_img_large_rr1_ep20_590k_tv_done_good/exp_ab128_img_large_rr1_ep20_590k_tv_0.00003_128_50_dp_0.3_wd_0.05_bce_3linear_s88_abcd/stdout.txt).<br />
Final server results: [results.txt](https://biglmdiag.blob.core.windows.net/oscar/exp/vqa/large/results.txt).<br />
Model checkpoint: [.zip](https://biglmdiag.blob.core.windows.net/oscar/exp/vqa/large/vqa_large_best.zip).
```bash
python oscar/run_vqa.py -j 4 --img_feature_dim 2054 --max_img_seq_length
    50 --data_label_type mask --img_feature_type faster_r-cnn --data_dir datasets/vqa/2k
    --model_type bert --model_name_or_path pretrained_models/large-vg-labels/ep_20_590000
    --task_name vqa_text --do_train_val --do_lower_case --max_seq_length 128 --per_gpu_eval_batch_size
    256 --per_gpu_train_batch_size 24 --learning_rate 3e-05 --num_train_epochs 25
    --label_file datasets/vqa/cache/trainval_ans2label.pkl --save_epoch 30
    --seed 88 --evaluate_during_training --logging_steps 4000 --drop_out 0.3 --weight_decay
    0.05 --warmup_steps 0 --loss_type bce --save_after_epoch 15 --output_dir results --img_feat_format pt --classifier linear --cls_hidden_scale 3 --txt_data_dir datasets/vqa/2k
```


## GQA
Script to finetune for Oscar base model.

Training logs: [eval_logs.json](https://biglmdiag.blob.core.windows.net/oscar/exp/gqa/base/ab175_base_ep107_1192k_0.4true_taeb_done_25eps_good/exp_ab175_base_ep107_1192k_0.4true_taeb_b_48_0.00005_165_45_dp_0.3_abce/results/eval_logs.json), [output.txt](https://biglmdiag.blob.core.windows.net/oscar/exp/gqa/base/ab175_base_ep107_1192k_0.4true_taeb_done_25eps_good/exp_ab175_base_ep107_1192k_0.4true_taeb_b_48_0.00005_165_45_dp_0.3_abce/stdout.txt).<br />
Final server results: [results.txt](https://biglmdiag.blob.core.windows.net/oscar/exp/gqa/base/ab165_img45_1568928610179_62515_test_done_good/results.txt).<br />
Model checkpoint: [.zip](https://biglmdiag.blob.core.windows.net/oscar/exp/gqa/base/gqa_base_best.zip).
```bash
python oscar/run_gqa.py -j 4 --img_feature_dim 2054 --max_img_seq_length
    45 --data_dir datasets/GQA/0.4true --model_type bert --model_name_or_path pretrained_models/base-vg-labels/ep_107_1192087
    --task_name gqa --do_lower_case --max_seq_length 165 --per_gpu_eval_batch_size
    256 --per_gpu_train_batch_size 48 --learning_rate 5e-05 --num_train_epochs 5 --output_dir
    results --label_file datasets/GQA/questions1.2/trainval_testdev_all_ans2label.pkl
    --img_feature_type faster_r-cnn --data_label_type all --train_data_type all --eval_data_type
    bal --label2ans_file datasets/GQA/questions1.2/trainval_testdev_all_label2ans.pkl
    --loss_type xe --save_epoch 2 --seed 88 --evaluate_during_training --logging_steps
    4000 --drop_out 0.3 --do_train --weight_decay 0.05 --warmup_steps 0
```

## NLVR2
Script to finetune for Oscar base model.

Training logs: [eval_logs.json](https://biglmdiag.blob.core.windows.net/oscar/exp/nlvr2/base/exp_rvln_base_ep107_1192k_wm1w_b72_0.00003_55_40_dp0.3_3mlp_wm10000_abcf_best/results/eval_logs.json), [output.txt](https://biglmdiag.blob.core.windows.net/oscar/exp/nlvr2/base/exp_rvln_base_ep107_1192k_wm1w_b72_0.00003_55_40_dp0.3_3mlp_wm10000_abcf_best/stdout.txt).<br />
Final server results: [results.txt](https://biglmdiag.blob.core.windows.net/oscar/exp/nlvr2/base/exp_nlvr_base_11123_testall_b24_0.00003_55_43_dp_0.3_mlp_abcj_best/stdout.txt).
```bash
python oscar/run_nlvr.py -j 4 --img_feature_dim 2054 --max_img_seq_length
    40 --data_dir datasets/nlvr2/ft_corpus --model_type bert --model_name_or_path pretrained_models/base-vg-labels/ep_107_1192087
    --task_name nlvr --do_lower_case --max_seq_length 55 --per_gpu_eval_batch_size
    64 --per_gpu_train_batch_size 72 --learning_rate 3e-05 --num_train_epochs 20 --output_dir
    results --img_feature_type faster_r-cnn --data_label_type all --train_data_type
    all --eval_data_type all --loss_type xe --save_epoch -1 --seed 88 --evaluate_during_training
    --logging_steps -1 --drop_out 0.3 --do_train --weight_decay 0.05 --warmup_steps
    10000 --classifier mlp --cls_hidden_scale 3 --num_choice 2 --use_pair
```

Script to finetune for Oscar large model.

Training logs: [eval_logs.json](https://biglmdiag.blob.core.windows.net/oscar/exp/nlvr2/large/large_1583307153868_14140/exp_rvln_large_ep55_1618k_b24_0.00002_seq55_img40_dp0.3_2mlp_wm5000_abcj/results/eval_logs.json), [output.txt](https://biglmdiag.blob.core.windows.net/oscar/exp/nlvr2/large/large_1583307153868_14140/exp_rvln_large_ep55_1618k_b24_0.00002_seq55_img40_dp0.3_2mlp_wm5000_abcj/stdout.txt).<br />
Final server results: [results.txt](https://biglmdiag.blob.core.windows.net/oscar/exp/nlvr2/large/large_1583307153868_14140/exp_nlvr_large_1583307153868_14140_testall_b24_0.00003_55_43_dp_0.3_mlp_abck/stdout.txt).
```bash
python oscar/run_nlvr.py -j 4 --img_feature_dim 2054 --max_img_seq_length
    40 --data_dir datasets/nlvr2/ft_corpus --model_type bert --model_name_or_path pretrained_models/large-vg-labels/ep_55_1617000
    --task_name nlvr --do_lower_case --max_seq_length 55 --per_gpu_eval_batch_size
    64 --per_gpu_train_batch_size 24 --learning_rate 3e-05 --num_train_epochs 20 --output_dir
    results --img_feature_type faster_r-cnn --data_label_type all --train_data_type
    all --eval_data_type all --loss_type xe --save_epoch -1 --seed 88 --evaluate_during_training
    --logging_steps -1 --drop_out 0.3 --do_train --weight_decay 0.05 --warmup_steps
    5000 --classifier mlp --cls_hidden_scale 2 --num_choice 2 --use_pair
```

<!---
Training logs: [eval_logs.json](https://biglmdiag.blob.core.windows.net/oscar/exp/nlvr2/large/large_1583307153868_14140/exp_rvln_large_ep55_1618k_b24_0.00002_seq55_img40_dp0.3_2mlp_wm5000_abcj/results/eval_logs.json), [output.txt](https://biglmdiag.blob.core.windows.net/oscar/exp/nlvr2/large/large_1583307153868_14140/exp_rvln_large_ep55_1618k_b24_0.00002_seq55_img40_dp0.3_2mlp_wm5000_abcj/stdout.txt).<br />
Final server results: [results.txt](https://biglmdiag.blob.core.windows.net/oscar/exp/nlvr2/large/large_1583307153868_14140/exp_nlvr_large_1583307153868_14140_testall_b24_0.00003_55_43_dp_0.3_mlp_abck/stdout.txt).
```bash
python oscar/run_nlvr.py -j 4 --img_feature_dim 2054 --max_img_seq_length
    40 --data_dir datasets/nlvr2/ft_corpus --model_type bert --model_name_or_path pretrained_models/base-vg-labels/ep_55_1617000
    --task_name nlvr --do_lower_case --max_seq_length 55 --per_gpu_eval_batch_size
    64 --per_gpu_train_batch_size 24 --learning_rate 3e-05 --num_train_epochs 20 --output_dir
    results --img_feature_type faster_r-cnn --data_label_type all --train_data_type
    all --eval_data_type all --loss_type xe --save_epoch -1 --seed 88 --evaluate_during_training
    --logging_steps -1 --drop_out 0.3 --do_train --weight_decay 0.05 --warmup_steps
    5000 --classifier mlp --cls_hidden_scale 2 --num_choice 2 --use_pair
```
--->

## Image Text Retrieval
Script to finetune for Oscar base model (4 V100 with 16G mem):

Training logs: [eval_logs.json](https://biglmdiag.blob.core.windows.net/oscar/exp/retrieval/base/eval_logs.json), [log.txt](https://biglmdiag.blob.core.windows.net/oscar/exp/retrieval/base/log.txt).
Model checkpoint: [checkpoint.zip](https://biglmdiag.blob.core.windows.net/oscar/exp/retrieval/base/checkpoint.zip).

```bash
python oscar/run_retrieval.py \
    --model_name_or_path pretrained_models/base-vg-labels/ep_67_588997 \
    --do_train \
    --do_lower_case \
    --evaluate_during_training \
    --num_captions_per_img_val 20 \
    --eval_caption_index_file minival_caption_indexs_top20.pt \
    --per_gpu_train_batch_size 32 \
    --learning_rate 0.00002 \
    --num_train_epochs 30 \
    --weight_decay 0.05 \
    --save_steps 5000 \
    --add_od_labels \
    --od_label_type vg \
    --max_seq_length 70 \
    --output_dir output/
```

Script to finetune for Oscar large model (8 V100 with 32G mem):

Training logs: [eval_logs.json](https://biglmdiag.blob.core.windows.net/oscar/exp/retrieval/large/eval_logs.json), [log.txt](https://biglmdiag.blob.core.windows.net/oscar/exp/retrieval/large/log.txt).
Model checkpoint: [checkpoint.zip](https://biglmdiag.blob.core.windows.net/oscar/exp/retrieval/large/checkpoint.zip).

```bash
python oscar/run_retrieval.py \
    --model_name_or_path pretrained_models/large-vg-labels/ep_7_816000 \
    --do_train \
    --do_lower_case \
    --evaluate_during_training \
    --num_captions_per_img_val 20 \
    --eval_caption_index_file minival_caption_indexs_top20.pt \
    --per_gpu_train_batch_size 16 \
    --learning_rate 0.00001 \
    --num_train_epochs 30 \
    --save_steps 5000 \
    --add_od_labels \
    --od_label_type vg \
    --max_seq_length 70 \
    --output_dir output/
```

Script to inference on COCO 1K test set:
```bash
python oscar/run_retrieval.py \
    --do_test \
    --do_eval \
    --test_split test \
    --num_captions_per_img_val 5 \
    --eval_img_keys_file test_img_keys_1k.tsv \
    --cross_image_eval \
    --per_gpu_eval_batch_size 64 \
    --eval_model_dir your_model_for_evaluation # could be base/large models.
```

Script to inference on COCO 5K test set:
```bash
python oscar/run_retrieval.py \
    --do_test \
    --do_eval \
    --test_split test \
    --num_captions_per_img_val 5 \
    --eval_img_keys_file test_img_keys.tsv \
    --cross_image_eval \
    --per_gpu_eval_batch_size 64 \
    --eval_model_dir your_model_for_evaluation # could be base/large models.
```


## Image Captioning on COCO
Script to finetune for Oscar base model (4 V100 with 16G mem):

Training logs: [log.txt](https://biglmdiag.blob.core.windows.net/oscar/exp/coco_caption/base/log.txt).
Model checkpoint: [checkpoint.zip](https://biglmdiag.blob.core.windows.net/oscar/exp/coco_caption/base/checkpoint.zip).

1) First train with cross-entropy loss:
```bash
python oscar/run_captioning.py \
    --model_name_or_path pretrained_models/base-vg-labels/ep_67_588997 \
    --do_train \
    --do_lower_case \
    --evaluate_during_training \
    --add_od_labels \
    --learning_rate 0.00003 \
    --per_gpu_train_batch_size 64 \
    --num_train_epochs 30 \
    --save_steps 5000 \
    --output_dir output/
```
2) Finetune with CIDEr optimization:
```bash
python oscar/run_captioning.py \
    --model_name_or_path your_checkpoint_from_cross_entropy \
    --do_train \
    --do_lower_case \
    --evaluate_during_training \
    --add_od_labels \
    --learning_rate 0.000005 \
    --per_gpu_train_batch_size 16 \
    --num_train_epochs 5 \
    --scst \
    --save_steps 2000 \
    --output_dir output/
```

Script to finetune for Oscar large model (8 V100 with 32G mem):
1) First train with cross-entropy loss:
```bash
python oscar/run_captioning.py \
    --model_name_or_path pretrained_models/large-vg-labels/ep_7_816000 \
    --do_train \
    --do_lower_case \
    --evaluate_during_training \
    --add_od_labels \
    --learning_rate 0.00001 \
    --per_gpu_train_batch_size 32 \
    --num_train_epochs 30 \
    --save_steps 5000 \
    --output_dir output/
```
2) Finetune with CIDEr optimization:
```bash
python oscar/run_captioning.py \
    --model_name_or_path your_checkpoint_from_cross_entropy \
    --do_train \
    --do_lower_case \
    --evaluate_during_training \
    --add_od_labels \
    --learning_rate 0.000005 \
    --per_gpu_train_batch_size 8 \
    --num_train_epochs 5 \
    --scst \
    --save_steps 2000 \
    --output_dir output/
```

Script to inference on COCO test set:
```bash
python oscar/run_captioning.py \
    --do_test \
    --do_eval \
    --test_yaml test.yaml \
    --per_gpu_eval_batch_size 64 \
    --num_beams 5 \
    --max_gen_length 20 \
    --eval_model_dir your_model_for_evaluation # could be bert base/large.
```
