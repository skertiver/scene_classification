#!/bin/bash

# export PYTHONPATH=/world/jiacongliao/place365/scene_classification:${PYTHONPATH}

python3 /world/jiacongliao/place365/scene_classification/train.py \
--gpus=0,1,2,3,4,5,6,7 \
--input_size=224 --crop_size=224 \
--learning_rate=0.1 --weight_decay=5e-4 --lr_decay_step=10 --min_lr=1e-4 \
--epochs=50 --batch_size=512 \
--pretrained_model= \
--save_path=/world/jiacongliao/model/places365/se_resnext50/se_resnext50_aug/ \
--logger=se_resnext50_aug.log \
--train_list=/world/jiacongliao/place365/places365_train_standard.lst1 \
--train_rootdir=/world/train_places365standard_1_2016-12-14/data_256/data_256 \
--val_list=/world/jiacongliao/place365/filelist_places365-standard/places365_val.txt \
--val_rootdir=/world/jiacongliao/place365/val_dataset/val_large/ \
--num_classes=365 \
--network=resnext.se_resnext50 \
--is_use_memcache=False \
--is_Scheduler_dropblock=False \
--is_cutout=True

echo complete