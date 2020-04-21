#!/bin/bash

python3 /world/jiacongliao/place365/scene_classification/evaluate.py \
--gpus=0,1,2,3,4,5,6,7 \
--input_size=256 \
--batch_size=256 \
--pretrained_model=/world/jiacongliao/model/places365/seresnet50_dropblock/dropblock0.7_group/seresnet50_ckpt_epoch30.pth \
--save_path=/world/jiacongliao/model/places365/seresnet50_dropblock/dropblock0.7_group/ \
--logger=serestnet50dropblock0.7_group.log \
--val_list=/world/jiacongliao/place365/filelist_places365-standard/places365_val.txt \
--val_rootdir=/world/jiacongliao/place365/val_dataset/val_large/ \
--num_classes=365 \
--convert2fcn=true \
--network=seresnet.resnet50_fcn

echo complete


# python /world/jiacongliao/place365/scene_classification/network/eval_pytorch_resnet.py \
# --gpus=8,9 \
# --batch_size=16 \
# --input_size=256 \
# --pretrained_model=/world/jiacongliao/model/places365/public/Places2-CNNs/Places2-365-CNN/Places365_resnet.pth \
# --save_path=/world/jiacongliao/model/places365/public/Places2-CNNs/Places2-365-CNN/ \
# --logger=places365_resnet2.log \
# --val_list=/world/jiacongliao/place365/filelist_places365-standard/places365_val.txt \
# --val_rootdir=/world/jiacongliao/place365/val_dataset/val_large/ \
# --convert2fcn=true

# echo complete