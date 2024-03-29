# Adversarial Robustness via Random Projection Filters

## Environment

* torch 1.7.1
* torchvision 0.8.2
* torchattacks 3.2.6

## Training of RPF

* To train a ResNet18 with RPF on CIFAR-10:
```
python train.py --network ResNet18 --dataset cifar10 --attack_iters 10 --lr_schedule multistep --epochs 200 --adv_training --rp --rp_block -1 -1 --rp_out_channel 48 --rp_weight_decay 1e-2 --save_dir resnet18_c10_RPF
```

* To train a ResNet50 with RPF on ImageNet:
```
python train_imagenet.py --pretrained --lr 0.02 --lr_schedule cosine --batch_size 1024 --epochs 90 --adv_train --rp --rp_block -1 -1 --rp_out_channel 48 --rp_weight_decay 1e-2 --save_dir resnet50_imagenet_RPF
```


## Evaluation of RPF

* To evaluate the performance of ResNet18 with RPF on CIFAR-10:

```
python evaluate.py --dataset cifar10 --network ResNet18 --rp --rp_out_channel 48 --rp_block -1 -1 --save_dir eval_r18_c10 --pretrain [path_to_model]
```

* To evaluate the performance of ResNet50 with RPF on ImageNet:

```
python train_imagenet.py --evaluate --rp --rp_out_channel 48 --save_dir eval_r50_imagenet --eval_model_path [path_to_model]
```

## Pretrained Models
Pretrained models are provided in [google-drive](https://drive.google.com/drive/folders/1-MbjFfUo-RjGe9_i1xlqQKHSkV0lABTC?usp=sharing).