# Pruning Networks with Cross-Layer Ranking & k-Reciprocal Nearest Filters

Pytorch implementation of our paper under review -- Pruning Networks with Cross-Layer Ranking & k-Reciprocal Nearest Filters

## Running code

You can run the following code to prune model on CIFAR-10:
```shell
python cifar.py 
--arch vgg_cifar 
--cfg vgg16 
--data_path /data/cifar 
--job_dir ./experiment/cifar/vgg_1 
--pretrain_model /home/xuchi/CLR-RNF/pretrain/vgg16_cifar10.pt 
--lr 0.01 
--lr_decay_step 50 100 
--weight_decay 0.005  
--num_epochs 150 
--gpus 0
--pr_target 0.7 
--graph_gpu
```


 You can run the following code to prune resnets on ImageNet: 

```shell
python imagenet.py 
--dataset imagenet 
--data_path /data/ImageNet/ 
--pretrain_model /data/model/resnet50.pth 
--job_dir /data/experiment/resnet50 
--arch resnet_imagenet 
--cfg resnet50 
--lr 0.1 
--lr_type step
--num_epochs 90 
--train_batch_size 256 
--weight_decay 1e-4 
--gpus 0 1 2 
--pr_target 0.7 
--graph_gpu
```

 You can run the following code to prune mobilenet_v1 on ImageNet: 

```shell
python imagenet.py 
--dataset imagenet 
--arch mobilenet_v1
--cfg mobilenet_v1 
--data_path /media/disk2/zyc/ImageNet2012 
--pretrain_model ./pretrain/checkpoints/mobilenet_v1.pth.tar 
--job_dir ./experiment/imagenet/mobilenet_v1 
--lr 0.1 
--lr_type cos
--weight_decay 4e-5 
--num_epochs 150 
--gpus 0  
--train_batch_size 256 
--eval_batch_size 256 
--pr_target 0.58
--graph_gpu
```


 You can run the following code to prune mobilenet_v2 on ImageNet: 

```shell
python imagenet.py 
--dataset imagenet 
--arch mobilenet_v2 
--cfg mobilenet_v2 
--data_path /media/disk2/zyc/ImageNet2012 
--pretrain_model ./pretrain/checkpoints/mobilenet_v2.pth.tar 
--job_dir ./experiment/imagenet/mobilenet_v2 
--lr 0.1 
--lr_type cos
--weight_decay 4e-5 
--num_epochs 150 
--gpus 0  
--train_batch_size 256 
--eval_batch_size 256 
--pr_target 0.25
--graph_gpu
```

You can run the following code to get FLOPs prune ratio under a given parameters prune target:

```shell
python get_flops.py 
--arch resnet_imagenet 
--cfg resnet50 
--pretrain_model /media/disk2/zyc/prune_result/resnet_50/pruned_checkpoint/resnet50-19c8e357.pth 
--job_dir ./experiment/imagenet/resnet50_flop 
--graph_gpu 
--pr_target 0.1
```

You can run the following code to compare the loss between graph，Kmeans & random: 

```shell
python cal_graph_loss.py 
--arch vgg_cifar 
--cfg vgg16 
--data_path /data/cifar 
--job_dir ./experiment/vgg
--pretrain_model pretrain/vgg16_cifar10.pt 
--gpus 0 
--graph_gpu
```


You can run the following code to test our model:

```shell
python test.py
--arch resnet_imagenet 
--cfg resnet50 
--data_path /media/disk2/zyc/ImageNet2012 
--resume ./pretrain/checkpoints/model_best.pt 
--pretrain_model /media/disk2/zyc/prune_result/resnet_50/pruned_checkpoint/resnet50-19c8e357.pth 
--pr_target 0.44 
--job_dir ./experiment/imagenet/test 
--eval_batch_size 256
```
## CIFAR-10

| Full Model            | Parameter(PR)   | Flops(PR)       |lr_type | lightening | Accuracy | Model             |
| --------------------- | -------------- | ---------------- |--------| -------- | -----------|--------------------|
| VGG-16 (Baseline)     | 14.73M(0.0%)  | 314.04M(0.0%)   |  step  | &#10006; | 93.02%   | [pre-trained](https://drive.google.com/open?id=1sAax46mnA01qK6S_J5jFr19Qnwbl1gpm) |
| VGG-16-0.86           | 0.74M(94.95%) | 81.31M(74.11%)  |  step  | &#10006; | 93.32%   | [pruned](https://drive.google.com/drive/folders/12LkQCfAPXHovR7mTYfOuyIfMuFoxaa4c?usp=sharing) |
| ResNet-56 (Baseline)  | 0.85M(0.0%)   | 126.56M(0.0%)   |  step  | &#10006; | 93.26%   | [pre-trained](https://drive.google.com/open?id=1pt-LgK3kI_4ViXIQWuOP0qmmQa3p2qW5) |
| ResNet-56-0.56        | 0.39M(54.47%)  | 55.26M(56.34%)  |  step  | &#10006; | 93.27%   | [pruned](https://drive.google.com/drive/folders/1Yijljk_-imnrlm8tPPq8UkXAdkSwp4MU?usp=sharing) |
| ResNet-110 (Baseline) | 1.73M(0.0%)   | 254.99M(0.0%)   |  step  | &#10006; | 93.53%   | [pre-trained](https://drive.google.com/open?id=1Uqg8_J-q2hcsmYTAlRtknCSrkXDqYDMD) |
| ResNet-110-0.69       | 0.53M(69.14%) | 86.80M(65.96%)  |  step  | &#10006; | 93.71%   | [pruned](https://drive.google.com/drive/folders/1IrGVxCPBNHsd7LElehaRkHQhc1_Mvi15?usp=sharing) |
| GoogLeNet (Baseline)  | 6.17M(0.0%)   | 1529.43M(0.0%)  |  step  | &#10006; | 95.03%   | [pre-trained](https://drive.google.com/open?id=1YNno621EuTQTVY2cElf8YEue9J4W5BEd) |
| GoogLeNet-0.91        | 2.18M(64.70%) | 491.54M(67.86%) |  step  | &#10006; | 94.85%   | [pruned](https://drive.google.com/drive/folders/1I0k-WBVFoLT0kzN1cROkNudSI3jAY8LG?usp=sharing) |





## ImageNet
| Architecture| Parameter(PR) | Flops(PR)       |lr_type | lightening | Top1-Acc | Top5-Acc | Model                   |
| ---------- | -------------- | ----------------|------- | ----------| -------- | -------- | --------------------------------------- |
| ResNet-50(Baseline) | 25.56M(0.0%) | 4113.56M(0.0%) | step | &#10006; | 76.01% | 92.96% | [pre-trained](https://download.pytorch.org/models/resnet50-19c8e357.pth) |
| ResNet-50-0.52 | 6.90M(72.98%) | 931.02M(77.37%) |step | &#10006;| 71.112% | 90.424% | [pruned](https://drive.google.com/drive/folders/1rTUfyCWWNtSsMNknPw2Ddo1WDzcxY4P8?usp=sharing) |
| ResNet-50-0.44  | 9.00M(64.77%) | 1227.23M(70.17%)|step| &#10006; | 72.656% | 91.085% | [pruned](https://drive.google.com/drive/folders/1ICOf5k3yXEX6dOdZMaBqF4nCEeazrn3D?usp=sharing) |
| ResNet-50-0.2 | 16.92M(33.80%) | 2445.83M(40.54%)|step| &#10006; | 74.851% | 92.305% | [pruned](https://drive.google.com/drive/folders/1XHPCS0SD2MBWdBfSiYYVymXe61gZVqu5?usp=sharing) |
| ResNet-50-0.44  | 9.00M(64.77%)| 1227.23M(70.17%)|cos| &#10006; | 73.344% | 91.271% | [pruned](https://drive.google.com/drive/folders/1qBsgORfkcGj8K90jHlCCg_UFmKFSr255?usp=sharing)|
| ResNet-50-0     |              |                 |   | &#10004; |         |          | [pruned]|


## Other Arguments

```shell
optional arguments:
  -h, --help            show this help message and exit
  --gpus GPUS [GPUS ...]
                        Select gpu_id to use. default:[0]
  --dataset DATASET     Select dataset to train. default:cifar10
  --data_path DATA_PATH
                        The dictionary where the input is stored.
                        default:/home/data/cifar10/
  --job_dir JOB_DIR     The directory where the summaries will be stored.
                        default:./experiments
  --arch ARCH           Architecture of model. default:resnet_imagenet. optional:resnet_cifar/mobilenet_v1/mobilenet_v2
  --cfg CFG             Detail architecuture of model. default:resnet56. optional:resnet110/18/34/50/101/152 mobilenet_v1/mobilenet_v2
  --graph_gpu           Use gpu to graph the filters or not. default:False
  --init_method INIT_METHOD
                        Initital method of pruned model. default:direct_project. optional:random_project
  --pr_target           Target prune ratio of parameters 
  --lr_type             lr scheduler. default: step. optional:exp/cos/step/fixed
  --criterion           Loss function. default:Softmax. optional:SmoothSoftmax
  --graph_method        Method to recontruct the graph of filters. default:knn other:kmeans/random
  --resume              Continue training from specific checkpoint. For example:./experiment/imagenet/resnet50_redidual/checkpoint/model_last.pt
  --use_dali            If this parameter exists, use dali module to load ImageNet data.
```
