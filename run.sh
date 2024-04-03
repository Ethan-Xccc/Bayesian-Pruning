vgg16(){
python cifar_copy.py \
--arch vgg_cifar \
--cfg vgg16 \
--data_path /data/cifar \
--job_dir ./experiment/cifar/vgg_1 \
--pretrain_model /data/pretrain/vgg16_cifar10.pt \
--lr 0.01 \
--lr_decay_step 50 100 \
--weight_decay 0.005  \
--num_epochs 150 \
--gpus 0 \
--pr_target 0.86 \
--graph_gpu
}

resnet56(){
python cifar.py \
--arch resnet_cifar \
--cfg resnet56 \
--dataset cifar10 \
--data_path /data/cifar \
--job_dir ./experiment/cifar/resnet_1 \
--pretrain_model /home/xuchi/CLR-RNF/pretrain/resnet_56.pt \
--lr 0.01 \
--lr_decay_step 50 100 \
--weight_decay 0.005  \
--num_epochs 150 \
--gpus 0 \
--pr_target 0.56 \
--graph_gpu
}

vgg16
