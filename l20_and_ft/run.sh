

CUDA_VISIBLE_DEVICES=0,1 python imagenet.py --arch my_resnet_imagenet   --num_epochs 90 --weight_decay 4e-5 --gpus 0 1  --job_dir pr_0.44 --criterion SmoothSoftmax --pr_target 0.44


