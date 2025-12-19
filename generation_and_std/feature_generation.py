import os
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import torchvision
from torchvision import datasets, transforms

# 保持原有的模型import不变
from models.cifar10.vgg import vgg_16_bn
from models.cifar10.vgg_cifar import vgg16_cifar
from models.cifar10.resnet import resnet_56,resnet_110
from models.cifar10.resnet_cifar import resnet56, resnet110
from models.cifar100.resnet import resnet_56_cifar100, resnet_110_cifar100
from models.imagenet.resnet import resnet_50

from data import imagenet
import utils.common as utils

parser = argparse.ArgumentParser(description='Rank extraction')

parser.add_argument(
    '--data_dir',
    type=str,
    default='cifar10',
    help='dataset path')
parser.add_argument(
    '--dataset',
    type=str,
    default='cifar10',
    choices=('cifar10','cifar100','imagenet'),
    help='dataset')
parser.add_argument(
    '--job_dir',
    type=str,
    default='result/tmp',
    help='The directory where the summaries will be stored.')
parser.add_argument(
    '--arch',
    type=str,
    default='vgg_16_bn',
    choices=('resnet_50','vgg_16_bn','vgg_16_bn_cifar100','resnet_56','resnet_56_cifar100','resnet_110','resnet_110_cifar100'),
    help='The architecture to prune')
parser.add_argument(
    '--pretrain_dir',
    type=str,
    default='vgg_16_bn.pt',
    help='load the model from the specified checkpoint')
parser.add_argument(
    '--limit',
    type=int,
    default=20, # 修改默认值为20，符合你的描述
    help='The num of batch to get rank.')
parser.add_argument(
    '--batch_size',
    type=int,
    default=128,
    help='Batch size for training.')
parser.add_argument(
    '--gpu',
    type=str,
    default='0,1',
    help='Select gpu to use')
parser.add_argument(
    '--rank_conv_dir',
    type=str,
    default='apoz',
    help='rank_conv_dir')
args = parser.parse_args()
args.data_dir = args.data_dir 
args.pretrain_dir = args.pretrain_dir

cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data
print('==> Preparing data..')
if args.dataset=='cifar10':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
elif args.dataset=='cifar100':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
    ])
    
    trainset = torchvision.datasets.CIFAR100(root=args.data_dir, train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, shuffle=True, batch_size=args.batch_size,num_workers=4)
elif args.dataset=='imagenet':
    data_tmp = imagenet.Data(args)
    train_loader = data_tmp.train_loader
    val_loader = data_tmp.test_loader

rank_conv_dir = args.rank_conv_dir
if not os.path.isdir(rank_conv_dir):
    os.mkdir(rank_conv_dir)
# Model
print('==> Building model..')

if args.arch =='vgg16_cifar':
    net = eval(args.arch)()
elif args.arch =='resnet56_cifar':
    net = resnet56()
elif args.arch =='resnet110_cifar':
    net = resnet110()
else:
    net = eval(args.arch)(compress_rate=[0.]*100)
net = net.to(device)
print(net)

if len(args.gpu)>1 and torch.cuda.is_available():
    device_id = []
    for i in range((len(args.gpu) + 1) // 2):
        device_id.append(i)
    net = torch.nn.DataParallel(net, device_ids=device_id)

if args.pretrain_dir:
    print(args.pretrain_dir)
    print('==> Resuming from checkpoint..')
    if args.arch=='vgg_16_bn' or args.arch=='resnet_56' or args.arch=='resnet110_cifar':
        checkpoint = torch.load(args.pretrain_dir, map_location='cuda:'+args.gpu if torch.cuda.is_available() else 'cpu')
    else:
        checkpoint = torch.load(args.pretrain_dir)
    if args.arch=='resnet_50':
        net.load_state_dict(checkpoint)
    else:        
        net.load_state_dict(checkpoint['state_dict'])
else:
    print('please speicify a pretrain model ')
    raise NotImplementedError

criterion = nn.CrossEntropyLoss()

# ==========================================
# 修改部分：全局变量用于存储采样数据
# ==========================================
sampled_data = [] # 替换原来的 feature_result 和 total

def get_feature_hook(self, input, output):    
    global sampled_data
    # output shape: [N, C, H, W]
    N, C, H, W = output.shape
    num_samples = 10 # 抽样点数

    # 随机生成10个空间位置坐标 (h, w)
    # 这些坐标在同一个Batch内的所有图片和所有通道上是共享的（或者随机性取决于是否在循环内生成）
    # 这里我们为当前batch生成一组随机坐标
    idx_h = torch.randint(0, H, (num_samples,), device=output.device)
    idx_w = torch.randint(0, W, (num_samples,), device=output.device)

    # 进行采样
    # output[:, :, idx_h, idx_w] 的结果形状为 [N, C, 10]
    # 含义：取所有图片(N)，所有通道(C)，在idx_h和idx_w指定位置的值
    sampled_val = output[:, :, idx_h, idx_w]

    # 现在的形状是 [N, C, 10]
    # 我们需要将其转换为 [N, 10, C]，这样当我们展平前两个维度时，就变成了 [N*10, C]
    # 这样每一行代表一个采样点，每一列代表一个通道
    sampled_val = sampled_val.permute(0, 2, 1).contiguous() # [N, 10, C]
    
    # 展平前两维 -> [N * 10, C]
    sampled_val = sampled_val.view(-1, C)

    # 转移到CPU并存入列表，防止显存爆炸
    sampled_data.append(sampled_val.cpu())


def inference():
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    limit = args.limit

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            if batch_idx >= limit:
               break

            inputs, targets = inputs.to(device), targets.to(device)

            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            utils.progress_bar(batch_idx, limit, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

# ==========================================
# 主逻辑修改：保存 sampled_data 而不是 mean
# ==========================================

if args.arch=='vgg_16_bn' or args.arch=='vgg16_cifar':

    if len(args.gpu) > 1:
        relucfg = net.module.relucfg
    else:
        relucfg = net.relucfg

    for i, cov_id in enumerate(relucfg):
        cov_layer = net.features[cov_id]
        
        if not isinstance(cov_layer, nn.ReLU):
            cov_layer = net.features[cov_id-1]
        print(f"Processing layer {i+1}: {cov_layer}")
        
        sampled_data = [] # 重置列表
        handler = cov_layer.register_forward_hook(get_feature_hook)
        inference()
        handler.remove()

        if not os.path.isdir(rank_conv_dir+'/'+args.arch+'_limit%d'%(args.limit)):
            os.mkdir(rank_conv_dir+'/'+args.arch+'_limit%d'%(args.limit))
        
        # 合并所有batch的数据
        # 形状: [(Batch * Limit * 10), Channels]
        # 例如: 128 * 20 * 10 = 25600 行
        final_features = torch.cat(sampled_data, dim=0).numpy()

        save_path = rank_conv_dir+'/'+args.arch+'_limit%d'%(args.limit)+'/apoz_rank_conv' + str(i + 1) + '.npy'
        np.save(save_path, final_features)
        print(f'Saved shape: {final_features.shape} to {save_path}')
        
        sampled_data = [] # 清空内存

elif args.arch=='resnet_56' or args.arch=='resnet56_cifar' or args.arch=='resnet_56_cifar100':

    cov_layer = eval('net.relu')
    sampled_data = [] # 重置
    handler = cov_layer.register_forward_hook(get_feature_hook)
    inference()
    handler.remove()

    if not os.path.isdir(rank_conv_dir+'/' + args.arch+'_limit%d'%(args.limit)):
        os.mkdir(rank_conv_dir+'/' + args.arch+'_limit%d'%(args.limit))
    
    final_features = torch.cat(sampled_data, dim=0).numpy()
    np.save(rank_conv_dir+'/' + args.arch+'_limit%d'%(args.limit)+ '/apoz_rank_conv%d' % (1) + '.npy', final_features)
    print(f'Saved shape: {final_features.shape}')
    sampled_data = []

    # ResNet56 per block
    cnt=0
    for i in range(3):
        block = eval('net.layer%d' % (i + 1))
        for j in range(9):
            cov_layer = block[j].relu1
            sampled_data = []
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            
            final_features = torch.cat(sampled_data, dim=0).numpy()
            np.save(rank_conv_dir+'/' + args.arch +'_limit%d'%(args.limit)+ '/apoz_rank_block%d'%(cnt + 1)+'_relu1.npy', final_features)
            print(f'Block {cnt+1} Relu1 Saved shape: {final_features.shape}')
            sampled_data = []
            
            cov_layer = block[j].relu2
            sampled_data = []
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            
            final_features = torch.cat(sampled_data, dim=0).numpy()
            np.save(rank_conv_dir+'/'+ args.arch +'_limit%d'%(args.limit)+ '/apoz_rank_block%d'%(cnt + 1)+'_relu2.npy', final_features)
            print(f'Block {cnt+1} Relu2 Saved shape: {final_features.shape}')
            cnt += 1
            sampled_data = []

elif args.arch=='resnet_110' or args.arch=='resnet110_cifar':

    cov_layer = eval('net.relu')
    sampled_data = []
    handler = cov_layer.register_forward_hook(get_feature_hook)
    inference()
    handler.remove()

    if not os.path.isdir(rank_conv_dir+'/' + args.arch+'_limit%d'%(args.limit)):
        os.mkdir(rank_conv_dir+'/' + args.arch+'_limit%d'%(args.limit))
    
    final_features = torch.cat(sampled_data, dim=0).numpy()
    np.save(rank_conv_dir+'/' + args.arch+'_limit%d'%(args.limit) + '/new_rank_conv%d' % (1) + '.npy', final_features)
    print(f'Saved shape: {final_features.shape}')
    sampled_data = []

    cnt = 1
    # ResNet110 per block
    for i in range(3):
        block = eval('net.layer%d' % (i + 1))
        for j in range(18):
            cov_layer = block[j].relu1
            sampled_data = []
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()

            final_features = torch.cat(sampled_data, dim=0).numpy()
            np.save(rank_conv_dir+'/' + args.arch  + '_limit%d' % (args.limit) + '/new_rank_conv%d'%(cnt + 1)+'.npy', final_features)
            sampled_data = []

            cov_layer = block[j].relu2
            sampled_data = []
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            
            final_features = torch.cat(sampled_data, dim=0).numpy()
            np.save(rank_conv_dir+'/' + args.arch  + '_limit%d' % (args.limit) + '/new_rank_conv%d'%(cnt + 1)+'.npy', final_features)
            
            cnt += 1
            sampled_data = []

elif args.arch=='resnet_50':
    cov_layer = eval('net.maxpool')
    sampled_data = []
    handler = cov_layer.register_forward_hook(get_feature_hook)
    inference()
    handler.remove()

    if not os.path.isdir(rank_conv_dir+'/' + args.arch+'_limit%d'%(args.limit)):
        os.mkdir(rank_conv_dir+'/' + args.arch+'_limit%d'%(args.limit))
    
    final_features = torch.cat(sampled_data, dim=0).numpy()
    np.save(rank_conv_dir+'/' + args.arch+'_limit%d'%(args.limit) + '/apoz_rank_conv_%d' % (1) + '.npy', final_features)
    print(f'Saved shape: {final_features.shape}')
    sampled_data = []

    # ResNet50 per bottleneck
    cnt=1
    for i in range(4):
        block = eval('net.layer%d' % (i + 1))
        print('=====')
        for j in range(net.num_blocks[i]):
            cov_layer = block[j].relu1
            sampled_data = []
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            
            final_features = torch.cat(sampled_data, dim=0).numpy()
            np.save(rank_conv_dir+'/' + args.arch+'_limit%d'%(args.limit) + '/apoz_rank_conv_%d'%(cnt+1)+'.npy',
                    final_features)
            cnt+=1
            sampled_data = []

            cov_layer = block[j].relu2
            sampled_data = []
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            
            final_features = torch.cat(sampled_data, dim=0).numpy()
            np.save(rank_conv_dir+'/' + args.arch + '_limit%d' % (args.limit) + '/apoz_rank_conv_%d' % (cnt + 1) + '.npy',
                    final_features)
            cnt += 1
            sampled_data = []

            cov_layer = block[j].relu3
            sampled_data = []
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            if j==0:
                print(cnt +1)
                final_features = torch.cat(sampled_data, dim=0).numpy()
                print(final_features.shape)
                np.save(rank_conv_dir+'/' + args.arch + '_limit%d' % (args.limit) + '/apoz_rank_conv_%d' % (cnt + 1) + '.npy',
                        final_features) # shortcut conv
                cnt += 1
                sampled_data = [] # 这里需要单独reset，因为shortcut保存了一次
                
                # 注意：这里逻辑有点特殊，relu3的hook已经跑完了，数据在sampled_data里。
                # 如果shortcut保存消耗了sampled_data，那么relu3的数据需要重新inference吗？
                # 原代码逻辑是同一个hook跑一次 inference，但是在j==0时保存了两次（一次shortcut，一次conv3）。
                # 但原代码因为是累加器，所以其实是同一个累加结果保存了两次（假设shortcut和relu3共享了某种特性？或者原代码这里逻辑只是为了占位）。
                # **更正**：原代码在 `j==0` 时保存了shortcut，但没有重新跑inference。
                # 由于hook是挂在 relu3 上的，所以这里的 `sampled_data` 实际上是 relu3 的输出。
                # 如果你想获取 shortcut 的特征，应该挂在 shortcut 层上。但既然遵循“原样保存位置”，
                # 我这里就把 relu3 的采样结果保存两份（一份给shortcut文件名，一份给relu3文件名），
                # 或者如果你确定原代码这里是想保存同一份数据，那就这样操作。
                # 为了代码能跑通，我将复用 final_features 保存第二次。
            else:
                final_features = torch.cat(sampled_data, dim=0).numpy()

            np.save(rank_conv_dir+'/' + args.arch + '_limit%d' % (args.limit) + '/apoz_rank_conv_%d' % (cnt + 1) + '.npy',
                    final_features) # conv3
            cnt += 1
            sampled_data = []