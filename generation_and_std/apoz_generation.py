
import os
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import torchvision
from torchvision import datasets, transforms

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
    default=5,
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


#os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
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
        #transforms.ToPILImage(),
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
    '''def get_data_set():
        return imagenet_dali.get_imagenet_iter_dali('train', args.data_dir, args.batch_size,
                                                        num_threads=4, crop=224, device_id=0, num_gpus=1)
    train_loader = get_data_set()'''
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
    # Load checkpoint.
    print(args.pretrain_dir)
    print('==> Resuming from checkpoint..')
    if args.arch=='vgg_16_bn' or args.arch=='resnet_56' or args.arch=='resnet110_cifar':
        checkpoint = torch.load(args.pretrain_dir,map_location='cuda:'+args.gpu)
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
feature_result = torch.tensor(0.)#0的个数
total = torch.tensor(0.)  #非0的个数


#get feature map of certain layer via hook
def apoz(b):
    tmp_zeros = (b == 0).sum().item()
    tmp_ones = (b != 0).sum().item()
    #print(tmp_zeros/(tmp_zeros+tmp_ones))
    return tmp_zeros/(tmp_zeros+tmp_ones)

def fly(b):
    h,w = b.shape
    #print(tmp_zeros/(tmp_zeros+tmp_ones))
    return (sum(sum(b*b)))/(h*w)
#get feature map of certain layer via hook
def get_feature_hook(self, input, output):    
    global feature_result
    global entropy
    global total
    a = output.shape[0]
    b = output.shape[1]
    
    # c = torch.tensor([torch.matrix_rank(output[i,j,:,:]).item() for i in range(a) for j in range(b)])
    c = torch.tensor([apoz(output[i,j,:,:].cpu()) for i in range(a) for j in range(b)])
    c = c.view(a, -1).float()
    c = c.sum(0)
    feature_result = feature_result * total + c
    total = total + a
    feature_result = feature_result / total
    
#def get_feature_hook1(self, input, output):    
    #global num_zeros
    #global num_ones
    #n,c,h,w = output.shape    #8388608
    
    #tmp_zeros = (output == 0).sum().item()
    #tmp_ones = (output != 0).sum().item()
    
    #num_zeros = num_zeros + tmp_zeros
    #num_ones = tmp_ones + num_ones
    #print(num_zeros+num_ones)
    #print('-----------')


def inference():
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    limit = args.limit

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            #use the first 5 batches to estimate the rank.
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
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))#'''



if args.arch=='vgg_16_bn' or args.arch=='vgg16_cifar':

    if len(args.gpu) > 1:
        relucfg = net.module.relucfg
    else:
        relucfg = net.relucfg

    for i, cov_id in enumerate(relucfg):
        cov_layer = net.features[cov_id]
        
        if not isinstance(cov_layer, nn.ReLU):
            cov_layer = net.features[cov_id-1]
        print(cov_layer)
        handler = cov_layer.register_forward_hook(get_feature_hook)
        inference()
        handler.remove()

        if not os.path.isdir(rank_conv_dir+'/'+args.arch+'_limit%d'%(args.limit)):
            os.mkdir(rank_conv_dir+'/'+args.arch+'_limit%d'%(args.limit))
        #print('zero',num_zeros)
        #print('one',num_ones)
        #print('all',num_zeros+num_ones)
        #apoz = num_zeros/(num_zeros+num_ones)
        #print(feature_result)
        np.save(rank_conv_dir+'/'+args.arch+'_limit%d'%(args.limit)+'/apoz_rank_conv' + str(i + 1) + '.npy', feature_result.numpy())
        print(feature_result.mean())
        #pro
        feature_result = torch.tensor(0.)
        total = torch.tensor(0.)

elif args.arch=='resnet_56' or args.arch=='resnet56_cifar' or args.arch=='resnet_56_cifar100':

    cov_layer = eval('net.relu')
    handler = cov_layer.register_forward_hook(get_feature_hook)
    inference()
    handler.remove()

    if not os.path.isdir(rank_conv_dir+'/' + args.arch+'_limit%d'%(args.limit)):
        os.mkdir(rank_conv_dir+'/' + args.arch+'_limit%d'%(args.limit))
    np.save(rank_conv_dir+'/' + args.arch+'_limit%d'%(args.limit)+ '/apoz_rank_conv%d' % (1) + '.npy', feature_result.numpy())
    feature_result = torch.tensor(0.)
    total = torch.tensor(0.)

    # ResNet56 per block
    cnt=0
    for i in range(3):
        block = eval('net.layer%d' % (i + 1))
        for j in range(9):
            cov_layer = block[j].relu1
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            np.save(rank_conv_dir+'/' + args.arch +'_limit%d'%(args.limit)+ '/apoz_rank_block%d'%(cnt + 1)+'_relu1.npy', feature_result.numpy())
            
            print(feature_result.mean())
            feature_result = torch.tensor(0.)
            total = torch.tensor(0.)
            
            cov_layer = block[j].relu2
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            np.save(rank_conv_dir+'/'+ args.arch +'_limit%d'%(args.limit)+ '/apoz_rank_block%d'%(cnt + 1)+'_relu2.npy', feature_result.numpy())
            print(feature_result.mean())
            cnt += 1
            feature_result = torch.tensor(0.)
            total = torch.tensor(0.)
            



elif args.arch=='resnet_110' or args.arch=='resnet110_cifar':

    cov_layer = eval('net.relu')
    handler = cov_layer.register_forward_hook(get_feature_hook)
    inference()
    handler.remove()

    if not os.path.isdir(rank_conv_dir+'/' + args.arch+'_limit%d'%(args.limit)):
        os.mkdir(rank_conv_dir+'/' + args.arch+'_limit%d'%(args.limit))
    #np.save(rank_conv_dir+'/' + args.arch+'_limit%d'%(args.limit) + '/apoz_rank_conv%d' % (1) + '.npy', feature_result.numpy())
    np.save(rank_conv_dir+'/' + args.arch+'_limit%d'%(args.limit) + '/new_rank_conv%d' % (1) + '.npy', feature_result.numpy())
    feature_result = torch.tensor(0.)
    total = torch.tensor(0.)

    cnt = 1
    # ResNet110 per block
    for i in range(3):
        block = eval('net.layer%d' % (i + 1))
        for j in range(18):
            cov_layer = block[j].relu1
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()

            #np.save(rank_conv_dir+'/' + args.arch  + '_limit%d' % (args.limit) + '/apoz_rank_block%d'%(cnt + 1)+'_relu1.npy', feature_result.numpy())
            np.save(rank_conv_dir+'/' + args.arch  + '_limit%d' % (args.limit) + '/new_rank_conv%d'%(cnt + 1)+'.npy', feature_result.numpy())
            cnt += 1
            
            
            feature_result = torch.tensor(0.)
            total = torch.tensor(0.)

            cov_layer = block[j].relu2
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            
            #np.save(rank_conv_dir+'/' + args.arch  + '_limit%d' % (args.limit) + '/apoz_rank_block%d'%(cnt + 1)+'_relu2.npy', feature_result.numpy())
            np.save(rank_conv_dir+'/' + args.arch  + '_limit%d' % (args.limit) + '/new_rank_conv%d'%(cnt + 1)+'.npy', feature_result.numpy())
            
            cnt += 1
            feature_result = torch.tensor(0.)
            total = torch.tensor(0.)

elif args.arch=='resnet_50':
    cov_layer = eval('net.maxpool')
    handler = cov_layer.register_forward_hook(get_feature_hook)
    inference()
    handler.remove()

    if not os.path.isdir(rank_conv_dir+'/' + args.arch+'_limit%d'%(args.limit)):
        os.mkdir(rank_conv_dir+'/' + args.arch+'_limit%d'%(args.limit))
    np.save(rank_conv_dir+'/' + args.arch+'_limit%d'%(args.limit) + '/apoz_rank_conv_%d' % (1) + '.npy', feature_result.numpy())
    feature_result = torch.tensor(0.)
    total = torch.tensor(0.)

    # ResNet50 per bottleneck
    cnt=1
    for i in range(4):
        block = eval('net.layer%d' % (i + 1))
        print('=====')
        for j in range(net.num_blocks[i]):
            cov_layer = block[j].relu1
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            # np.save(rank_conv_dir+'/' + args.arch+'_limit%d'%(args.limit) + '/apoz_rank_conv_relu1_%d'%(cnt+1)+'.npy',
            #         feature_result.numpy())
            np.save(rank_conv_dir+'/' + args.arch+'_limit%d'%(args.limit) + '/apoz_rank_conv_%d'%(cnt+1)+'.npy',
                    feature_result.numpy())
            cnt+=1
            feature_result = torch.tensor(0.)
            total = torch.tensor(0.)

            cov_layer = block[j].relu2
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            # np.save(rank_conv_dir+'/' + args.arch + '_limit%d' % (args.limit) + '/apoz_rank_conv_relu2_%d' % (cnt + 1) + '.npy',
            #         feature_result.numpy())
            np.save(rank_conv_dir+'/' + args.arch + '_limit%d' % (args.limit) + '/apoz_rank_conv_%d' % (cnt + 1) + '.npy',
                    feature_result.numpy())
            cnt += 1
            feature_result = torch.tensor(0.)
            total = torch.tensor(0.)

            cov_layer = block[j].relu3
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            if j==0:
                print(cnt +1)
                # np.save(rank_conv_dir+'/' + args.arch + '_limit%d' % (args.limit) + '/apoz_rank_conv_short_%d' % (cnt + 1) + '.npy',
                #         feature_result.numpy())#shortcut conv
                np.save(rank_conv_dir+'/' + args.arch + '_limit%d' % (args.limit) + '/apoz_rank_conv_%d' % (cnt + 1) + '.npy',
                        feature_result.numpy())#shortcut conv
                cnt += 1
            # np.save(rank_conv_dir+'/' + args.arch + '_limit%d' % (args.limit) + '/apoz_rank_conv_relu3_%d' % (cnt + 1) + '.npy',
            #         feature_result.numpy())#conv3
            np.save(rank_conv_dir+'/' + args.arch + '_limit%d' % (args.limit) + '/apoz_rank_conv_%d' % (cnt + 1) + '.npy',
                    feature_result.numpy())#conv3
            cnt += 1
            feature_result = torch.tensor(0.)
            total = torch.tensor(0.)


#'''
