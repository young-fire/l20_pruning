import torch #203
import torch.nn as nn
import torch.optim as optim
from utils.options import args
import utils.common as utils
from thop import profile
from std_layer import kept_number
import os
import time
import math
from data import imagenet
from l20 import l20_rc
from importlib import import_module
if args.arch == 'my_resnet_imagenet':
    from model.my_resnet_imagenet import BasicBlock, Bottleneck
else :
    raise ValueError('invalid method : {:}'.format(args.method))

from utils.common import *


device = torch.device(f"cuda:{args.gpus[0]}") if torch.cuda.is_available() else 'cpu'
print(device)
checkpoint = utils.checkpoint(args)
now = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
logger = utils.get_logger(os.path.join(args.job_dir, 'logger'+now+'.log'))
if args.criterion == 'Softmax':
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
elif args.criterion == 'SmoothSoftmax':
    criterion = CrossEntropyLabelSmooth(1000,0.1)
    criterion = criterion.cuda()
else:
    raise ValueError('invalid criterion : {:}'.format(args.criterion))


# load training data
print('==> Preparing data..')
data_tmp = imagenet.Data(args)
train_loader = data_tmp.trainLoader
val_loader = data_tmp.testLoader



# Load pretrained model
print('==> Loading pretrained model..')
if args.pretrain_model is None or not os.path.exists(args.pretrain_model):
    raise ('Pretrained_model path should be exist!')
ckpt = torch.load(args.pretrain_model)#.to(device)
if args.arch == 'resnet_imagenet' or args.arch == 'my_resnet_imagenet' :
    origin_model = import_module(f'model.{args.arch}').resnet(args.cfg).to(device)
    origin_model.load_state_dict(ckpt)
else:
    raise('arch not exist!')

def l20(weight,m,logger,npy_dir):
    print(npy_dir)
    W = weight.cpu().clone()
    if weight.dim() == 4:  #Convolution layer
        W = W.view(W.size(0), -1)
    else:
        raise('The weight dim must be 4!')
    indices = l20_rc(npy_dir,m)
    Wprune = torch.index_select(W,0,torch.tensor(list(indices)))
    Wprune = Wprune.cpu()
    m_matrix = ''
    return m_matrix, Wprune, indices

def my_graph_resnet(pr_arget):


    centroids_state_dict = {}
    prune_state_dict = []
    indices = []

    current_index = 0
   
    blocks_num = [3, 4, 6, 3]
    block_index = 0
    stage = 0

   
    all_cfg = kept_number(pr=pr_arget)
    logger.info(all_cfg)
    conv12_number = [2, 3,  6, 7,  9, 10,  
                     12, 13,  16, 17,  19, 20,  22, 23,  
                     25, 26,  29, 30,  32, 33,  35, 36,  38, 39,  41, 42,  
                     44, 45,  48, 49, 51, 52]
    conv3_number = [ 5,  8,  11,
                      15,  18,  21,  24, 
                     28,  31,  34, 37,  40,  43, 
                      47, 50,  53]
    downsample_number = [ 5, 15, 28, 47]
    current_index = 0 #conv12_index
    conv3_index = 0
    downsample_index = 0
    conv1_weight = origin_model.state_dict()['conv1.weight']
    npy_dir = '../generation_and_std/feature/resnet_50_limit10/feature_rank_conv_1.npy'
    if os.path.exists(npy_dir):
        print(npy_dir)
    else:
        raise('npy file not exist!')
    print(all_cfg[-5])
    _,  lastcentroids, lastindice = l20(conv1_weight, all_cfg[-5],logger,npy_dir)   # 这里64,3,3,,3已经变成52,3,3,3(lastcentroids),lastindice是对应的索引
    # W = conv1_weight.cpu().clone()
    # if conv1_weight.dim() == 4:  #Convolution layer
    #     W = W.view(W.size(0), -1)
    # lastindice = l20_rc(npy_dir,all_cfg[-5])
    # Wprune = torch.index_select(W,0,torch.tensor(list(lastindice)))
    # lastcentroids = Wprune.cpu()

    centroids_state_dict['conv1.weight'] = lastcentroids

    centroids_state_dict['bn1.weight'] = origin_model.state_dict()['bn1.weight'][list(lastindice)].cpu()
    centroids_state_dict['bn1.bias'] = origin_model.state_dict()['bn1.bias'][list(lastindice)].cpu()
    centroids_state_dict['bn1.running_var'] = origin_model.state_dict()['bn1.running_var'][list(lastindice)].cpu()
    centroids_state_dict['bn1.running_mean'] = origin_model.state_dict()['bn1.running_mean'][list(lastindice)].cpu()
    '''
    prune_state_dict.append('bn1.bias')
    prune_state_dict.append('bn1.running_var')
    prune_state_dict.append('bn1.running_mean')
    '''
    last_downsample_indice = lastindice
    last_downsample_indice_1 = None
    for name, module in origin_model.named_modules():
        #print(name)
        if name.endswith('downsample'):
            downsample_weight = origin_model.state_dict()[name+'.0.weight']



            npy_dir = '../generation_and_std/feature/resnet_50_limit10/feature_rank_conv_'+str(downsample_number[downsample_index])+'.npy'
            print(all_cfg[-4+downsample_index])
            _,  centroids, indice = l20(downsample_weight, all_cfg[-4+downsample_index],logger,npy_dir)
            downsample_index +=1

            centroids_state_dict[name + '.0.weight'] = centroids.reshape((-1, downsample_weight.size(1), downsample_weight.size(2), downsample_weight.size(3)))
            if args.init_method == 'random_project':
                centroids_state_dict[name + '.0.weight'] = random_project(torch.FloatTensor(centroids_state_dict[name + '.0.weight']), len(last_downsample_indice))
            else:
                centroids_state_dict[name + '.0.weight'] = direct_project(torch.FloatTensor(centroids_state_dict[name + '.0.weight']), last_downsample_indice)
           
            centroids_state_dict[name + '.1.weight'] = origin_model.state_dict()[name + '.1.weight'][list(indice)].cpu()
            centroids_state_dict[name + '.1.bias'] = origin_model.state_dict()[name + '.1.bias'][list(indice)].cpu()
            centroids_state_dict[name + '.1.running_var'] = origin_model.state_dict()[name + '.1.running_var'][list(indice)].cpu()
            centroids_state_dict[name + '.1.running_mean'] = origin_model.state_dict()[name + '.1.running_mean'][list(indice)].cpu()
            '''
            prune_state_dict.append(name + '.1.weight')
            prune_state_dict.append(name + '.1.bias')
            prune_state_dict.append(name + '.1.running_var')
            prune_state_dict.append(name + '.1.running_mean')
            '''
            last_downsample_indice = last_downsample_indice_1


        if isinstance(module, Bottleneck):

            conv1_weight = module.conv1.weight.data

            npy_dir = '../generation_and_std/feature/resnet_50_limit10/feature_rank_conv_'+str(conv12_number[current_index])+'.npy'
            _,  centroids, indice = l20(conv1_weight, all_cfg[current_index],logger,npy_dir)
           
        #    cfg.append(len(centroids))
            indices.append(indice)
            centroids_state_dict[name + '.conv1.weight'] = centroids.reshape((-1, conv1_weight.size(1), conv1_weight.size(2), conv1_weight.size(3)))       #conv1的输出
            
            if args.init_method == 'random_project':
                centroids_state_dict[name + '.conv1.weight'] = random_project(torch.FloatTensor(centroids_state_dict[name + '.conv1.weight']), len(lastindice))
            else:
                centroids_state_dict[name + '.conv1.weight'] = direct_project(torch.FloatTensor(centroids_state_dict[name + '.conv1.weight']), lastindice) #conv1的输入

            centroids_state_dict[name + '.bn1.weight'] = origin_model.state_dict()[name + '.bn1.weight'][list(indice)].cpu()
            centroids_state_dict[name + '.bn1.bias'] = origin_model.state_dict()[name + '.bn1.bias'][list(indice)].cpu()
            centroids_state_dict[name + '.bn1.running_var'] = origin_model.state_dict()[name + '.bn1.running_var'][list(indice)].cpu()
            centroids_state_dict[name + '.bn1.running_mean'] = origin_model.state_dict()[name + '.bn1.running_mean'][list(indice)].cpu()

            '''
            prune_state_dict.append(name + '.bn1.weight')
            prune_state_dict.append(name + '.bn1.bias')
            prune_state_dict.append(name + '.bn1.running_var')
            prune_state_dict.append(name + '.bn1.running_mean')
            '''

            current_index += 1

            conv2_weight = module.conv2.weight.data
            npy_dir = '../generation_and_std/feature/resnet_50_limit10/feature_rank_conv_'+str(conv12_number[current_index])+'.npy'
            _,  centroids, indice = l20(conv2_weight, all_cfg[current_index],logger,npy_dir)
            # cfg.append(len(centroids))
            centroids_state_dict[name + '.conv2.weight'] = centroids.reshape((-1, conv2_weight.size(1), conv2_weight.size(2), conv2_weight.size(3)))           #conv2的输出

            if args.init_method == 'random_project':
                centroids_state_dict[name + '.conv3.weight'] = random_project(module.conv3.weight.data, len(centroids))
            else:
                centroids_state_dict[name + '.conv3.weight'] = direct_project(module.conv3.weight.data, indice)                                                #conv3的输入

            centroids_state_dict[name + '.bn2.weight'] = origin_model.state_dict()[name + '.bn2.weight'][list(indice)].cpu()
            centroids_state_dict[name + '.bn2.bias'] = origin_model.state_dict()[name + '.bn2.bias'][list(indice)].cpu()
            centroids_state_dict[name + '.bn2.running_var'] = origin_model.state_dict()[name + '.bn2.running_var'][list(indice)].cpu()
            centroids_state_dict[name + '.bn2.running_mean'] = origin_model.state_dict()[name + '.bn2.running_mean'][list(indice)].cpu()

            '''
            prune_state_dict.append(name + '.bn2.weight')
            prune_state_dict.append(name + '.bn2.bias')
            prune_state_dict.append(name + '.bn2.running_var')
            prune_state_dict.append(name + '.bn2.running_mean')
            '''
            current_index+=1

            conv3_weight = centroids_state_dict[name + '.conv3.weight']
            npy_dir = '../generation_and_std/feature/resnet_50_limit10/feature_rank_conv_'+str(conv3_number[conv3_index])+'.npy'                 
            print(all_cfg[-4+stage])
            _,  centroids, indice = l20(conv3_weight, all_cfg[-4+stage],logger,npy_dir)
            conv3_index +=1
            centroids_state_dict[name + '.conv3.weight'] = centroids.reshape((-1, conv3_weight.size(1), conv3_weight.size(2), conv3_weight.size(3)))        #conv3的输出
            
            lastindice = indice
            centroids_state_dict[name + '.bn3.weight'] = origin_model.state_dict()[name + '.bn3.weight'][list(indice)].cpu()
            centroids_state_dict[name + '.bn3.bias'] = origin_model.state_dict()[name + '.bn3.bias'][list(indice)].cpu()
            centroids_state_dict[name + '.bn3.running_var'] = origin_model.state_dict()[name + '.bn3.running_var'][list(indice)].cpu()
            centroids_state_dict[name + '.bn3.running_mean'] = origin_model.state_dict()[name + '.bn3.running_mean'][list(indice)].cpu()

            '''
            prune_state_dict.append(name + '.bn3.weight')
            prune_state_dict.append(name + '.bn3.bias')
            prune_state_dict.append(name + '.bn3.running_var')
            prune_state_dict.append(name + '.bn3.running_mean')
            '''

            last_downsample_indice_1 = indice

            block_index += 1
            if block_index == blocks_num[stage]:
                block_index = 0
                stage += 1
    
    fc_weight = origin_model.state_dict()['fc.weight'].cpu()
    
    pr_fc_weight = torch.randn(fc_weight.size(0),len(lastindice))
    for i, ind in enumerate(indice):
        pr_fc_weight[:,i] = fc_weight[:,ind] 

    centroids_state_dict['fc.weight'] = pr_fc_weight.cpu() #fc只改输入

    
    '''
    prune_state_dict.append('fc.weight')
    prune_state_dict.append('fc.bias')
    '''
    # cfg.extend(block_cfg)
    cfg = all_cfg
    print(cfg)

    model = import_module(f'model.{args.arch}').resnet(args.cfg, layer_cfg=cfg).to(device)
    if args.init_method == 'random_project' or args.init_method == 'direct_project':
        pretrain_state_dict = origin_model.state_dict()
        state_dict = model.state_dict()
        centroids_state_dict_keys = list(centroids_state_dict.keys())

        index = 0
        for k, v in centroids_state_dict.items():

            if k.endswith('.conv2.weight') and args.cfg != 'resnet18' and args.cfg != 'resnet34':
                if args.init_method == 'random_project':
                    centroids_state_dict[k] = random_project(torch.FloatTensor(centroids_state_dict[k]),                                                #conv2的输入
                                                             len(indices[index]))
                else:
                    centroids_state_dict[k] = direct_project(torch.FloatTensor(centroids_state_dict[k]), indices[index])
                index += 1

        for k, v in state_dict.items():
            #print(k)
            if k in prune_state_dict:
                continue
            elif k in centroids_state_dict_keys:
                print(k)
                state_dict[k] = torch.FloatTensor(centroids_state_dict[k]).view_as(state_dict[k])
            else:
                state_dict[k] = pretrain_state_dict[k]
        model.load_state_dict(state_dict)
    else:
        pass
    return model, cfg  

def my_graph_resnet18(pr_target):


    centroids_state_dict = {}
    prune_state_dict = []
    indices = []

    current_index = 0
   
    blocks_num = [2,2,2,2]
    block_index = 0
    stage = 0

    all_cfg = [50,  50,  100,  100,  200,  200, 
             400,  400,  61,  61,  100,  200,  400]  #69.9
    #all_cfg = [64,  64,  128,  128,  256,  256, 
             #512,  512,  64,  64,  128,  256,  512]  #69.9
    
    logger.info(all_cfg)
    conv1_number = [2,  6,    #64
                     12,   16,  #128
                     25,  29,    #256 
                     44,   48 ]  #512
    conv2_number = [2,  6,    
                     12,   16,  
                     25,  29,      
                     44,   48 ]
    downsample_number = [ 12, 25, 44]
    current_index = 0 #conv12_index
    conv3_index = 0
    downsample_index = 0
    conv1_weight = origin_model.state_dict()['conv1.weight']
    npy_dir = '../generation_and_std/feature/resnet_18_limit10/feature_rank_conv_1.npy'
    
    _,  lastcentroids, lastindice = l20(conv1_weight, all_cfg[-5],logger,npy_dir)  # 这里64,3,3,,3已经变成52,3,3,3(lastcentroids),lastindice是对应的索引

    centroids_state_dict['conv1.weight'] = lastcentroids

    centroids_state_dict['bn1.weight'] = origin_model.state_dict()['bn1.weight'][list(lastindice)].cpu()
    centroids_state_dict['bn1.bias'] = origin_model.state_dict()['bn1.bias'][list(lastindice)].cpu()
    centroids_state_dict['bn1.running_var'] = origin_model.state_dict()['bn1.running_var'][list(lastindice)].cpu()
    centroids_state_dict['bn1.running_mean'] = origin_model.state_dict()['bn1.running_mean'][list(lastindice)].cpu()
    '''
    prune_state_dict.append('bn1.bias')
    prune_state_dict.append('bn1.running_var')
    prune_state_dict.append('bn1.running_mean')
    '''
    last_downsample_indice = lastindice
    last_downsample_indice_1 = None
    for name, module in origin_model.named_modules():
        #print(name)
        if name.endswith('downsample'):
            downsample_weight = origin_model.state_dict()[name+'.0.weight']



            npy_dir = '../generation_and_std/feature/resnet_50_limit10/feature_rank_conv_'+str(downsample_number[downsample_index])+'.npy'
            print(all_cfg[-3+downsample_index])
            _,  centroids, indice = l20(downsample_weight, all_cfg[-3+downsample_index],logger,npy_dir)                                                  
            downsample_index +=1

            centroids_state_dict[name + '.0.weight'] = centroids.reshape((-1, downsample_weight.size(1), downsample_weight.size(2), downsample_weight.size(3)))       #downsample的输出
            if args.init_method == 'random_project':
                centroids_state_dict[name + '.0.weight'] = random_project(torch.FloatTensor(centroids_state_dict[name + '.0.weight']), len(last_downsample_indice))
            else:
                centroids_state_dict[name + '.0.weight'] = direct_project(torch.FloatTensor(centroids_state_dict[name + '.0.weight']), last_downsample_indice)        #downsample的输入
           
            centroids_state_dict[name + '.1.weight'] = origin_model.state_dict()[name + '.1.weight'][list(indice)].cpu()
            centroids_state_dict[name + '.1.bias'] = origin_model.state_dict()[name + '.1.bias'][list(indice)].cpu()
            centroids_state_dict[name + '.1.running_var'] = origin_model.state_dict()[name + '.1.running_var'][list(indice)].cpu()
            centroids_state_dict[name + '.1.running_mean'] = origin_model.state_dict()[name + '.1.running_mean'][list(indice)].cpu()
            '''
            prune_state_dict.append(name + '.1.weight')
            prune_state_dict.append(name + '.1.bias')
            prune_state_dict.append(name + '.1.running_var')
            prune_state_dict.append(name + '.1.running_mean')
            '''
            last_downsample_indice = last_downsample_indice_1

        if isinstance(module, BasicBlock):
            
            conv1_weight = module.conv1.weight.data
            npy_dir = '../generation_and_std/feature/resnet_50_limit10/feature_rank_conv_'+str(conv1_number[current_index])+'.npy'
            _,  centroids, indice = l20(conv1_weight, all_cfg[current_index],logger,npy_dir)  

            
            centroids_state_dict[name + '.conv1.weight'] = centroids.reshape((-1, conv1_weight.size(1), conv1_weight.size(2), conv1_weight.size(3)))                                                                                              #conv1的输出
            if args.init_method == 'random_project':
                centroids_state_dict[name + '.conv1.weight'] = random_project(torch.FloatTensor(centroids_state_dict[name + '.conv1.weight']), len(lastindice))
            else:
                centroids_state_dict[name + '.conv1.weight'] = direct_project(torch.FloatTensor(centroids_state_dict[name + '.conv1.weight']), lastindice)      #conv1的输入
                
           
            # prune_state_dict.append(name + '.bn1.weight')
            # prune_state_dict.append(name + '.bn1.bias')
            # prune_state_dict.append(name + '.bn1.running_var')
            # prune_state_dict.append(name + '.bn1.running_mean')
            centroids_state_dict[name + '.bn1.weight'] = origin_model.state_dict()[name + '.bn1.weight'][list(indice)].cpu()
            centroids_state_dict[name + '.bn1.bias'] = origin_model.state_dict()[name + '.bn1.bias'][list(indice)].cpu()
            centroids_state_dict[name + '.bn1.running_var'] = origin_model.state_dict()[name + '.bn1.running_var'][list(indice)].cpu()
            centroids_state_dict[name + '.bn1.running_mean'] = origin_model.state_dict()[name + '.bn1.running_mean'][list(indice)].cpu()
            
            
            if args.init_method == 'random_project':
                centroids_state_dict[name + '.conv2.weight'] = random_project(module.conv2.weight.data, len(centroids))
            else:
                centroids_state_dict[name + '.conv2.weight'] = direct_project(module.conv2.weight.data, indice)                                                 #conv2的输入 
                
            conv2_weight = centroids_state_dict[name + '.conv2.weight']
            npy_dir = '../generation_and_std/feature/resnet_18_limit10/feature_rank_conv_'+str(conv2_number[current_index])+'.npy'
            #print(name + '.conv2.weight')
            #print(all_cfg[-3+stage])
            _,  centroids, indice = l20(conv2_weight, all_cfg[-4+stage],logger,npy_dir)                                                                   
            #print('----',all_cfg[-4+stage])
            centroids_state_dict[name + '.conv2.weight'] = centroids.reshape((-1, conv2_weight.size(1), conv2_weight.size(2), conv2_weight.size(3)))           #conv2的输出
            #print(name + '.conv2.weight')
            #print(centroids_state_dict[name + '.conv2.weight'].shape)
            centroids_state_dict[name + '.bn2.weight'] = origin_model.state_dict()[name + '.bn2.weight'][list(indice)].cpu()
            centroids_state_dict[name + '.bn2.bias'] = origin_model.state_dict()[name + '.bn2.bias'][list(indice)].cpu()
            centroids_state_dict[name + '.bn2.running_var'] = origin_model.state_dict()[name + '.bn2.running_var'][list(indice)].cpu()
            centroids_state_dict[name + '.bn2.running_mean'] = origin_model.state_dict()[name + '.bn2.running_mean'][list(indice)].cpu()
            
            
            current_index+=1
            lastindice = indice
            last_downsample_indice_1 = indice

            block_index += 1
            if block_index == blocks_num[stage]:
                block_index = 0
                stage += 1
       
    
    fc_weight = origin_model.state_dict()['fc.weight'].cpu()
    
    pr_fc_weight = torch.randn(fc_weight.size(0),len(lastindice))
    for i, ind in enumerate(indice):
        pr_fc_weight[:,i] = fc_weight[:,ind]   #fc只改输入

    centroids_state_dict['fc.weight'] = pr_fc_weight.cpu()
    
    '''
    prune_state_dict.append('fc.weight')
    prune_state_dict.append('fc.bias')
    '''
    
    cfg = all_cfg
    print(cfg)

    model = import_module(f'model.{args.arch}').resnet(args.cfg, layer_cfg=cfg).to(device)
    print(model)
    if args.init_method == 'random_project' or args.init_method == 'direct_project':
        pretrain_state_dict = origin_model.state_dict()
        state_dict = model.state_dict()
        centroids_state_dict_keys = list(centroids_state_dict.keys())

        for k, v in state_dict.items():
            #print(k)
            if k in prune_state_dict:
                continue
            elif k in centroids_state_dict_keys:
                #print(centroids_state_dict[k])
                #print(pretrain_state_dict[k])
                state_dict[k] = torch.FloatTensor(centroids_state_dict[k]).view_as(state_dict[k])
                if  "bn1" in k:
                    print(k)
                    #state_dict[k] = torch.FloatTensor(pretrain_state_dict[k].cpu()).view_as(state_dict[k])
                else :
                    print('1111',k)
            else:
                print('0000',k)
                state_dict[k] = pretrain_state_dict[k]
        model.load_state_dict(state_dict)
    else:
        pass
    return model, cfg  


def train(epoch, train_loader, model, criterion, optimizer):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    data_time = utils.AverageMeter('Data', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')

    model.train()
    end = time.time()
    #scheduler.step()

    num_iter = len(train_loader)

    print_freq = num_iter // 10
    #i = 0 
    for batch_idx, (images, targets) in enumerate(train_loader):
        #if i > 5:
            #break
        #i += 1
        images = images.cuda()
        targets = targets.cuda()
        data_time.update(time.time() - end)

        adjust_learning_rate(optimizer, epoch, batch_idx, num_iter)

        # compute output
        logits = model(images)
        loss = criterion(logits, targets)

        # measure accuracy and record loss
        prec1, prec5 = utils.accuracy(logits, targets, topk=(1, 5))
        n = images.size(0)
        losses.update(loss.item(), n)  # accumulated loss
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % print_freq == 0 and batch_idx != 0:
            logger.info(
                'Epoch[{0}]({1}/{2}): '
                'Loss {loss.avg:.4f} '
                'Prec@1(1,5) {top1.avg:.2f}, {top5.avg:.2f}'.format(
                    epoch, batch_idx, num_iter, loss=losses,
                    top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg

def validate(val_loader, model, criterion, args):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')


    model.eval()
    with torch.no_grad():
        end = time.time()
        #i = 0
        for batch_idx, (images, targets) in enumerate(val_loader):
            #if i > 5:
                #break
            #i += 1
            images = images.cuda()
            targets = targets.cuda()

            # compute output
            logits = model(images)
            loss = criterion(logits, targets)

            # measure accuracy and record loss
            pred1, pred5 = utils.accuracy(logits, targets, topk=(1, 5))
            n = images.size(0)
            losses.update(loss.item(), n)
            top1.update(pred1[0], n)
            top5.update(pred5[0], n)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        logger.info(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                    .format(top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg


def adjust_learning_rate(optimizer, epoch, step, len_epoch):
    #Warmup
    if args.lr_type == 'step':
        factor = epoch // 30

        if epoch >= 80:
            factor = factor + 1

        lr = args.lr * (0.1 ** factor)
    elif args.lr_type == 'cos':  # cos without warm-up
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * (epoch - 5) / (args.num_epochs - 5)))
    else:
        raise NotImplementedError
    if epoch < 5:
            lr = lr * float(1 + step + epoch * len_epoch) / (5. * len_epoch)
    if step == 0:
        print('current learning rate:{0}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    start_epoch = 0
    best_top1_acc = 0.0
    best_top5_acc = 0.0

    #validate(val_loader, origin_model, criterion, args)
    
    print('==> Building Model..')
    if args.resume == None:

        if args.pretrain_model is None or not os.path.exists(args.pretrain_model):
            raise ('Pretrained_model path should be exist!')
        if args.arch == 'my_resnet_imagenet' and args.cfg=='resnet50':
            model, cfg = my_graph_resnet(args.pr_target)
        elif args.arch == 'my_resnet_imagenet' and args.cfg=='resnet18':
            model, cfg = my_graph_resnet18(args.pr_target)
        else:
            raise('arch not exist!')
        print("Graph Down!")
        logger.info(model)
        #model = model.to(device)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    else:
        resumeckpt = torch.load(args.resume)
        state_dict = resumeckpt['state_dict']
        cfg = resumeckpt['cfg']
        
        if args.arch == 'resnet_imagenet':
            model = import_module(f'model.{args.arch}').resnet(args.cfg, layer_cfg=cfg).to(device)
        elif args.arch == 'my_resnet_imagenet':
            model = import_module(f'model.{args.arch}').resnet(args.cfg, layer_cfg=cfg).to(device)
        
        else:
            raise('arch not exist!')

        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        model.load_state_dict(state_dict)
        optimizer.load_state_dict(resumeckpt['optimizer'])
        start_epoch = resumeckpt['epoch']

    
    # calculate model size
    input_image_size = 224
    input_image = torch.randn(1, 3, input_image_size, input_image_size).to(device)
    flops, params = profile(model, inputs=(input_image,))
    logger.info('Params: %.2f' % (params))
    logger.info('Flops: %.2f' % (flops))

    if len(args.gpus) != 1:
        model = nn.DataParallel(model, device_ids=args.gpus).cuda()
    for epoch in range(start_epoch, args.num_epochs):
        #valid_obj, test_top1_acc, test_top5_acc = validate(val_loader, model, criterion, args)
        #aaa
        train_obj, train_top1_acc,  train_top5_acc = train(epoch,  train_loader, model, criterion, optimizer)
        valid_obj, test_top1_acc, test_top5_acc = validate(val_loader, model, criterion, args)

        is_best = best_top5_acc < test_top5_acc
        best_top1_acc = max(best_top1_acc, test_top1_acc)
        best_top5_acc = max(best_top5_acc, test_top5_acc)

        model_state_dict = model.module.state_dict() if len(args.gpus) > 1 else model.state_dict()

        state = {
            'state_dict': model_state_dict,
            'best_top1_acc': best_top1_acc,
            'best_top5_acc': best_top5_acc,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'arch': args.cfg,
            'cfg': cfg
        }
        checkpoint.save_model(state, epoch + 1, is_best)

    logger.info('Best Top-1 accuracy: {:.3f} Top-5 accuracy: {:.3f}'.format(float(best_top1_acc), float(best_top5_acc)))

    
if __name__ == '__main__':
    main()