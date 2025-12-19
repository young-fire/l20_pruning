import numpy as np
import math
from model.my_resnet_imagenet import resnet
from thop import profile
import torch
def apr_number(pr=0.53,types = '' ,lamda = 0):
    base_dir ="../generation_and_std/apoz/resnet_50_limit20/"
   
    all_values = []
    file_means = {}
    res_channel = [64, 64,  64, 256,  64,  64, 256, 64,  64, 256, 
            128,  128, 512,  128,  128,  512, 128,  128,  512, 128,  128,  512,  
            256,  256, 1024,  256,  256, 1024, 256,  256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024,
            512,  512,  512,  512, 512,  512]   
    npy_files =[]
    # excluded4, 14, 27, 46
    excluded_numbers = {4, 14, 27, 46,47,50,53}
    result_list = [i for i in range(1, 54) if i not in excluded_numbers]
    # print(result_list)
    for i, number in enumerate(result_list):
        file = types+"apoz_rank_conv" + str(number)+".npy"
        npy_files.append(file)
        data = np.load(base_dir +file, allow_pickle=True)
        # print(data)
        file_mean = np.mean([np.mean(lst) for lst in data])
        file_means[file] = file_mean
   ########################################################chose1  
   
        for idx, value in enumerate(data):
            
            af_value = (1-value)/math.pow(res_channel[i],lamda)    
            all_values.append((af_value, file, idx))
    all_values.sort()
    #######################################################chose2
    #     for idx, value in enumerate(data):
    #         af_value = value*math.pow(res_channel[i],lamda)
    #         all_values.append((af_value, file, idx))
    # all_values.sort(reverse=True)
    ##########################################################
    threshold_index = int(len(all_values) * pr)
    remaining_values = all_values[threshold_index:]
    print(all_values[threshold_index:threshold_index+1])    
    results = {}
    print(len(npy_files))
    for i in range(46):
        results[npy_files[i]] = []
    
    for value, file, idx in remaining_values:
        results[file].append(idx)
    
    sorted_files = sorted(npy_files, key=lambda x: int(x.split('apoz_rank_conv')[1].split('.')[0]))
    # print(sorted_files)
    all_mean =[]
    conv_number = []
    for file in sorted_files:
        indices = results[file]
        mean_value = file_means[file]
        print(f"File: {file},  Mean: {mean_value}")   
        conv_number.append(len(indices))
        all_mean.append(mean_value)
    return conv_number

def kept_number(pr=0.53):


    # pr =0.302 #0.44 0.577 0.6135
    res = apr_number(pr=pr,types='stdgyh', lamda =0)   
    res= res[0:42]+[2048]+res[42:44]+[2048]+res[44:]+[2048]  
    #############################################################################my_ratio
    res_conv1and2 = []
    for i in range(1, len(res), 3):
        res_conv1and2.extend(res[i:i+2])
    res_first_conv =  res[0]
    res_layerall =  res[3::3]
    res_layer1 = res_layerall[0:3]
    res_layer2 = res_layerall[3:7]
    res_layer3 = res_layerall[7:13]
    res_layer4 = res_layerall[13:16]
    res_conv1and2.append(res_first_conv) 
    res_conv1and2.append(int(sum(res_layer1)/3))
    res_conv1and2.append(int(sum(res_layer2)/4))
    res_conv1and2.append(int(sum(res_layer3)/6))
    res_conv1and2.append(int(sum(res_layer4)/3))

    # print(len(res_conv1and2))
    c =[64]*6 + [128]*8+[256]*12+[512]*6 + [64] + [256]+[512]+[1024]+[2048]
    d = [(res_channel-conv_number)/res_channel for conv_number,res_channel in zip(res_conv1and2,c)]

    d = [round(value, 2) for value in d]
    # print(d)
    #########################################################################

    model = resnet(cfg = 'resnet50', layer_cfg = res_conv1and2)
    input_image_size=224
    input_image = torch.randn(1, 3, input_image_size, input_image_size)
    flops, params = profile(model, inputs=(input_image,))
    print('Params: %.6f' % (params/1000000))   
    print('Flops: %.6f' % (flops/1000000))   
    print((flops/4111514624)/(params/25557032))
    print('kept_number',res_conv1and2)
    return res_conv1and2


