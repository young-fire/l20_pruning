import numpy as np
import glob
import os

name_dir ="apoz/resnet_50_limit20/" 
npy_files = glob.glob(name_dir+"apoz_rank*.npy")


for file in npy_files:
    data = np.load(file, allow_pickle=True)
    
    
    mean = np.mean(data)
    std = np.std(data)
    normalized_data = (data - mean) / std
    new_file_name = name_dir+"stdgyh"+file.split('/')[2]
    print(new_file_name)
    np.save(new_file_name, normalized_data)

    
    
    print(f"Processed and saved {new_file_name}")

