'''
Version: V1.0.0
Author: kll 3327175694@qq.com
Date: 2024-09-18 20:36:23
LastEditors: kll 3327175694@qq.com
LastEditTime: 2025-08-03 20:02:40
FilePath: run.py
Copyright 2024 Marvin, All Rights Reserved. 
Description: 
'''
from simclr import SimCLR
import yaml
from data_aug.dataset_wrapper import DataSetWrapper
import os, glob
import pandas as pd
import argparse

'''
 SimCLR 框架进行数据增强并训练嵌入模型'''

# generate_csv 函数用于生成一个 CSV 文件 all_patches.csv，其中包含指定目录下所有符合条件的图像文件路径。
# generate_csv 函数的作用是基于输入的图像层级参数生成图像文件的路径，并将其保存到一个 CSV 文件中。
def generate_csv(args):
# args.level：表示图像的放大级别，取值为 'low' 或 'high'。
# args.multiscale：指定是否启用多尺度处理，取值为 1 表示启用，0 表示禁用。
# args.dataset：数据集文件夹名称。
    if args.level=='high' and args.multiscale==1:
        path_temp = os.path.join('..', 'dataset', args.dataset, 'cell', '*', '*', '*', '*.jpeg')
        patch_path = glob.glob(path_temp) # /class_name/bag_name/5x_name/*.jpeg
    if args.level=='low' and args.multiscale==1:
        path_temp = os.path.join('..', 'WSI', args.dataset, 'pyramid', '*', '*', '*.jpeg')
        patch_path = glob.glob(path_temp) # /class_name/bag_name/*.jpeg
    if args.multiscale==0:
        # path_temp = os.path.join('..', 'WSI', args.dataset, 'single', '*', '*', '*.jpeg')
        # patch_path = glob.glob(path_temp) # /class_name/bag_name/*.jpeg
        path_temp = os.path.join('/tempDataset', args.dataset, 'abnormal', '*', '*', '*.jpeg')
        patch_path = glob.glob(path_temp) # /class_name/bag_name/*.jpeg
    df = pd.DataFrame(patch_path)
    df.to_csv('all_patches.csv', index=False)
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--level', type=str, default='high', help='Magnification level to compute embedder (low/high)')
    parser.add_argument('--multiscale', type=int, default=1, help='Whether the patches are cropped from multiscale (0/1-no/yes)')
    parser.add_argument('--dataset', type=str, default='WSILabel', help='Dataset folder name')
    # parser.add_argument('--dataset', type=str, default='TCGA-lung', help='Dataset folder name')
    args = parser.parse_args()
    config = yaml.load(open("simclr/config.yaml", "r"), Loader=yaml.FullLoader)
    gpu_ids = eval(config['gpu_ids'])
    os.environ['CUDA_VISIBLE_DEVICES']=','.join(str(x) for x in gpu_ids)  
    #  数据集包装器：初始化 DataSetWrapper 实例，以指定的 batch 大小和数据集配置参数创建数据集对象
    dataset = DataSetWrapper(config['batch_size'], **config['dataset']) 
    # 生成 CSV 文件：调用 generate_csv(args)，根据 args 中的参数生成 CSV 文件  
    generate_csv(args)
    simclr = SimCLR(dataset, config)
    simclr.train()


if __name__ == "__main__":
    main()

# python run.py --level high --multiscale 1 --dataset WSILabel/abnormalNormalCell