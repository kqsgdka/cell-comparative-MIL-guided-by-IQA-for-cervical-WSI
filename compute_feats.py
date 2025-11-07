import dsmil as mil

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms.functional as VF
from torchvision import transforms

import sys, argparse, os, glob, copy
import pandas as pd
import numpy as np
from PIL import Image
from collections import OrderedDict
from sklearn.utils import shuffle
from torchvision import transforms

os.chdir('/home75/kanglanlan/WSI/code/dsmil-wsi-cervical')


'''
使用MIL和SIMCLR预训练模型，提取图像特征用于分类任务
核心包括：图像处理；特征提取；和数据保存流程，主要集中WSI数据集中的图像
'''

# BagDataset 该类负责读取存储在CSV文件中的图像路径，并支持图像的可选变换
class BagDataset():
    def __init__(self, csv_file, transform=None):
        self.files_list = csv_file
        self.transform = transform
    def __len__(self):
        return len(self.files_list)
    def __getitem__(self, idx):
        temp_path = self.files_list[idx]
        img = os.path.join(temp_path)
        try:
            img = Image.open(img)
            img = img.resize((128, 128))
            sample = {'input': img}
            if self.transform:
                sample = self.transform(sample)
            return sample, temp_path
        except UnidentifiedImageError:
            print(f"文件 {img} 损坏，跳过该文件。")
            return None, temp_path


# 将图像转换为PyTorch的张量格式
class ToTensor(object):
    def __call__(self, sample):
        img = sample['input']
        img = VF.to_tensor(img)
        return {'input': img} 



# 按顺序将系列转换应用于每一张图像，类似于torchvision.transforms.Compose  
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img



# '''
# 构建一个数据加载器Dataloader以批量加载数据，接受CSV文件路径，方便后续特征处理的批处理操作
# 	从指定的CSV文件路径加载图像，应用定义的转换，并创建一个DataLoader以分批迭代数据'''
def bag_dataset(args, csv_file_path):
    transformed_dataset = BagDataset(csv_file=csv_file_path,
                                    transform=Compose([
                                        ToTensor()
                                    ]))
    # if len(csv_file_path) < args.batch_size:
    #     args.batch_size = max(1, len(csv_file_path)//4)
    # else:
    #     pass
    dataloader = DataLoader(transformed_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
    return dataloader, len(transformed_dataset)


'''从图像中提取特征并保存为CSV文件;
设定图像的放大倍率，确定要提取特征图像的路径，
利用预训练分类器i_classifier提取特征，在不启用梯度情况下，提取特征，
最后将特征保存为CSV文件'''
# 步骤： 对于每一WSI，加载所有的图像块，并传递给i_classifier，提取特征，保存为CSV文件
def compute_feats(args, bags_list, i_classifier, save_path=None, magnification='single'):
    i_classifier.eval()
    num_bags = len(bags_list)
    Tensor = torch.FloatTensor
    for i in range(0, num_bags):
        feats_list = []
        paths_list = []
        if magnification=='single' or magnification=='low':
            csv_file_path = glob.glob(os.path.join(bags_list[i], '*.jpg')) + glob.glob(os.path.join(bags_list[i], '*.jpeg'))
            # csv_file_path = glob.glob(os.path.join(bags_list[i], '*.jpg')) + glob.glob(os.path.join(bags_list[i], '*.jpeg'))
        elif magnification=='high':
            csv_file_path = glob.glob(os.path.join(bags_list[i], '*'+os.sep+'*.jpg')) + glob.glob(os.path.join(bags_list[i], '*'+os.sep+'*.jpeg'))
            print('csv_file_path:', len(csv_file_path))
        dataloader, bag_size = bag_dataset(args, csv_file_path)
        with torch.no_grad():
            for iteration, batch in enumerate(dataloader):
                patches = batch[0]['input'].float().cuda() 
                feats, classes = i_classifier(patches)
                feats = feats.cpu().numpy()
                feats_list.extend(feats)
                paths_list.extend(batch[1])
                sys.stdout.write('\r Computed: {}/{} -- {}/{}'.format(i+1, num_bags, iteration+1, len(dataloader)))

        if len(feats_list) == 0:
            print('No valid patch extracted from: ' + bags_list[i])
        else:
            df = pd.DataFrame(feats_list)
            dfPaths = pd.DataFrame(paths_list)
            os.makedirs(os.path.join(save_path, bags_list[i].split(os.path.sep)[-2]), exist_ok=True)
            df.to_csv(os.path.join(save_path, bags_list[i].split(os.path.sep)[-2], bags_list[i].split(os.path.sep)[-1]+'.csv'), index=False, float_format='%.4f')
            # # 写入CSV文件文件里包的 patch 路径顺序
            # dfPaths = pd.DataFrame(paths_list_abnormal+paths_list_normal)
            # os.makedirs(os.path.join(save_path,'patchsNamePath', bags_list[i].split(os.path.sep)[-2]), exist_ok=True)
            # dfPaths.to_csv(os.path.join(os.path.join(save_path,'patchsNamePath', bags_list[i].split(os.path.sep)[-2], bags_list[i].split(os.path.sep)[-1]+'.csv')), index=False, float_format='%.4f')
            # # 写入特征
            # df_abnormal = pd.DataFrame(feats_list_abnormal)
            # df_normal = pd.DataFrame(feats_list_normal)
            # os.makedirs(os.path.join(save_path, bags_list[i].split(os.path.sep)[-2], bags_list[i].split(os.path.sep)[-1]), exist_ok=True)
            # df_abnormal.to_csv(os.path.join(save_path, bags_list[i].split(os.path.sep)[-2], bags_list[i].split(os.path.sep)[-1], "abnormal"+'.csv'), index=False, float_format='%.4f')
            # df_normal.to_csv(os.path.join(save_path, bags_list[i].split(os.path.sep)[-2], bags_list[i].split(os.path.sep)[-1], "normal"+'.csv'), index=False, float_format='%.4f')

'''
函数针对具有多重放大倍率（高分倍率和低分倍率）的数据集。实现“tree_fusion”，用于不同分辨率下的特征融合；
使用低分辨率的嵌入器（embedder_low）提取图像特征，
对于每个低分辨率图像，其对应的高分辨率图像会使用embedder_high提取特征，
根据tree——fusion参数选择融合方式：fusion：直接加权；cat：拼接特征。
保存融合之后的特征
'''
def compute_tree_feats(args, bags_list, embedder_low, embedder_high, save_path=None):
    embedder_low.eval()
    embedder_high.eval()
    num_bags = len(bags_list)
    Tensor = torch.FloatTensor
    with torch.no_grad():
        for i in range(0, num_bags): 
            low_patches = glob.glob(os.path.join(bags_list[i], '*.jpg')) + glob.glob(os.path.join(bags_list[i], '*.jpeg'))
            feats_list = []
            feats_tree_list = []
            dataloader, bag_size = bag_dataset(args, low_patches)
            for iteration, batch in enumerate(dataloader):
                patches = batch[0]['input'].float().cuda()
                feats, classes = embedder_low(patches)
                feats = feats.cpu().numpy()
                feats_list.extend(feats)
            for idx, low_patch in enumerate(low_patches):
                high_folder = os.path.dirname(low_patch) + os.sep + os.path.splitext(os.path.basename(low_patch))[0]
                high_patches = glob.glob(high_folder+os.sep+'*.jpg') + glob.glob(high_folder+os.sep+'*.jpeg')
                if len(high_patches) == 0:
                    pass
                else:
                    for high_patch in high_patches:
                        img = Image.open(high_patch)
                        img = VF.to_tensor(img).float().cuda()
                        feats, classes = embedder_high(img[None, :])
                        
                        if args.tree_fusion == 'fusion':
                            feats = feats.cpu().numpy()+0.25*feats_list[idx]
                        elif args.tree_fusion == 'cat':
                            feats = np.concatenate((feats.cpu().numpy(), feats_list[idx][None, :]), axis=-1)
                        else:
                            raise NotImplementedError(f"{args.tree_fusion} is not an excepted option for --tree_fusion. This argument accepts 2 options: 'fusion' and 'cat'.")
                        
                        feats_tree_list.extend(feats)
                sys.stdout.write('\r Computed: {}/{} -- {}/{}'.format(i+1, num_bags, idx+1, len(low_patches)))
            if len(feats_tree_list) == 0:
                print('No valid patch extracted from: ' + bags_list[i])
            else:
                df = pd.DataFrame(feats_tree_list)
                os.makedirs(os.path.join(save_path, bags_list[i].split(os.path.sep)[-2]), exist_ok=True)
                df.to_csv(os.path.join(save_path, bags_list[i].split(os.path.sep)[-2], bags_list[i].split(os.path.sep)[-1]+'.csv'), index=False, float_format='%.4f')
            print('\n')            

def main():
    parser = argparse.ArgumentParser(description='Compute TCGA features from SimCLR embedder')
    parser.add_argument('--num_classes', default=2, type=int, help='Number of output classes [2]')
    parser.add_argument('--batch_size', default=1024, type=int, help='Batch size of dataloader [128]')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of threads for datalodaer')
    parser.add_argument('--gpu_index', type=int, nargs='+', default=([0,]), help='GPU ID(s) [0]')
    parser.add_argument('--backbone', default='resnet18', type=str, help='Embedder backbone [resnet18]')
    parser.add_argument('--norm_layer', default='batch', type=str, help='Normalization layer [instance]')
    parser.add_argument('--magnification', default='tree', type=str, help='Magnification to compute features. Use `tree` for multiple magnifications. Use `high` if patches are cropped for multiple resolution and only process higher level, `low` for only processing lower level.')
    # parser.add_argument('--magnification', default='single', type=str, help='Magnification to compute features. Use `tree` for multiple magnifications. Use `high` if patches are cropped for multiple resolution and only process higher level, `low` for only processing lower level.')
    parser.add_argument('--weights', default='ImageNet', type=str, help='Folder of the pretrained weights, simclr/runs/*')
    # parser.add_argument('--weights', default='ImageNet', type=str, help='Folder of the pretrained weights, simclr/runs/*')
    parser.add_argument('--weights_high', default='ImageNet', type=str, help='Folder of the pretrained weights of high magnification, FOLDER < `simclr/runs/[FOLDER]`')
    parser.add_argument('--weights_low', default='ImageNet', type=str, help='Folder of the pretrained weights of low magnification, FOLDER <`simclr/runs/[FOLDER]`')
    parser.add_argument('--tree_fusion', default='cat', type=str, help='Fusion method for high and low mag features in a tree method [cat|fusion]')
    parser.add_argument('--dataset', default='WSILabel', type=str, help='Dataset folder name [TCGA-lung-single]')
    # parser.add_argument('--dataset', default='Camelyon16_ImageNet', type=str, help='Dataset folder name [TCGA-lung-single]')
    args = parser.parse_args()
    gpu_ids = tuple(args.gpu_index)
    os.environ['CUDA_VISIBLE_DEVICES']=','.join(str(x) for x in gpu_ids)

    if args.norm_layer == 'instance':
        norm=nn.InstanceNorm2d
        pretrain = False
    elif args.norm_layer == 'batch':  
        norm=nn.BatchNorm2d
        if args.weights == 'ImageNet':
            pretrain = True
        else:
            pretrain = False
# 参数设置^

# 加载预训练模型
    if args.backbone == 'resnet18':
        resnet = models.resnet18(pretrained=pretrain, norm_layer=norm)
        num_feats = 512
    if args.backbone == 'resnet34':
        resnet = models.resnet34(pretrained=pretrain, norm_layer=norm)
        num_feats = 512
    if args.backbone == 'resnet50':
        resnet = models.resnet50(pretrained=pretrain, norm_layer=norm)
        num_feats = 2048
    if args.backbone == 'resnet101':
        resnet = models.resnet101(pretrained=pretrain, norm_layer=norm)
        num_feats = 2048
    for param in resnet.parameters():
        param.requires_grad = False
    resnet.fc = nn.Identity()
    
    # 加载权重
    if args.magnification == 'tree' and args.weights_high != None and args.weights_low != None:
        i_classifier_h = mil.IClassifier(resnet, num_feats, output_class=args.num_classes).cuda()
        i_classifier_l = mil.IClassifier(copy.deepcopy(resnet), num_feats, output_class=args.num_classes).cuda()
        
        if args.weights_high == 'ImageNet' or args.weights_low == 'ImageNet' or args.weights== 'ImageNet':
            if args.norm_layer == 'batch':
                print('Use ImageNet features.')
            else:
                raise ValueError('Please use batch normalization for ImageNet feature')
        else:
            weight_path = os.path.join('simclr', 'runs', args.weights_high, 'checkpoints', 'model.pth')
            state_dict_weights = torch.load(weight_path)
            for i in range(4):
                state_dict_weights.popitem()
            state_dict_init = i_classifier_h.state_dict()
            new_state_dict = OrderedDict()
            for (k, v), (k_0, v_0) in zip(state_dict_weights.items(), state_dict_init.items()):
                name = k_0
                new_state_dict[name] = v
            i_classifier_h.load_state_dict(new_state_dict, strict=False)
            os.makedirs(os.path.join('embedder', args.dataset), exist_ok=True)
            torch.save(new_state_dict, os.path.join('embedder', args.dataset, 'embedder-high.pth'))

            weight_path = os.path.join('simclr', 'runs', args.weights_low, 'checkpoints', 'model.pth')
            state_dict_weights = torch.load(weight_path)
            for i in range(4):
                state_dict_weights.popitem()
            state_dict_init = i_classifier_l.state_dict()
            new_state_dict = OrderedDict()
            for (k, v), (k_0, v_0) in zip(state_dict_weights.items(), state_dict_init.items()):
                name = k_0
                new_state_dict[name] = v
            i_classifier_l.load_state_dict(new_state_dict, strict=False)
            os.makedirs(os.path.join('embedder', args.dataset), exist_ok=True)
            torch.save(new_state_dict, os.path.join('embedder', args.dataset, 'embedder-low.pth'))
            print('Use pretrained features.')


    elif args.magnification == 'single' or args.magnification == 'high' or args.magnification == 'low':  
        i_classifier = mil.IClassifier(resnet, num_feats, output_class=args.num_classes).cuda()

        if args.weights == 'ImageNet':
            if args.norm_layer == 'batch':
                print('Use ImageNet features.')
            else:
                print('Please use batch normalization for ImageNet feature')
        else:
            if args.weights is not None:
                weight_path = os.path.join('simclr', 'run', args.weights, 'checkpoints', 'model.pth')
            else:
                weight_path = glob.glob('simclr/run/*/checkpoints/*.pth')[-1]

            state_dict_weights = torch.load(weight_path) 
            print('Loaded state_dict_weights:', state_dict_weights.keys())
            print(len(state_dict_weights)) 

            state_dict_init = i_classifier.state_dict()

            print('state_dict_weights:', state_dict_weights.keys())
            print('state_dict_init:', state_dict_init.keys())
            
            new_state_dict = OrderedDict()

            for k, v in state_dict_weights.items():
                # 条件1：权重文件中的键必须存在于模型参数中
                if k not in state_dict_init:
                    print(f"过滤掉不存在的参数: {k}")
                    continue
                
                # 条件2：确保参数是张量类型（排除嵌套的OrderedDict等）
                if not isinstance(v, torch.Tensor):
                    print(f"过滤掉非张量类型参数: {k} (类型: {type(v)})")
                    continue
                
                # 条件3：确保参数形状匹配
                if v.shape != state_dict_init[k].shape:
                    print(f"过滤掉形状不匹配的参数: {k} (权重形状: {v.shape}, 模型形状: {state_dict_init[k].shape})")
                    continue
                
                # 所有条件满足，保留该参数
                new_state_dict[k] = v

            # for (k, v), (k_0, v_0) in zip(state_dict_weights['model'].items(), state_dict_init.items()):
            #     name = k_0
            #     new_state_dict[name] = v
            i_classifier.load_state_dict(new_state_dict, strict=False)
            os.makedirs(os.path.join('embedder', args.dataset), exist_ok=True)
            torch.save(new_state_dict, os.path.join('embedder', args.dataset, 'embedder.pth'))
            print('Use pretrained features.')
    # 根据放大倍率类型，使用相应的特征提取函数来处理图像
    if args.magnification == 'tree' or args.magnification == 'low' or args.magnification == 'high' :
        # bags_path = os.path.join('datasets', args.dataset, '*', '*')
        bags_path = os.path.join('WSI', args.dataset, 'pyramid512_20x_5x', '*', '*')
    else:
        bags_path = os.path.join('datasets', args.dataset, 'single', '*', '*')
    feats_path = os.path.join('datasets', 'WSILabel_csmil')
    # feats_path = os.path.join('datasets', args.dataset)
        
    os.makedirs(feats_path, exist_ok=True)
    bags_list = glob.glob(bags_path)
    
    if args.magnification == 'tree':
        compute_tree_feats(args, bags_list, i_classifier_l, i_classifier_h, feats_path)
    else:
        compute_feats(args, bags_list, i_classifier, feats_path, args.magnification)
    n_classes = glob.glob(os.path.join('datasets', args.dataset, '*'+os.path.sep))
    n_classes = sorted(n_classes)
    all_df = []
    # 最后，代码将提取到的特征保存在指定文件夹下，生成一个带标签的数据集，方便后续的特征训练。
    for i, item in enumerate(n_classes):
        bag_csvs = glob.glob(os.path.join(item, '*.csv'))
        bag_df = pd.DataFrame(bag_csvs)
        bag_df['label'] = i
        bag_df.to_csv(os.path.join('datasets', 'WSILabel_csmil', item.split(os.path.sep)[2]+'.csv'), index=False)
        # bag_df.to_csv(os.path.join('datasets', args.dataset, item.split(os.path.sep)[2]+'.csv'), index=False)
        all_df.append(bag_df)
    bags_path = pd.concat(all_df, axis=0, ignore_index=True)
    bags_path = shuffle(bags_path)
    bags_path.to_csv(os.path.join('WSI', 'WSILabel_csmil', 'WSILabel_csmil'+'.csv'), index=False)

if __name__ == '__main__':
    main()