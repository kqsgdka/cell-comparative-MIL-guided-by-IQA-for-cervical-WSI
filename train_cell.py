import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms.functional as VF
from torchvision import transforms
import timm
import timm.scheduler

import sys
import argparse
import os
import copy
import itertools
import glob
import datetime
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             roc_auc_score, roc_curve,f1_score,auc)
from sklearn.datasets import load_svmlight_file
from collections import OrderedDict
from torch.autograd import Variable
import random
import torch.nn.functional as F
from utils.SupConLoss import SupConBagLoss
import matplotlib.pyplot as plt


def get_bag_feats_v2(csv_file_df, args):
    if args.dataset == 'TCGA-lung-default':
        feats_csv_path = 'datasets/tcga-dataset/tcga_lung_data_feats/' + csv_file_df.iloc[0].split('/')[1] + '.csv'
    else:
        feats_csv_path = csv_file_df.strip().split(',')[0]
    if feats_csv_path == '0':
        print("feats_csv_path is 0")
        print(csv_file_df)
    # 读取 abnormal 和 normal 的特征
    feats_csv_path_abnormal = os.path.join("/home75/kanglanlan/WSI/code/dsmil-wsi-cervical", feats_csv_path, 'abnormal.csv')
    feats_csv_path_normal = os.path.join("/home75/kanglanlan/WSI/code/dsmil-wsi-cervical", feats_csv_path, 'normal.csv')
    if os.path.getsize(feats_csv_path_abnormal) > 1:
        df_abnormal = pd.read_csv(feats_csv_path_abnormal)
        feats_abnormal = shuffle(df_abnormal).reset_index(drop=True)
        feats_abnormal = feats_abnormal.to_numpy()
        feats_abnormal = torch.from_numpy(feats_abnormal).to(torch.float64)
    else:
        feats_abnormal = None
    if os.path.getsize(feats_csv_path_normal) > 1:
        df_normal = pd.read_csv(feats_csv_path_normal)
        feats_normal = shuffle(df_normal).reset_index(drop=True)
        feats_normal = feats_normal.to_numpy()
        feats_normal = torch.from_numpy(feats_normal).to(torch.float64)
    else:
        feats_normal = None        
        
    label = np.zeros(1)
    if args.num_classes==1:
        label[0] = csv_file_df.iloc[1]
    else:
        label[0] = csv_file_df.strip().split(',')[-1]
        
    return label, feats_abnormal, feats_normal


def train(train_feats, milnet, bag_criterion, supcon_criterion, optimizer, args):
    # 将模型设为训练模式
    milnet.train()
    # 初始化总损失为0
    total_loss = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for i in range(len(train_feats)):
        optimizer.zero_grad()  # 将优化器的梯度缓存清零，避免梯度累积
        bag_label, bag_feats_abnormal, bag_feats_normal = get_bag_feats_v2(train_feats[i], args)   # 获取每个样本的标签和特征
        if bag_feats_abnormal is None:
                pass
        else:
            bag_feats_abnormal = bag_feats_abnormal.float()
            bag_feats_abnormal = bag_feats_abnormal.view(-1, args.feats_size).cuda()   # 将特征调整为指定的大小（通常是二维）
        if bag_feats_normal is None:
            pass
        else:
            bag_feats_normal = bag_feats_normal.float()
            bag_feats_normal = bag_feats_normal.view(-1, args.feats_size).cuda()
        bag_label = torch.tensor(bag_label).cuda()  # 将标签转换为张量
        if bag_feats_abnormal is not None:
            bag_feats = torch.cat([bag_feats_abnormal, bag_feats_normal], dim=0)
            bag_feats = bag_feats.view(-1, args.feats_size).cuda()
        else:
            bag_feats = bag_feats_normal
            bag_feats = bag_feats.view(-1, args.feats_size).cuda()


        if args.model == 'dsmil':
            # refer to dsmil code
            ins_prediction, bag_prediction, _, _ = milnet(bag_feats)
            max_prediction, _ = torch.max(ins_prediction, 0)
            bag_loss = bag_criterion(bag_prediction, bag_label.long())

            # 如果使用交叉熵需要更改一下
            max_prediction=max_prediction.reshape(1,-1)

            max_loss = bag_criterion(max_prediction, bag_label.long())
            loss = 0.5 * bag_loss + 0.5 * max_loss
        
        elif args.model == 'cellCompareMIL':
            bag_prediction,instance_feats,_ = milnet(bag_feats_abnormal, bag_feats_normal)
            # print("output:",bag_prediction)
            bag_loss = bag_criterion(bag_prediction, bag_label.long())
            supcon_loss = supcon_criterion(instance_feats, bag_label.long())
            # 综合损失
            log_vars = torch.nn.Parameter(torch.zeros(2))
            loss = torch.exp(log_vars[0]) * bag_loss + log_vars[0] + torch.exp(log_vars[1]) * supcon_loss + log_vars[1]

            # #test
            # print("bag_prediction shape:", bag_prediction.shape)
            # print("bag_label shape:", bag_label.shape)

        elif args.model == 'abmil' or args.model == 'TransMIL' or args.model == 'RNN' or  args.model == "DT_MIL":
            bag_prediction = milnet(bag_feats)
            loss = bag_criterion(bag_prediction, bag_label.long())
        elif args.model == "trans_attention_MIL":
            bag_prediction,_ = milnet(bag_feats)
            loss = bag_criterion(bag_prediction, bag_label.long())
        elif args.model == "Snuffy":
            ins_prediction, bag_prediction, attentions = milnet(bag_feats)
            loss = bag_criterion(bag_prediction, bag_label.long())
        elif args.model == "DGMIL":
            bag_prediction = milnet(bag_feats)
            bag_label = bag_label.repeat(bag_prediction.shape[0])
            loss = bag_criterion(bag_prediction, bag_label.long())
        elif args.model == "PHIM_MIL":
            bag_prediction, attn_scores, fused_feats = milnet(bag_feats)
            # print("output:",output)
            loss = bag_criterion(bag_prediction, bag_label.long())
        elif args.model == "PSA_MIL":
            bag_prediction, diversity_loss = milnet(bag_feats)
            # print("output:",output)
            classification_loss = bag_criterion(bag_prediction, bag_label.long())
            loss = classification_loss + 0.1 * diversity_loss

        loss.backward()
        optimizer.step()
        total_loss = total_loss + loss.item()
        sys.stdout.write('\r Training bag [%d/%d] bag loss: %.4f' % (i, len(train_feats), loss.item()))
    return total_loss / len(train_feats)

# 用于在特征数据中进行随机采样和填充，执行patch dropout。
def dropout_patches(feats, p):
    idx = np.random.choice(np.arange(feats.shape[0]), int(
        feats.shape[0] * (1 - p)), replace=False)
    sampled_feats = np.take(feats, idx, axis=0)
    pad_idx = np.random.choice(
        np.arange(sampled_feats.shape[0]), int(feats.shape[0] * p), replace=False)
    pad_feats = np.take(sampled_feats, pad_idx, axis=0)
    sampled_feats = np.concatenate((sampled_feats, pad_feats), axis=0)
    return sampled_feats

# 测试过程，通过MIL模型的预测值与真实标签比对，计算不同评价指标（如AUC、准确率等）
'''
	对每个样本提取标签和特征，根据模型类型进行预测并计算损失。
	保存预测结果和标签信息以便后续使用。
	使用指标（AUC、准确率、F1等）进行性能评估'''
def test(test_df, milnet, bag_criterion, supcon_criterion, args):
    milnet.eval()
    total_loss = 0
    test_labels = []
    test_predictions = []
    test_score=[]
    save_data_list=[]
    
    with torch.no_grad():
        for i in range(len(test_df)):
            bag_label, bag_feats_abnormal, bag_feats_normal = get_bag_feats_v2(test_df[i], args)   # 获取每个样本的标签和特征
            if bag_feats_abnormal is None:
                pass
            else:
                bag_feats_abnormal = bag_feats_abnormal.float()
                bag_feats_abnormal = bag_feats_abnormal.view(-1, args.feats_size).cuda()   # 将特征调整为指定的大小（通常是二维）
            if bag_feats_normal is None:
                pass
            else:
                bag_feats_normal = bag_feats_normal.float()
                bag_feats_normal = bag_feats_normal.view(-1, args.feats_size).cuda()   # 将特征调整为指定的大小（通常是二维）
            bag_label = torch.tensor(bag_label).cuda()
            if bag_feats_abnormal is not None:
                bag_feats = torch.cat([bag_feats_abnormal, bag_feats_normal], dim=0)
                bag_feats = bag_feats.view(-1, args.feats_size).cuda()
            else:
                bag_feats = bag_feats_normal
                bag_feats = bag_feats.view(-1, args.feats_size).cuda()
            save_data={}

            if args.model == 'dsmil':
                ins_prediction, bag_prediction, _, _ = milnet(bag_feats)
                max_prediction, _ = torch.max(ins_prediction, 0)
                # 如果使用交叉熵需要更改一下
                max_prediction=max_prediction.reshape(1,-1)

                bag_loss = bag_criterion(bag_prediction, bag_label.long())
                max_loss = bag_criterion(max_prediction, bag_label.long())

                loss = 0.5 * bag_loss + 0.5 * max_loss
            elif args.model == 'cellCompareMIL':
                # bag_prediction,_ = milnet(bag_feats_abnormal, bag_feats_normal)
                # loss = bag_criterion(bag_prediction, bag_label.long())
                bag_prediction,instance_feats,_ = milnet(bag_feats_abnormal, bag_feats_normal)
                bag_loss = bag_criterion(bag_prediction, bag_label.long())
                supcon_loss = supcon_criterion(instance_feats, bag_label.long())
                # 综合损失
                log_vars = torch.nn.Parameter(torch.zeros(2))
                loss = torch.exp(log_vars[0]) * bag_loss + log_vars[0] + torch.exp(log_vars[1]) * supcon_loss + log_vars[1]
            elif args.model == 'abmil' or args.model == 'TransMIL' or args.model == 'RNN' or args.model == "DT_MIL":
                bag_prediction = milnet(bag_feats)
                loss = bag_criterion(bag_prediction, bag_label.long())
            elif args.model == "trans_attention_MIL":
                bag_prediction,_ = milnet(bag_feats)
                loss = bag_criterion(bag_prediction, bag_label.long())
            elif args.model == "Snuffy":
                ins_prediction, bag_prediction, attentions = milnet(bag_feats)
                loss = bag_criterion(bag_prediction, bag_label.long())
            elif args.model == "DGMIL":
                bag_prediction = milnet(bag_feats)
                bag_label = bag_label.repeat(bag_prediction.shape[0])
                loss = bag_criterion(bag_prediction, bag_label.long())
            elif args.model == "PHIM_MIL":
                bag_prediction, attn_scores, fused_feats = milnet(bag_feats)
                loss = bag_criterion(bag_prediction, bag_label.long())

            total_loss = total_loss + loss.item()
            sys.stdout.write('\r Testing bag [%d/%d] bag loss: %.4f' % (i, len(test_df), loss.item()))
            test_labels.extend(bag_label.cpu().numpy().tolist())
            if args.average:
                bag_prediction=0.5 * max_prediction + 0.5 * bag_prediction
                _, predicted = torch.max(bag_prediction.data, 1)
                test_predictions.extend(predicted.cpu().numpy().tolist())
                test_score.extend(F.softmax(bag_prediction, dim=1)[:, 1].cpu().numpy().tolist())
            else:
                _, predicted = torch.max(bag_prediction.data, 1)
                test_predictions.extend(predicted.cpu().numpy().tolist())
                test_score.extend(F.softmax(bag_prediction, dim=1)[:, 1].cpu().numpy().tolist())
            save_data["bag_label"]=bag_label.cpu().numpy().tolist()
            save_data["bag_prediction"]=bag_prediction.cpu().numpy().tolist()
            save_data_list.append(save_data)


                

    test_labels = np.array(test_labels)
    test_labels = test_labels.reshape(len(test_labels), -1)
    test_score = np.array(test_score)   
    # test_predictions = np.array(test_predictions)
    print("test_label shape:", test_labels.shape)
    print("test_score shape:", test_score.shape)


    #绘制ROC曲线并计算AUC
    fpr, tpr, thresholds = roc_curve(test_labels, test_score)
    roc_auc = auc(fpr, tpr)

    AUC = roc_auc_score(test_labels, test_score)  # 仅仅计算一类的auc
    p=precision_score(test_labels, test_predictions, average='macro')
    r=recall_score(test_labels, test_predictions, average='macro')
    acc=accuracy_score(test_labels, test_predictions)
    f1=f1_score(test_labels, test_predictions, average='macro')
    avg = np.mean([p, r, acc])
    test_loss = total_loss / len(test_df)
    return p, r, acc, avg, f1,AUC, test_loss, save_data_list,fpr,tpr,roc_auc
    



def main():
    parser = argparse.ArgumentParser(description='Train DSMIL on 20x patch features learned by SimCLR')
    parser.add_argument('--num_classes', default=2, type=int, help='Number of output classes [2]')
    parser.add_argument('--num_heads', default=4, type=int)
    parser.add_argument('--mlp_multiplier', default=4, type=int, help='inverted mlp anti-bottbleneck')
    parser.add_argument('--activation', default='relu', type=str, help='activation function used in semi transforer')
    parser.add_argument('--encoder_dropout', default=0.1, type=float, help='dropout in encoder')
    parser.add_argument('--depth', default=1, type=int, help="depth of transformer = number of encoder blocks")
    parser.add_argument('--random_patch_share', default=0.5, type=float, help='dropout in encoder')
    parser.add_argument('--big_lambda', default=200, type=int, help='top k')
    parser.add_argument(
        '--weight_init__weight_init_i__weight_init_b',
        default=['xavier_normal', 'xavier_normal', 'xavier_normal'],
        help='weight initialization')


    parser.add_argument('--feats_size', default=512, type=int, help='Dimension of the feature size [512]')
    # parser.add_argument('--feats_size', default=2048, type=int, help='Dimension of the feature size [512]')
    parser.add_argument('--lr', default=0.0002, type=float, help='Initial learning rate [0.0002]')
    parser.add_argument('--num_epochs', default=200, type=int, help='Number of total training epochs [40|200]')
    parser.add_argument('--gpu_index', type=int, nargs='+', default=(3,), help='GPU ID(s) [0]')
    parser.add_argument('--weight_decay', default=5e-3, type=float, help='Weight decay [5e-3]')
    parser.add_argument('--dataset', default='abnormalNormalCellFeature', type=str, help='pyramid20X_resnet18_ImageNet,pyramid20X_resnet18_moco,pyramid20X_resnet50_ImageNet_MoCo')
    # parser.add_argument('--dataset', default='pyramid20X_resnet50_ImageNet_MoCo', type=str, help='pyramid20X_resnet18_ImageNet,pyramid20X_resnet18_moco,pyramid20X_resnet50_ImageNet_MoCo')
    parser.add_argument('--split', default=0.2, type=float, help='Training/Validation split [0.2]')
    parser.add_argument('--model', default='cellCompareMIL', type=str, help='MIL model [dsmil]')
    # parser.add_argument('--model', default='PHIM_MIL', type=str, help='MIL model [dsmil]')
    parser.add_argument('--dropout_patch', default=0, type=float, help='Patch dropout rate [0]')
    parser.add_argument('--dropout_node', default=0, type=float, help='Bag classifier dropout rate [0]')
    parser.add_argument('--non_linearity', default=1, type=float, help='Additional nonlinear operation [0]')
    parser.add_argument('--data_root', required=False, default='datasets', type=str, help='path to data root')
    parser.add_argument(
        '--average',
        type=bool,
        default=False,
        help='Average the score of max-pooling and bag aggregating')

    args = parser.parse_args()
    # GPU设置
    gpu_ids = tuple(args.gpu_index)
    print(f'Using GPU: {gpu_ids}')
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in gpu_ids)
    torch.cuda.device_count()

    # 初始化模型
    if args.model == 'dsmil':
        import model.dsmil as mil
        i_classifier = mil.FCLayer(
        in_size=args.feats_size, out_size=args.num_classes).cuda()
        b_classifier = mil.BClassifier(input_size=args.feats_size,output_class=args.num_classes,dropout_v=args.dropout_node,nonlinear=args.non_linearity).cuda()
        milnet = mil.MILNet(i_classifier, b_classifier).cuda()
        state_dict_weights = torch.load('init.pth')
        try:
            milnet.load_state_dict(state_dict_weights, strict=False)
        except BaseException:
            del state_dict_weights['b_classifier.v.1.weight']
            del state_dict_weights['b_classifier.v.1.bias']
            milnet.load_state_dict(state_dict_weights, strict=False)
    
    elif args.model == 'cellCompareMIL':
        from model.cellCompareMIL import cellCompareMIL as mil
        milnet = mil(args.feats_size,args.num_classes).cuda()

    elif args.model == 'TransMIL':
        from model.TransMIL import TransMIL as mil
        milnet=mil(fears=args.feats_size,n_classes=args.num_classes).cuda()

    elif args.model == 'abmil':
        from model.abmil import Attention as mil
        milnet=mil(if_softmax=True,if_resnet50=False).cuda()
    elif args.model == 'RNN':
        from model.RNN_model_fratures import SimpleRNN
        milnet=SimpleRNN(args.feats_size,args.num_classes).cuda()
    elif args.model == "DT_MIL":
        from model.DT_MIL import PureTransformer
        milnet=PureTransformer(args.feats_size,args.num_classes).cuda()
        # milnet = nn.DataParallel(milnet, device_ids=gpu_ids)
    elif args.model == "trans_attention_MIL":
        from model.TransMIL_MIL import TransMIL
        milnet=TransMIL(args.feats_size,args.num_classes).cuda()
    elif args.model == "CrossViT_feature_singele":
        from model.CrossViT_feature_singele import CrossViT
    elif args.model == "DGMIL":
        from model.DGMIL import Linear_projection_MAE
        milnet=Linear_projection_MAE().cuda()
    elif args.model == "PHIM_MIL":
        from model.PHIM_MIL import PHIM_MIL
        milnet=PHIM_MIL(feature_dim=args.feats_size, proto_num=8, fusion_dim=args.feats_size, fusion_method='concat').cuda()
    elif args.model == "PSA_MIL":
        from model.PSA_MIL import PSA_MIL
        milnet=PSA_MIL(feature_dim=args.feats_size, embed_dim=args.feats_size, num_heads=args.num_heads, num_classes=args.num_classes).cuda()
    elif args.model == "Snuffy":
        import model.Snuffy as mil
        i_classifier = mil.FCLayer(in_size=args.feats_size,
                                      out_size=args.num_classes).cuda()
        c = copy.deepcopy
        attn = mil.MultiHeadedAttention(
            args.num_heads,
            args.feats_size,
        ).cuda()
        ff = mil.PositionwiseFeedForward(
            args.feats_size,
            args.feats_size * args.mlp_multiplier,
            args.activation,
            args.encoder_dropout
        ).cuda()
        b_classifier = mil.BClassifier(
            mil.Encoder(
                mil.EncoderLayer(
                    args.feats_size,
                    c(attn),
                    c(ff),
                    args.encoder_dropout,
                    args.big_lambda,
                    args.random_patch_share
                ), args.depth
            ),
            args.num_classes,
            args.feats_size
        ).cuda()
        # b_classifier = mil.BClassifier(input_size=args.feats_size,output_class=args.num_classes,dropout_v=args.dropout_node,nonlinear=args.non_linearity).cuda()
        milnet = mil.MILNet(i_classifier, b_classifier).cuda()
        init_funcs_registry = {
            'trunc_normal': nn.init.trunc_normal_,
            'kaiming_uniform': nn.init.kaiming_uniform_,
            'kaiming_normal': nn.init.kaiming_normal_,
            'xavier_uniform': nn.init.xavier_uniform_,
            'xavier_normal': nn.init.xavier_normal_,
            'orthogonal': nn.init.orthogonal_
        }
        modules = [(args.weight_init__weight_init_i__weight_init_b[1], 'i_classifier'),
                   (args.weight_init__weight_init_i__weight_init_b[2], 'b_classifier')]
        # print('modules:', modules)
        for init_func_name, module_name in modules:
            init_func = init_funcs_registry.get(init_func_name)
            # print('init_func:', init_func)
            for name, p in milnet.named_parameters():
                if p.dim() > 1 and name.split(".")[0] == module_name:
                    init_func(p)
        



    # 配置优化器和损失函数
    # bag_criterion = nn.CrossEntropyLoss()
    # 使用监督对比损失supcon
    supcon_criterion = SupConBagLoss().cuda()
    bag_criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.Adam(milnet.parameters(), lr=args.lr, betas=(
        0.5, 0.9), weight_decay=args.weight_decay)
    
    scheduler = timm.scheduler.CosineLRScheduler(optimizer, t_initial=args.num_epochs, lr_min=0.000005, warmup_lr_init=0.000001, warmup_t=5)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs, 0.000005)




    # loading the list of test data
    test_feats = open(f'{args.data_root}/{args.dataset}/test.csv', 'r').readlines()
    test_feats = np.array(test_feats)
    

    #loading the list of train data
    train_feats = open(f'{args.data_root}/{args.dataset}/train.csv', 'r').readlines()
    # 替换成自己的数据集
    # train_feats = open(f'{args.data_root}/{args.dataset}/train_2_class.txt', 'r').readlines()
    train_feats = np.array(train_feats)

    # path1 = os.path.join(args.data_root, args.dataset)
    # print(path1) # datasets/pyramid20X_resnet18_moco



    best_score = 0
    save_path = os.path.join('weights_pt', args.dataset,args.model,"交叉熵且阈值为0.5")

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(save_path,"test_results"), exist_ok=True)

    run = len(glob.glob(os.path.join(save_path, '*.pth')))

    # 记录训练参数
    with open(os.path.join(save_path, f"result_th{run}.txt"), "+a") as f:
        f.write(str(args))
        f.write("\n")


    with open(os.path.join(save_path, f"result{run}.csv"), "+a") as f:
        f.write(f"Epoch train_loss_bag test_loss_bag p r acc avg f1 AUC\n")
    

    for epoch in range(1, args.num_epochs):
        #打乱numpy
        train_feats = train_feats[np.random.permutation(len(train_feats))]

        data_loader1 = DataLoader(train_feats, batch_size=512, shuffle=True)

        # 调整 batch_size 为较小的值，例如 32
        data_loader2 = DataLoader(test_feats, batch_size=512, shuffle=True)

        train_loss_bag = train(train_feats, milnet, bag_criterion, supcon_criterion, optimizer, args)  # iterate all bags
        p, r, acc, avg, f1,AUC, test_loss,save_data_list,fpr,tpr,roc_auc = test(test_feats, milnet, bag_criterion, supcon_criterion, args)

        # # plt.figure()
        # # 绘制模型 ROC 曲线
        # plt.plot(fpr_model, tpr_model, color='darkcyan', lw=2, label=f'Our Method (AUC = {roc_auc_model:.4f})')
        # plt.plot(fpr_model, tpr_model, color='darkcyan', lw=2, label=f'Our Method (AUC = {roc_auc_model:.4f})')
        # plt.plot(fpr, tpr, color='blue', lw=2, label=f'Our method (AUC = 0.9366 ± 0.0065)')
        # plt.plot([0, 1], [0, 1], color='red', lw=1, linestyle='--')
        # plt.xlabel('1-Specificity')
        # plt.ylabel("Sensitivity")
        # plt.title(f'ROC curve')
        # plt.legend(loc="lower right")
        # plt.grid(alpha=0.3)
        # plt.savefig(os.path.join("./dsmil-wsi-cervical/weights_pt/abnormalNormalCellFeature/cellCompareMIL/交叉熵且阈值为0.5", f"ROC/{epoch+1}_ROC.png"))

        # print(f"Epoch [{epoch}/{args.num_epochs}] ,Loss: {train_loss_bag/len(train_feats):.4f}")
        print(f"Epoch [{epoch}/{args.num_epochs}]: "
          f"Train Loss: {train_loss_bag:.4f}, "
          f"Precision: {p:.4f}, "
          f"Recall: {r:.4f}, "
          f"Accuracy: {acc:.4f}, "
          f"Average: {avg:.4f}, "
          f"F1 Score: {f1:.4f}, "
          f"AUC: {AUC:.4f}, "
          f"Test Loss: {test_loss:.4f}")


        # 保存预测结果和label信息
        torch.save(save_data_list,os.path.join(save_path,f"test_results/{epoch}_test_result.pt"))



        with open(os.path.join(save_path, f"result{run}.csv"), "+a") as f:
            f.write(
                f"{epoch} {train_loss_bag} {test_loss} {p*100:.2f} {r*100:.2f} {acc*100:.2f} {avg*100:.2f} {f1*100:.2f} {AUC*100:.2f}\n")

        scheduler.step(epoch=epoch)
        current_score = acc
        if current_score >= best_score:
            best_score = current_score
            # 输出最佳的分数
            print(f"Saving best model with accuracy: {best_score:.4f}")
            save_name = os.path.join(save_path, str(run + 1) + '.pth')
            torch.save(milnet.state_dict(), save_name)

if __name__ == '__main__':
    main()
