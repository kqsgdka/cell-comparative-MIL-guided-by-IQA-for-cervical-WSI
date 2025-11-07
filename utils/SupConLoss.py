# '''
# Version: V1.0.0
# Author: kll 3327175694@qq.com
# Date: 2025-03-24 11:02:20
# LastEditors: kll 3327175694@qq.com
# LastEditTime: 2025-03-26 09:51:09
# FilePath: SupConLoss.py
# Copyright 2025 Marvin, All Rights Reserved. 
# Description: 
# '''
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class SupConLoss(nn.Module):
#     def __init__(self, temperature=0.05):
#         super(SupConLoss, self).__init__()
#         self.temperature = temperature

#     def forward(self, features, labels=None, mask=None):
#         """
#         计算监督对比损失 (Supervised Contrastive Loss)

#         Args:
#             features: 特征张量，形状为 [batch_size, n_views, feature_dim]
#             labels: 标签张量，形状为 [batch_size]
#             mask: 自定义正样本对比掩码，形状为 [batch_size, batch_size]

#         Returns:
#             loss: 监督对比损失值
#         """
#         device = features.device
#         batch_size = features.shape[0]

#         # L2 归一化
#         features = F.normalize(features, dim=1)
#         # print("Normalized features:", features)
#         features = features.unsqueeze(1)  # 为二维张量添加一个新的维度

#         # 计算特征相似度矩阵 [N, N]
#         similarity_matrix = torch.matmul(features, features.transpose(1, 2)) / self.temperature

#         # 去掉对角线元素
#         mask_self = torch.eye(batch_size, dtype=torch.bool).to(device)
#         similarity_matrix.masked_fill_(mask_self.unsqueeze(1), -1e9)

#         # 计算正样本掩码
#         if labels is not None:
#             labels = labels.contiguous().view(-1, 1)
#             if labels.shape[0] != batch_size:
#                 raise ValueError("标签数量与批次大小不一致")
#             mask = torch.eq(labels, labels.T).float().to(device)
#         elif mask is None:
#             mask = torch.ones_like(similarity_matrix).to(device)

#         # 计算对比损失
#         similarity_matrix = torch.clamp(similarity_matrix, min=-10, max=10)  # 进行裁剪
#         exp_sim = torch.exp(similarity_matrix)
#         mask = mask.unsqueeze(1).repeat(1, features.shape[1], 1)

#         epsilon = 1e-10  # 小常数
#         log_prob = similarity_matrix - torch.log(torch.sum(exp_sim, dim=2, keepdim=True) + epsilon)
#         # log_prob = similarity_matrix - torch.log(torch.sum(exp_sim, dim=2, keepdim=True))
#         mean_log_prob = torch.sum(mask * log_prob, dim=2) / torch.sum(mask, dim=2)

#         # 计算最终损失
#         loss = -torch.mean(mean_log_prob)
#         return loss


# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class SupConBagLoss(nn.Module):
#     def __init__(self, temperature=0.1):
#         super(SupConBagLoss, self).__init__()
#         self.temperature = temperature

#     def forward(self, features, labels):
#         """
#         计算包级特征的监督对比损失

#         Args:
#             features: 包级特征张量，形状为 [batch_size, feature_dim]
#             labels: 包标签张量，形状为 [batch_size]

#         Returns:
#             loss: 监督对比损失值
#         """
#         device = features.device
#         batch_size = features.shape[0]

#         # L2 归一化
#         features = F.normalize(features, dim=1)

#         # 计算相似度矩阵 [N, N]
#         similarity_matrix = torch.matmul(features, features.T) / self.temperature

#         # 去掉对角线元素（自身相似性）
#         mask_self = torch.eye(batch_size, dtype=torch.bool).to(device)
#         similarity_matrix.masked_fill_(mask_self, -1e9)

#         # 计算正样本掩码
#         labels = labels.contiguous().view(-1, 1)
#         mask = torch.eq(labels, labels.T).float().to(device)

#         # 计算对比损失
#         exp_sim = torch.exp(similarity_matrix)
#         epsilon = 1e-10  # 小常数防止log(0)

#         # 计算对数概率
#         log_prob = similarity_matrix - torch.log(torch.sum(exp_sim, dim=1, keepdim=True) + epsilon)

#         # 计算掩码平均对数概率
#         mean_log_prob = torch.sum(mask * log_prob, dim=1) / torch.sum(mask, dim=1)

#         # 计算最终损失
#         loss = -torch.mean(mean_log_prob)
#         return loss




import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConBagLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConBagLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        计算监督对比损失 (Supervised Contrastive Loss)

        Args:
            features: 特征张量，形状为 [batch_size, num_instances, feature_dim]
            labels: 标签张量，形状为 [batch_size]

        Returns:
            loss: 监督对比损失值
        """
        device = features.device
        batch_size, num_instances, feature_dim = features.shape

        # L2 归一化
        features = F.normalize(features, dim=2)

        # 扁平化特征用于相似度计算
        features = features.view(batch_size * num_instances, feature_dim)

        # 计算相似度矩阵
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        # 防止自身对比
        mask_self = torch.eye(batch_size * num_instances, dtype=torch.bool).to(device)
        similarity_matrix.masked_fill_(mask_self, -1e9)

        # 计算正样本掩码
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # 计算对比损失
        exp_sim = torch.exp(similarity_matrix)
        epsilon = 1e-10
        log_prob = similarity_matrix - torch.log(torch.sum(exp_sim, dim=1, keepdim=True) + epsilon)

        # 计算正样本对的平均对数概率
        mean_log_prob = torch.sum(mask * log_prob, dim=1) / torch.sum(mask, dim=1)

        # 计算最终损失
        loss = -torch.mean(mean_log_prob)
        return loss