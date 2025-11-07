import torch
import torch.nn as nn
import torch.nn.functional as F





class Attention(nn.Module):
    def __init__(self,feature_len=1536,out_chans=2,BatchNorm=False):
        super(Attention, self).__init__()
        #原始版是L=500 D=128 10月20日之后，实验改为了2048和1024
        self.L = 2048         
        self.D = 1024
        self.K = 1
        self.out_chans=out_chans
        self.BatchNorm=BatchNorm
        
        #这个需要根据特征提取器获得的特征长度来决定
        self.feature_len=feature_len
        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(self.feature_len, self.L),
            nn.ReLU(),
        )

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )
        if not self.BatchNorm:
            self.classifier = nn.Sequential(
                nn.Linear(self.L*self.K, self.out_chans),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(self.L*self.K, self.L*self.K),
                nn.LayerNorm(self.L*self.K),
                nn.ReLU(),
                nn.Linear(self.L*self.K, self.L*self.K),
                nn.LayerNorm(self.L*self.K),
                nn.ReLU(),
                nn.Linear(self.L*self.K, self.out_chans)
            )

    def forward(self, x):
        x = x.squeeze(0)

        H = self.feature_extractor_part2(x)  # NxL
        A = self.attention(H)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.matmul(A, H)  # KxL

        Y_prob = self.classifier(M)


        return Y_prob, A


class GatedAttention(nn.Module):
    def __init__(self,feature_len=1536,out_chans=2,BatchNorm=False):
        super(GatedAttention, self).__init__()
        self.L = 2048
        self.D = 1024
        self.K = 1
        self.out_chans=out_chans
        self.BatchNorm=BatchNorm

        #这个需要根据特征提取器获得的特征长度来决定
        self.feature_len=feature_len

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(self.feature_len, self.L),
            nn.ReLU(),
        )

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

        if not self.BatchNorm:
            self.classifier = nn.Sequential(
                nn.Linear(self.L*self.K, self.out_chans),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(self.L*self.K, self.L*self.K),
                nn.GroupNorm(num_groups=32,num_channels=self.L*self.K),
                nn.ReLU(),
                nn.Linear(self.L*self.K, self.L*self.K),
                nn.GroupNorm(num_groups=32,num_channels=self.L*self.K),
                nn.ReLU(),
                nn.Linear(self.L*self.K, self.out_chans)
            )

    def forward(self, x):
        x = x.squeeze(0)
        H = self.feature_extractor_part2(x)  # NxL

        A_V = self.attention_V(H)  # NxD
        A_U = self.attention_U(H)  # NxD
        A = self.attention_weights(A_V * A_U) # element wise multiplication # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.matmul(A, H)  # KxL

        Y_prob = self.classifier(M)

        return Y_prob, A




class NoisyAnd(nn.Module):
    def __init__(self,feature_len=1536,out_chans=2,BatchNorm=False):
        super(NoisyAnd, self).__init__()
        self.out_chans=out_chans
        self.BatchNorm=BatchNorm

        

        #这个需要根据特征提取器获得的特征长度来决定
        self.feature_len=feature_len
        self.b = torch.nn.Parameter(torch.tensor(0.01))
        self.sigmoid = nn.Sigmoid()

        if not self.BatchNorm:
            self.classifier = nn.Sequential(
                nn.Linear(self.feature_len, self.out_chans),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(self.feature_len, self.feature_len),
                nn.LayerNorm(self.feature_len),
                nn.ReLU(),
                nn.Linear(self.feature_len, self.feature_len),
                nn.LayerNorm(self.feature_len),
                nn.ReLU(),
                nn.Linear(self.feature_len, self.out_chans)
            )

    def forward(self, x):
        x = x.squeeze(0)

        a=x.shape[0]
        mean = torch.mean(x, 0,True)
        
        res = (self.sigmoid(a * (mean - self.b)) - self.sigmoid(-a * self.b)) / (               self.sigmoid(a * (1 - self.b)) - self.sigmoid(-a * self.b))

        Y_prob = self.classifier(res)

        return Y_prob, res


#这个在embeding 层次上做pooling MI=Net  在Instance score上做池化表示mi-net
class AveragePooling(nn.Module):
    def __init__(self,feature_len=1536,out_chans=2,BatchNorm=False):
        super(AveragePooling, self).__init__()
        self.out_chans=out_chans
        self.BatchNorm=BatchNorm

    
        #这个需要根据特征提取器获得的特征长度来决定
        self.feature_len=feature_len
        self.adapt_pooling=torch.nn.AdaptiveAvgPool1d(self.feature_len)
        if not self.BatchNorm:
            self.classifier = nn.Sequential(
                nn.Linear(self.feature_len, self.out_chans),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(self.feature_len, self.feature_len),
                nn.LayerNorm(self.feature_len),
                nn.ReLU(),
                nn.Linear(self.feature_len, self.feature_len),
                nn.LayerNorm(self.feature_len),
                nn.ReLU(),
                nn.Linear(self.feature_len, self.out_chans)
            )

    def forward(self, x):
        x = x.squeeze(0)
        x=torch.mean(x, 0,True)
        x=self.classifier(x)
        
        return x, x



#这个在embeding 层次上做pooling MI=Net  在Instance score上做池化表示mi-net
class AveragePooling_mi(nn.Module):
    def __init__(self,feature_len=1536,out_chans=2,BatchNorm=False):
        super(AveragePooling_mi, self).__init__()
        self.out_chans=out_chans
        self.BatchNorm=BatchNorm
     

        #这个需要根据特征提取器获得的特征长度来决定
        self.feature_len=feature_len
        self.adapt_pooling=torch.nn.AdaptiveAvgPool1d(self.feature_len)
        if not self.BatchNorm:
            self.classifier = nn.Sequential(
                nn.Linear(self.feature_len, self.out_chans),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(self.feature_len, self.feature_len),
                nn.LayerNorm(self.feature_len),
                nn.ReLU(),
                nn.Linear(self.feature_len, self.feature_len),
                nn.LayerNorm(self.feature_len),
                nn.ReLU(),
                nn.Linear(self.feature_len, self.out_chans)
            )


    def forward(self, x):
        x = x.squeeze(0)
        x=self.classifier(x)
        x=torch.mean(x, 0,True)
        return x, x



class MaxPooling_mi(nn.Module):
    def __init__(self,feature_len=1536,out_chans=2,BatchNorm=False):
        super(MaxPooling_mi, self).__init__()
        self.out_chans=out_chans
        self.BatchNorm=BatchNorm

        #这个需要根据特征提取器获得的特征长度来决定
        self.feature_len=feature_len
        self.adapt_pooling=torch.nn.AdaptiveAvgPool1d(self.feature_len)
        if not self.BatchNorm:
            self.classifier = nn.Sequential(
                nn.Linear(self.feature_len, self.out_chans),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(self.feature_len, self.feature_len),
                nn.LayerNorm(self.feature_len),
                nn.ReLU(),
                nn.Linear(self.feature_len, self.feature_len),
                nn.LayerNorm(self.feature_len),
                nn.ReLU(),
                nn.Linear(self.feature_len, self.out_chans)
            )


    def forward(self, x):
        x = x.squeeze(0)
        x=self.classifier(x)
        x,_=torch.max(x,0,True)
        
        return x, x


class MaxPooling(nn.Module):
    def __init__(self,feature_len=1536,out_chans=2,BatchNorm=False):
        super(MaxPooling, self).__init__()
        self.out_chans=out_chans
        self.BatchNorm=BatchNorm


        self.feature_len=feature_len
        self.adapt_pooling=torch.nn.AdaptiveAvgPool1d(self.feature_len)
        if not self.BatchNorm:
            self.classifier = nn.Sequential(
                nn.Linear(self.feature_len, self.out_chans),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(self.feature_len, self.feature_len),
                nn.LayerNorm(self.feature_len),
                nn.ReLU(),
                nn.Linear(self.feature_len, self.feature_len),
                nn.LayerNorm(self.feature_len),
                nn.ReLU(),
                nn.Linear(self.feature_len, self.out_chans)
            )


    def forward(self, x):
        x = x.squeeze(0)
        x,_=torch.max(x,0,True)
        x=self.classifier(x)
        
        return x, x
    



if __name__=="__main__":
    model=MaxPooling(BatchNorm=True)
    input=torch.zeros((1,20,1536))
    a,b=model(input)
    print(a)
    
