
import torch
import torch.nn as nn


"""
https://github.com/szc19990412/TransMIL
TransMIL: Transformer based Correlated Multiple Instance Learning for Whole Slide Image Classification
"""
class SimpleRNN(nn.Module):
    def __init__(self,embedding_dim=1536, output_dim=2):
        super(SimpleRNN, self).__init__()
        #原始版是L=500 D=128 10月20日之后，实验改为了2048和1024
        self.hidden_dim=1024
        self.embedding_dim=embedding_dim
        self.rnn=nn.RNN(input_size =self.embedding_dim,hidden_size =self.hidden_dim,num_layers=1)
        self.fc = nn.Linear(self.hidden_dim, output_dim)
        self.layernormal=nn.LayerNorm(self.hidden_dim)
        

    def forward(self, x):
        # batch_size seq_len input_size-> seq_len, batch_size, input_size
        #x=x.permute(1,0,2)
        x=x.unsqueeze(0)
        x=torch.einsum("bse->sbe",x)

        x,ht=self.rnn(x)
        ht=torch.mean(ht,dim=0)
        ht=self.layernormal(ht)
        output=self.fc(ht)
        return output

    
class SimpleLSTM(nn.Module):
    def __init__(self,embedding_dim=1536, output_dim=2,dropout=False):
        super(SimpleLSTM, self).__init__()
        #原始版是L=500 D=128 10月20日之后，实验改为了2048和1024
        self.hidden_dim=1024
        self.embedding_dim=embedding_dim
        self.rnn=nn.LSTM(input_size =self.embedding_dim,hidden_size =self.hidden_dim,num_layers=1)
        self.fc = nn.Linear(self.hidden_dim, output_dim)
        self.layernormal=nn.LayerNorm(self.hidden_dim)
        if dropout:
            self.dropout=nn.Dropout(0.5)
        else:
            self.dropout=nn.Identity()
        

    def forward(self, x):
        # batch_size seq_len input_size-> seq_len, batch_size, input_size
        #x=x.permute(1,0,2)
        x=torch.einsum("bse->sbe",x)

        x,(ht,c)=self.rnn(x)
        ht=self.layernormal(ht)
        ht=torch.mean(ht,dim=0)
        

        output=self.fc(ht)
        return output


if __name__=="__main__":

    model=SimpleLSTM()
    inputs=torch.zeros((1,20,1536))
    a=model(inputs)
    print(a.shape)