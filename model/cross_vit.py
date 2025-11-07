import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

# 这个两个函数用于检查一个值是否存在以及为参数提供默认值
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# pre-layernorm
'''
PreNorm类先对输入进行Layer Normalization，再将其传入后续的网络层
（如Attention或FeedForward）。这个操作使模型在处理过程中保持数值稳定性
'''
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# feedforward
'''
FeedForward网络用于对输入进行非线性变换，常用于Transformer中的中间层。通过两个线性层和GELU激活函数来实现，带有dropout来防止过拟合'''
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

# attention

'''
实现了多头注意力机制。输入特征通过线性层投影为查询（Q）、键（K）和值（V），然后通过点积计算注意力权重，进而加权求和得到输出。多头机制允许模型同事关注不同的特征。'''
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context = None, kv_include_self = False):
        b, n, _, h = *x.shape, self.heads
        context = default(context, x)

        if kv_include_self:
            context = torch.cat((x, context), dim = 1) # cross attention requires CLS token includes itself as key / value

        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# transformer encoder, for small and large patches

'''
是标准的Transformer编码器，由多个Transformer层组成。每个Transformer层包含一个多头注意力机制和一个前馈网络。depth参数控制了编码器的层数。'''
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

# projecting CLS tokens, in the case that small and large patch tokens have different dimensions

class ProjectInOut(nn.Module):
    def __init__(self, dim_in, dim_out, fn):
        super().__init__()
        self.fn = fn

        need_projection = dim_in != dim_out
        self.project_in = nn.Linear(dim_in, dim_out) if need_projection else nn.Identity()
        self.project_out = nn.Linear(dim_out, dim_in) if need_projection else nn.Identity()

    def forward(self, x, *args, **kwargs):
        x = self.project_in(x)
        x = self.fn(x, *args, **kwargs)
        x = self.project_out(x)
        return x

# cross attention transformer

'''
交叉注意力机制
用于处理不同尺度的图像（大块和小块），它包含两个相互交叉的注意力机制：小块图像作为查询，大图块图像作为键和值
这种设计允许模型在不同尺度的图像间共享信息'''
class CrossTransformer(nn.Module):
    def __init__(self, sm_dim, lg_dim, depth, heads, dim_head, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                ProjectInOut(sm_dim, lg_dim, PreNorm(lg_dim, Attention(lg_dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                ProjectInOut(lg_dim, sm_dim, PreNorm(sm_dim, Attention(sm_dim, heads = heads, dim_head = dim_head, dropout = dropout)))
            ]))

    def forward(self, sm_tokens, lg_tokens):
        (sm_cls, sm_patch_tokens), (lg_cls, lg_patch_tokens) = map(lambda t: (t[:, :1], t[:, 1:]), (sm_tokens, lg_tokens))

        for sm_attend_lg, lg_attend_sm in self.layers:
            sm_cls = sm_attend_lg(sm_cls, context = lg_patch_tokens, kv_include_self = True) + sm_cls
            lg_cls = lg_attend_sm(lg_cls, context = sm_patch_tokens, kv_include_self = True) + lg_cls

        sm_tokens = torch.cat((sm_cls, sm_patch_tokens), dim = 1)
        lg_tokens = torch.cat((lg_cls, lg_patch_tokens), dim = 1)
        return sm_tokens, lg_tokens


# multi-scale encoder
'''
结合了不同尺度的编码器（小块和大块)以及交叉注意力机制，通过多层次编码来增强特征提取能力
'''
class MultiScaleEncoder(nn.Module):
    def __init__(
        self,
        *,
        depth,
        sm_dim,
        lg_dim,
        sm_enc_params,
        lg_enc_params,
        cross_attn_heads,
        cross_attn_depth,
        cross_attn_dim_head = 64,
        dropout = 0.
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Transformer(dim = sm_dim, dropout = dropout, **sm_enc_params),
                Transformer(dim = lg_dim, dropout = dropout, **lg_enc_params),
                CrossTransformer(sm_dim = sm_dim, lg_dim = lg_dim, depth = cross_attn_depth, heads = cross_attn_heads, dim_head = cross_attn_dim_head, dropout = dropout)
            ]))

    def forward(self, sm_tokens, lg_tokens):
        for sm_enc, lg_enc, cross_attend in self.layers:
            sm_tokens, lg_tokens = sm_enc(sm_tokens), lg_enc(lg_tokens)
            sm_tokens, lg_tokens = cross_attend(sm_tokens, lg_tokens)

        return sm_tokens, lg_tokens

# patch-based image to token embedder
'''
ImageEmbedder将输入图像拆分为多个小块，并将每个小块映射为特征向量，随后再加上位置信息。这一过程用于将2D图像转换为Transformer可以处理的1D token序列。'''
class ImageEmbedder(nn.Module):
    def __init__(
        self,
        *,
        dim,
        image_size,
        patch_size,
        dropout = 0.
    ):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size ** 2

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]

        return self.dropout(x)

# cross ViT class
'''
CrossViT是整个模型的核心部分，结合了小块和大块的图像嵌入器以及多尺度编码器。通过最后的分类头，将提取到的全局特征用于分类任务。'''
class CrossViT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        num_classes,
        sm_dim,
        lg_dim,
        sm_patch_size = 12,
        sm_enc_depth = 1,
        sm_enc_heads = 8,
        sm_enc_mlp_dim = 2048,
        sm_enc_dim_head = 64,
        lg_patch_size = 16,
        lg_enc_depth = 4,
        lg_enc_heads = 8,
        lg_enc_mlp_dim = 2048,
        lg_enc_dim_head = 64,
        cross_attn_depth = 2,
        cross_attn_heads = 8,
        cross_attn_dim_head = 64,
        depth = 3,
        dropout = 0.1,
        emb_dropout = 0.1
    ):
        super().__init__()
        self.sm_image_embedder = ImageEmbedder(dim = sm_dim, image_size = image_size, patch_size = sm_patch_size, dropout = emb_dropout)
        self.lg_image_embedder = ImageEmbedder(dim = lg_dim, image_size = image_size, patch_size = lg_patch_size, dropout = emb_dropout)

        self.multi_scale_encoder = MultiScaleEncoder(
            depth = depth,
            sm_dim = sm_dim,
            lg_dim = lg_dim,
            cross_attn_heads = cross_attn_heads,
            cross_attn_dim_head = cross_attn_dim_head,
            cross_attn_depth = cross_attn_depth,
            sm_enc_params = dict(
                depth = sm_enc_depth,
                heads = sm_enc_heads,
                mlp_dim = sm_enc_mlp_dim,
                dim_head = sm_enc_dim_head
            ),
            lg_enc_params = dict(
                depth = lg_enc_depth,
                heads = lg_enc_heads,
                mlp_dim = lg_enc_mlp_dim,
                dim_head = lg_enc_dim_head
            ),
            dropout = dropout
        )

        self.sm_mlp_head = nn.Sequential(nn.LayerNorm(sm_dim), nn.Linear(sm_dim, num_classes))
        self.lg_mlp_head = nn.Sequential(nn.LayerNorm(lg_dim), nn.Linear(lg_dim, num_classes))

    def forward(self, img):
        sm_tokens = self.sm_image_embedder(img)
        lg_tokens = self.lg_image_embedder(img)

        sm_tokens, lg_tokens = self.multi_scale_encoder(sm_tokens, lg_tokens)

        sm_cls, lg_cls = map(lambda t: t[:, 0], (sm_tokens, lg_tokens))

        sm_logits = self.sm_mlp_head(sm_cls)
        lg_logits = self.lg_mlp_head(lg_cls)

        return sm_logits + lg_logits
    
    


# cross ViT  feature class
'''CrossViT_feature_single 类实现了对比学习机制，使用交叉熵损失和余弦相似度损失来优化分类和特征空间的区分能力。
模型希望通过减少同类样本之间的距离、增大不同类样本的距离，从而增强模型的判别能力'''

class ImageEmbedder_feature(nn.Module):
    def __init__(
        self,
        patch_dim,
        dim,
        num_patches,
        dropout = 0.
    ):
        super().__init__()


        self.to_patch_embedding=nn.Linear(patch_dim, dim)


        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]

        return self.dropout(x)






class CrossViT_feature(nn.Module):
    def __init__(
        self,
        num_classes,
        patch_dim,
        dim=768,
        num_patches=20,
        enc_depth = 2,
        enc_heads = 8,
        enc_mlp_dim = 2048,
        enc_dim_head = 64,
        cross_attn_depth = 2,
        cross_attn_heads = 8,
        cross_attn_dim_head = 64,
        depth = 3,
        dropout = 0.1,
    ):
        super().__init__()
        self.sm_image_embedder =ImageEmbedder_feature(patch_dim,dim,num_patches=int(num_patches/2))
        self.lg_image_embedder = ImageEmbedder_feature(patch_dim,dim,num_patches=int(num_patches/2))

        self.multi_scale_encoder = MultiScaleEncoder(
            depth = depth,
            sm_dim = dim,
            lg_dim = dim,
            cross_attn_heads = cross_attn_heads,
            cross_attn_dim_head = cross_attn_dim_head,
            cross_attn_depth = cross_attn_depth,
            sm_enc_params = dict(
                depth = enc_depth,
                heads = enc_heads,
                mlp_dim = enc_mlp_dim,
                dim_head = enc_dim_head
            ),
            lg_enc_params = dict(
                depth = enc_depth,
                heads = enc_heads,
                mlp_dim = enc_mlp_dim,
                dim_head = enc_dim_head
            ),
            dropout = dropout
        )

        self.sm_mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))
        self.lg_mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, x):
        x1,x2=torch.chunk(x,2,dim=1)
        x1 = self.sm_image_embedder(x1)
        x2 = self.lg_image_embedder(x2)

        x1, x2 = self.multi_scale_encoder(x1, x2)

        x1_cls, x2_cls = map(lambda t: t[:, 0], (x1, x2))

        x1_cls = self.sm_mlp_head(x1_cls)
        x2_cls = self.lg_mlp_head(x2_cls)

        return x1_cls + x2_cls

# 区分正异常细胞。使用了对比学习和交叉注意力机制来增强特征的表示能力
## 最后输出WSL结果的时候，仅依靠单个检测头的一个输出结果，不是两个。
class CrossViT_feature_singele(nn.Module):
    def __init__(
        self,
        num_classes,
        patch_dim,
        dim=768,
        num_patches=20,
        enc_depth = 2,
        enc_heads = 8,
        enc_mlp_dim = 2048,
        enc_dim_head = 64,
        cross_attn_depth = 2,
        cross_attn_heads = 8,
        cross_attn_dim_head = 64,
        depth = 3,
        dropout = 0.1,
    ):
        super().__init__()
        self.sm_image_embedder =ImageEmbedder_feature(patch_dim,dim,num_patches=int(num_patches/2))
        self.lg_image_embedder = ImageEmbedder_feature(patch_dim,dim,num_patches=int(num_patches/2))

        self.multi_scale_encoder = MultiScaleEncoder(
            depth = depth,
            sm_dim = dim,
            lg_dim = dim,
            cross_attn_heads = cross_attn_heads,
            cross_attn_dim_head = cross_attn_dim_head,
            cross_attn_depth = cross_attn_depth,
            sm_enc_params = dict(
                depth = enc_depth,
                heads = enc_heads,
                mlp_dim = enc_mlp_dim,
                dim_head = enc_dim_head
            ),
            lg_enc_params = dict(
                depth = enc_depth,
                heads = enc_heads,
                mlp_dim = enc_mlp_dim,
                dim_head = enc_dim_head
            ),
            dropout = dropout
        )

        self.sm_mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))
        self.lg_mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))
        
        
        # contrastive learning temperature
        self.CrossEntropyLoss=nn.CrossEntropyLoss()
        self.dist_loss=nn.CosineEmbeddingLoss(margin=0.1)
        self.temperature = nn.Parameter(torch.Tensor([1.]))

    def forward(self, x):
        x1,x2=torch.chunk(x,2,dim=1)
        x1 = self.sm_image_embedder(x1)
        x2 = self.lg_image_embedder(x2)

        x1, x2 = self.multi_scale_encoder(x1, x2)

        x1_cls, x2_cls = map(lambda t: t[:, 0], (x1, x2))

        x1_prd = self.sm_mlp_head(x1_cls)
        x2_prd = self.lg_mlp_head(x2_cls)
        return x1_prd,x2_prd,x1_cls, x2_cls

    
    def return_loss(self,x,label):
        x1_prd,x2_prd,x1_cls, x2_cls=self.forward(x)
        
        class_loss=self.CrossEntropyLoss(x1_prd,label)
        
        # calculate contrastive loss

        batch, device = x.shape[0], x.device
        sim = torch.einsum('i d, j d -> i j', x1_cls, x2_cls)
        sim = sim * self.temperature.exp()
        contrastive_labels = torch.arange(batch, device=device)

        contrastive_loss = (self.CrossEntropyLoss(sim, contrastive_labels) + self.CrossEntropyLoss(sim.t(), contrastive_labels)) * 0.5
        
        
        # 我们希望正常细胞和异常细胞的相似度能够小一些，所以对比损失要期望值越大越好，因此使用减号
        return class_loss-0.5*contrastive_loss,x1_prd
    
    def return_loss_dist(self,x,label):
        x1_prd,x2_prd,x1_cls, x2_cls=self.forward(x)
        
        class_loss=self.CrossEntropyLoss(x1_prd,label)
        
        # 计算预先相似度
        dist_labels = torch.zeros(x1_cls.shape[0], device=x1_cls.device)-1

        
        
        dis_loss=self.dist_loss(x1_cls,x2_cls,dist_labels)
        

        # 我们希望正常细胞和异常细胞的相似度能够小一些，所以对比损失要期望值越大越好，因此使用减号
        return class_loss+0.5*dis_loss,x1_prd
    
    def return_loss_two(self,x,label):
        x1_prd,x2_prd,x1_cls, x2_cls=self.forward(x)
        
        class_loss_abnormal=self.CrossEntropyLoss(x1_prd,label)
        
        normal_labels = torch.zeros_like(label,device=x.device)
        class_loss_normal=self.CrossEntropyLoss(x2_prd,normal_labels)
        
        
        
        # 我们希望正常细胞和异常细胞的相似度能够小一些，所以对比损失要期望值越大越好，因此使用减号
        return class_loss_abnormal+class_loss_normal*0.5,x1_prd
    
    def return_loss_two_dist(self,x,label):
        x1_prd,x2_prd,x1_cls, x2_cls=self.forward(x)
        
        class_loss_abnormal=self.CrossEntropyLoss(x1_prd,label)
        
        normal_labels = torch.zeros_like(label,device=x.device)
        class_loss_normal=self.CrossEntropyLoss(x2_prd,normal_labels)
        
        # 计算预先相似度


        dist_labels = torch.zeros(x1_cls.shape[0], device=x1_cls.device)-1

        
        
        dis_loss=self.dist_loss(x1_cls,x2_cls,dist_labels)
        

        # 我们希望正常细胞和异常细胞的相似度能够小一些，所以对比损失要期望值越大越好，因此使用减号
        return class_loss_abnormal+class_loss_normal*0.5+dis_loss*0.5,x1_prd
        
        

        

if __name__=="__main__":
    from torch.cuda.amp import autocast
    model = CrossViT_feature_singele(patch_dim=1536,num_patches=20,num_classes=2,dim=384)
    inputs=torch.zeros((5,20,1536))
    print(inputs.shape)
    with autocast():
        a,b=model.return_loss_two_dist(inputs,torch.LongTensor([1,1,1,1,1]))
    print(a,b)
    a.backward()
