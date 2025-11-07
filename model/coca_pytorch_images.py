import torch
from torch import einsum, nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# normalization
# they use layernorm without bias, something that pytorch does not offer


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

# residual


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


# rotary positional embedding
# https://arxiv.org/abs/2104.09864


class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, max_seq_len, *, device):
        seq = torch.arange(max_seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = einsum("i , j -> i j", seq, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)


def rotate_half(x):
    x = rearrange(x, "... (j d) -> ... j d", j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(pos, t):
    return (t * pos.cos()) + (rotate_half(t) * pos.sin())


# classic Noam Shazeer paper, except here they use SwiGLU instead of the more popular GEGLU for gating the feedforward
# https://arxiv.org/abs/2002.05202


class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


# parallel attention and feedforward with residual
# discovered by Wang et al + EleutherAI from GPT-J fame


class ParallelTransformerBlock(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8, ff_mult=4):
        super().__init__()
        self.norm = LayerNorm(dim)

        attn_inner_dim = dim_head * heads
        ff_inner_dim = dim * ff_mult
        self.fused_dims = (attn_inner_dim, dim_head, dim_head, (ff_inner_dim * 2))

        self.heads = heads
        self.scale = dim_head**-0.5
        self.rotary_emb = RotaryEmbedding(dim_head)

        self.fused_attn_ff_proj = nn.Linear(dim, sum(self.fused_dims), bias=False)
        self.attn_out = nn.Linear(attn_inner_dim, dim, bias=False)

        self.ff_out = nn.Sequential(
            SwiGLU(),
            nn.Linear(ff_inner_dim, dim, bias=False)
        )

        # for caching causal mask and rotary embeddings

        self.register_buffer("mask", None, persistent=False)
        self.register_buffer("pos_emb", None, persistent=False)

    def get_mask(self, n, device):
        if self.mask is not None and self.mask.shape[-1] >= n:
            return self.mask[:n, :n]

        mask = torch.ones((n, n), device=device, dtype=torch.bool).triu(1)
        self.register_buffer("mask", mask, persistent=False)
        return mask

    def get_rotary_embedding(self, n, device):
        if self.pos_emb is not None and self.pos_emb.shape[-2] >= n:
            return self.pos_emb[:n]

        pos_emb = self.rotary_emb(n, device=device)
        self.register_buffer("pos_emb", pos_emb, persistent=False)
        return pos_emb

    def forward(self, x, attn_mask=None):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        n, device, h = x.shape[1], x.device, self.heads

        # pre layernorm

        x = self.norm(x)

        # attention queries, keys, values, and feedforward inner

        q, k, v, ff = self.fused_attn_ff_proj(x).split(self.fused_dims, dim=-1)

        # split heads
        # they use multi-query single-key-value attention, yet another Noam Shazeer paper
        # they found no performance loss past a certain scale, and more efficient decoding obviously
        # https://arxiv.org/abs/1911.02150

        q = rearrange(q, "b n (h d) -> b h n d", h=h)

        # rotary embeddings

        positions = self.get_rotary_embedding(n, device)
        q, k = map(lambda t: apply_rotary_pos_emb(positions, t), (q, k))

        # scale

        q = q * self.scale

        # similarity

        sim = einsum("b h i d, b j d -> b h i j", q, k)

        # causal mask

        causal_mask = self.get_mask(n, device)
        sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        # extra attention mask - for masking out attention from abnormal CLS token to padding

        if exists(attn_mask):
            attn_mask = rearrange(attn_mask, 'b i j -> b 1 i j')
            sim = sim.masked_fill(~attn_mask, -torch.finfo(sim.dtype).max)

        # attention

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        # aggregate values

        out = einsum("b h i j, b j d -> b h i d", attn, v)

        # merge heads

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.attn_out(out) + self.ff_out(ff)

# cross attention - using multi-query + one-headed key / values as in PaLM w/ optional parallel feedforward

class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        context_dim=None,
        dim_head=64,
        heads=8,
        parallel_ff=False,
        ff_mult=4,
        norm_context=False
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head
        context_dim = default(context_dim, dim)

        self.norm = LayerNorm(dim)
        self.context_norm = LayerNorm(context_dim) if norm_context else nn.Identity()

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, dim_head * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        # whether to have parallel feedforward

        ff_inner_dim = ff_mult * dim

        self.ff = nn.Sequential(
            nn.Linear(dim, ff_inner_dim * 2, bias=False),
            SwiGLU(),
            nn.Linear(ff_inner_dim, dim, bias=False)
        ) if parallel_ff else None

    def forward(self, x, context):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        # pre-layernorm, for queries and context

        x = self.norm(x)
        context = self.context_norm(context)

        # get queries

        q = self.to_q(x)
        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)

        # scale

        q = q * self.scale

        # get key / values

        k, v = self.to_kv(context).chunk(2, dim=-1)

        # query / key similarity

        sim = einsum('b h i d, b j d -> b h i j', q, k)

        # attention

        sim = sim - sim.amax(dim=-1, keepdim=True)
        attn = sim.softmax(dim=-1)

        # aggregate

        out = einsum('b h i j, b j d -> b h i d', attn, v)

        # merge and combine heads

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        # add parallel feedforward (for multimodal layers)

        if exists(self.ff):
            out = out + self.ff(x)

        return out


#----------------------------------------------images coca --------------------
class CoCa_images(nn.Module):
    def __init__(
        self,
        patch_dim=3*40*40,
        patch_len=200,
        dim=386,
        num_classes=2,
        unimodal_depth=4,
        multimodal_depth=4,
        dim_head=64,
        heads=8,
        ff_mult=4,
        caption_loss_weight=1.,
        contrastive_loss_weight=1.,
        pad_id=0
    ):
        super().__init__()
        self.dim = dim

        self.pad_id = pad_id
        self.caption_loss_weight = caption_loss_weight
        self.contrastive_loss_weight = contrastive_loss_weight
        
        # # line embeddings
        # self.to_abnoraml_embedding =nn.Sequential(
        #     Rearrange('b s c w h -> b s (c w h)'),
        #     nn.Linear(patch_dim, dim),)

        # self.to_noraml_embedding =nn.Sequential(
        # Rearrange('b s c w h -> b s (c w h)'),
        # nn.Linear(patch_dim, dim),)
        
        # CNN embeddings
        self.squeue_len=patch_len
        self.to_abnoraml_embedding =nn.Sequential(
            Rearrange('b s c w h -> (b s) c w h'),
            nn.Sequential(
                nn.Conv2d(3, dim,kernel_size=(3, 3),stride=(2, 2)),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1,1))),
            Rearrange('(b s) c 1 1 -> b s c ',s=self.squeue_len),
        )

        self.to_noraml_embedding =nn.Sequential(
            Rearrange('b s c w h -> (b s) c w h'),
            nn.Sequential(
                nn.Conv2d(3, dim,kernel_size=(3, 3),stride=(2, 2)),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1,1))),
            Rearrange('(b s) c 1 1 -> b s c ',s=self.squeue_len),
        )

        # token embeddings
        self.abnormal_cls_token = nn.Parameter(torch.randn(dim))
        self.normal_cls_token = nn.Parameter(torch.randn(dim))

        # contrastive learning temperature

        self.temperature = nn.Parameter(torch.Tensor([1.]))

        # unimodal layers

        self.unimodal_layers_abnormal = nn.ModuleList([])
        for ind in range(unimodal_depth):
            self.unimodal_layers_abnormal.append(
                Residual(ParallelTransformerBlock(dim=dim, dim_head=dim_head, heads=heads, ff_mult=ff_mult)),
            )
        
        self.unimodal_layers_normal = nn.ModuleList([])
        for ind in range(unimodal_depth):
            self.unimodal_layers_normal.append( Residual(ParallelTransformerBlock(dim=dim, dim_head=dim_head, heads=heads, ff_mult=ff_mult)), )

        # multimodal layers

        self.multimodal_layers = nn.ModuleList([])
        for ind in range(multimodal_depth):
            self.multimodal_layers.append(nn.ModuleList([
                Residual(ParallelTransformerBlock(dim=dim, dim_head=dim_head, heads=heads, ff_mult=ff_mult)),
                Residual(CrossAttention(dim=dim, dim_head=dim_head, heads=heads, parallel_ff=True, ff_mult=ff_mult))
            ]))
        


        self.to_logits = nn.Sequential(
            LayerNorm(dim),
            nn.Linear(dim, num_classes, bias=False)
        )

    
    def embed_abnormal(self, abnormal):
        
        abnormal=self.to_abnoraml_embedding(abnormal)

        
        batch, device = abnormal.shape[0], abnormal.device

        seq = abnormal.shape[1]

        # append abnormal cls tokens

        abnormal_cls_tokens = repeat(self.abnormal_cls_token, 'd -> b 1 d', b=batch)
        abnormal_tokens = torch.cat((abnormal, abnormal_cls_tokens), dim=-2)

        # create specific mask for abnormal cls token at the end
        # to prevent it from attending to padding

        # cls_mask = rearrange(abnormal!=self.pad_id, 'b j -> b 1 j')
        # attn_mask = F.pad(cls_mask, (0, 1, seq, 0), value=True)

        # go through unimodal layers

        for attn_ff in self.unimodal_layers_abnormal:
            # abnormal_tokens = attn_ff(abnormal_tokens, attn_mask=attn_mask)
            abnormal_tokens = attn_ff(abnormal_tokens)

        # get abnormal cls token

        #abnormal_tokens, abnormal_cls_tokens = abnormal_tokens[:, :-1], abnormal_tokens[:, -1]
        return abnormal_tokens

    # embed noraml cell image
    def embed_normal(self, normal):
        normal=self.to_noraml_embedding(normal)
        batch, device = normal.shape[0], normal.device

        seq = normal.shape[1]

        # append normal cls tokens

        normal_cls_tokens = repeat(self.normal_cls_token, 'd -> b 1 d', b=batch)
        normal_tokens = torch.cat((normal, normal_cls_tokens), dim=-2)

        # create specific mask for normal cls token at the end
        # to prevent it from attending to padding

        # cls_mask = rearrange(normal!=self.pad_id, 'b j -> b 1 j')
        # attn_mask = F.pad(cls_mask, (0, 1, seq, 0), value=True)

        # go through unimodal layers

        for attn_ff in self.unimodal_layers_normal:
            # normal_tokens = attn_ff(normal_tokens, attn_mask=attn_mask)
            normal_tokens = attn_ff(normal_tokens)

        # get normal cls token

        #normal_tokens, normal_cls_tokens = normal_tokens[:, :-1], normal_tokens[:, -1]
        return normal_tokens

    def forward(
        self,
        input,
        labels=None,
        return_loss=False,
    ):
        batch, device = input.shape[0], input.device
        abnormal,normal=torch.chunk(input,2,dim=1)


        abnormal_tokens = self.embed_abnormal(abnormal)
        normal_tokens = self.embed_abnormal(normal)

        # go through multimodal layers
        abnormal_cls_token=abnormal_tokens[:,-1]
        normal_cls_token=normal_tokens[:,-1]

        # 直接使用所有的作为正常的特征
        for attn_ff, cross_attn in self.multimodal_layers:
            abnormal_tokens = attn_ff(abnormal_tokens)
            abnormal_tokens = cross_attn(abnormal_tokens, normal_tokens)
            
        cls_token=abnormal_tokens[:,-1]
        logits = self.to_logits(cls_token)


        if not return_loss:
            return logits

        # shorthand

        ce = F.cross_entropy

        # calculate caption loss (cross entropy loss)

        caption_loss = ce(logits, labels)
        caption_loss = caption_loss * self.caption_loss_weight
        
        # calculate contrastive loss

        sim = einsum('i d, j d -> i j', abnormal_cls_token, normal_cls_token)
        sim = sim * self.temperature.exp()
        contrastive_labels = torch.arange(batch, device=device)

        contrastive_loss = (ce(sim, contrastive_labels) + ce(sim.t(), contrastive_labels)) * 0.5
        contrastive_loss = contrastive_loss * self.contrastive_loss_weight

        return caption_loss + contrastive_loss,logits




if __name__=="__main__":

    # # feature
    # model = CoCa_feature(patch_dim=1536,dim=384,num_classes=2,unimodal_depth=4,multimodal_depth=4)
    # inputs=torch.zeros((5,20,1536))
    # label=torch.LongTensor([1,1,1,1,1])
    # a=model(inputs,label,True)
    # print(a)
    
    # images
    model = CoCa_images()
    inputs=torch.zeros((5,400,3,40,40))
    label=torch.LongTensor([1,1,1,1,1])
    a=model(inputs,label,True)
    print(a)