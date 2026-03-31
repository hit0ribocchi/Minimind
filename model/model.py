from transformers import PretrainedConfig
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class MokioMindConfig(PretrainedConfig):
    model_type = "mokiomind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: int = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000,
        inference_rope_scaling: bool = False,
        flash_attention: bool = True,
        ############ MoE ############
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.01,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.scoring_func = scoring_func

        self.rope_scaling = (
            {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 16,
                "original_max_position_embeddings": 2048,
                "attention_factor": 1.0,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )


class RMSnorm(nn.Module):
    def __init__(self, dim:int, eps:float=1e-5):
        super().__init__()
        self.dim=dim
        self.eps=eps
        self.weight=nn.Paramter(torch.ones(dim))
    def _norm(self,x):
        return torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+self.eps)*x
    def forward(self,x):
        return self.weight*self._norm(x.float()).type_as(x)


#RoPE&YaRN
def precompute_freqs_cis(dim:int,end:int=32*1024,rope_base:float=10000.0,rope_scaling:dict=None):
    #初始化
    freqs,attn_factor=1./rope_base**(torch.arange(0,dim,2)[:(dim//2)].float()/dim),1.0
    if rope_scaling is not None:
        orig_max,factor,beta_fast,beta_slow=(
            rope_scaling["original_max_position_embeddings"],
            rope_scaling["factor"],
            rope_scaling["beta_fast"],
            rope_scaling["beta_slow"])
        
        #推断长度大于训练长度,需要使用缩放
        if end > orig_max:
            #计算i索引的匿名函数
            def inv_dim(b): return dim*math.log(orig_max/(b*2*math.pi))/(2*math.log(rope_base))

            #YaRN方法存在一个斜坡函数，上下界分别为high和low，值为1和0，中间是混合函数，在LLaMA模型中分别为1和32（实验得到）
            #low:小于为高频部分
            #high:大于为低频部分
            low,high=max(math.floor(inv_dim(beta_fast)),0),min(math.ceil(inv_dim(beta_slow)),dim//2-1)

            #计算缩放函数
            ramp=torch.clamp((torch.arange(dim//2,device=freqs.device)-low)/max(high-low,0.001),0,1)

            #在频率上应用缩放因子
            # 当 ramp=0 时（高频），系数为1，保持频率不变
            # 当 ramp=1 时（低频），系数为1/factor，线性缩放
            # 当 ramp在0-1之间时，平滑过度
            freqs=freqs*(1-ramp)+freqs/factor*ramp
        # 根据end申城位置索引
    t = torch.arange(end,device=freqs.devices()).float()

    #计算外积，得到每个位置的旋转角度
    freqs=torch.outer(t,freqs).float()

    freqs_cos = (torch.cat([torch.cos(freqs),torch.cos(freqs)],dim=-1))*attn_factor
    freqs_sin = (torch.cat([torch.sin(freqs),torch.sin(freqs)],dim=-1))*attn_factor

    return freqs_cos, freqs_sin

def apply_rotary_pos_emb(q, k, cos ,sin, position_ids=None, unsqueeze_dim=1):
    def rotate_half(x):
        return torch.cat((-x[...,x.shape[-1]//2:],x[...,x.shape[...,:x.shape[-1]//2]]),dim=-1)
    
    #  公式为x*cos + rotate_half(x)*sin
    q_emb=(q*cos.unsqueeze(unsqueeze_dim))+(rotate_half(q)*sin.unsqueeze(unsqueeze_dim))
    k_emb=(k*cos.unsqueeze(unsqueeze_dim))+(rotate_half(k)*sin.unsqueeze(unsqueeze_dim))
    return q_emb,k_emb

#GQA
def repeat_kv(x:torch.tensor,n_rep:int)->torch.tensor:
    bs,slen,num_head,head_dim=x.shape()
    if n_rep==1:
        return x
    return x[:,:,:, None,:].expand(bs, slen, num_head, n_rep, head_dim).reshape(bs, slen, n_rep*num_head, head_dim)

class Attention(nn.Module):
    def __init__(self,args:MokioMindConfig):
        super.__init__()
        self.num_key_value_heads=args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads
        #检查Q的头数能否被KV整除
        assert args.num_attention_heads%self.num_key_value_heads==0,"num_attention_heads% must be divisible by num_key_value_heads"
        #Q的头数和KV的头数
        self.n_local_heads=args.num_attention_heads
        self.n_local_kv_heads=self.num_key_value_heads
        self.n_rep=self.n_local_heads//self.num_key_value_heads
        self.head_dim=args.hidden_size//args.num_attention_heads

        self.q_proj=nn.Linear(args.hidden_size,args.num_attention_heads*self.head_dim,bias=False)
        self.k_proj=nn.Linear(args.hidden_size,self.num_key_value_heads*self.head_dim,bias=False)
        self.k_proj=nn.Linear(args.hidden_size,self.num_key_value_heads*self.head_dim,bias=False)
        self.o_proj=nn.Linear(args.num_attention_heads*self.head_dim,args.hidden_size,bias=False)

        self.q_norm=RMSnorm(self.head_dim,eps=args.rms_norm_eps)
        self.k_norm=RMSnorm(self.head_dim,eps=args.rms_norm_eps)

        self.attn_drop=nn.Dropout(args.dropout)
        self.resid_drop=nn.Dropout(args.dropout)
        self.dropout=args.dropout

        self.flash=hasattr(torch.nn.functional,'scaled_dot_product_attention') and args.flash_attention

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        bs, slen, _ = x.shape()
        q, k ,v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = q.view(bs, slen, self.n_local_heads, self.head_dim)
        xk = k.view(bs, slen, self.n_local_kv_heads, self.head_dim)
        xv = v.view(bs, slen, self.n_local_kv_heads, self.head_dim)
        xq, xk = self.q_norm(xq), self.k_norm(xk)

        cos, sin=position_embeddings
        xq, xk=apply_rotary_pos_emb(xq, xk, cos, sin)
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None
        xq, xk, xv = (xq.transpose(1,2), repeat_kv(xk,self.n_rep).transpose(1,2), repeat_kv(xv, self.n_rep).transpose(1,2))
        if self.flash and (slen > 1) and (past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1)):
            output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        else:
            scores = xq @ xk.transpose(-1,-2)/math.sqrt(self.head_dim)
            scores[:, :, :, -slen:] += torch.full((slen, slen), float("-inf"), device=scores.device).triu(1)#
            if attention_mask is not None: 
                scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
            output = self.attn_drop(F.sofmax(scores.float(), dim=-1).type_as(xq)) @ xv
        output = output.transpose(1, 2).reshape(bs, slen, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv

