from transformers import PretrainedConfig, PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import MoeCausalLMOutputWithPast
from transformers.activations import ACT2FN
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class BocchiMindConfig(PretrainedConfig):
    model_type = "bocchimind"

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
        self.weight=nn.Parameter(torch.ones(dim))
    def _norm(self,x):
        return torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+self.eps)*x
    def forward(self,x):
        return self.weight*self._norm(x.float()).type_as(x)


#RoPE&YaRN
def precompute_freqs_cis(dim:int,end:int=32*1024,rope_base:float=10000.0,rope_scaling:dict=None):
    #初始化
    freqs,attn_factor=1./(rope_base**(torch.arange(0,dim,2)[:(dim//2)].float()/dim)),1.0
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
    t = torch.arange(end,device=freqs.device).float()

    #计算外积，得到每个位置的旋转角度
    freqs=torch.outer(t,freqs).float()

    freqs_cos = (torch.cat([torch.cos(freqs),torch.cos(freqs)],dim=-1))*attn_factor
    freqs_sin = (torch.cat([torch.sin(freqs),torch.sin(freqs)],dim=-1))*attn_factor

    return freqs_cos, freqs_sin

def apply_rotary_pos_emb(q, k, cos ,sin, position_ids=None, unsqueeze_dim=1):
    def rotate_half(x):
        return torch.cat((-x[...,x.shape[-1]//2:],x[..., :x.shape[-1]//2]),dim=-1)
    
    #  公式为x*cos + rotate_half(x)*sin
    q_emb=(q*cos.unsqueeze(unsqueeze_dim))+(rotate_half(q)*sin.unsqueeze(unsqueeze_dim))
    k_emb=(k*cos.unsqueeze(unsqueeze_dim))+(rotate_half(k)*sin.unsqueeze(unsqueeze_dim))
    return q_emb,k_emb

#GQA
def repeat_kv(x:torch.tensor,n_rep:int)->torch.tensor:
    bs,slen,num_head,head_dim=x.shape
    if n_rep==1:
        return x
    return x[:,:,:, None,:].expand(bs, slen, num_head, n_rep, head_dim).reshape(bs, slen, n_rep*num_head, head_dim)

class Attention(nn.Module):
    def __init__(self,args:BocchiMindConfig):
        super().__init__()
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
        self.v_proj=nn.Linear(args.hidden_size,self.num_key_value_heads*self.head_dim,bias=False)
        self.o_proj=nn.Linear(args.num_attention_heads*self.head_dim,args.hidden_size,bias=False)

        self.q_norm=RMSnorm(self.head_dim,eps=args.rms_norm_eps)
        self.k_norm=RMSnorm(self.head_dim,eps=args.rms_norm_eps)

        self.attn_drop=nn.Dropout(args.dropout)
        self.resid_dropout=nn.Dropout(args.dropout)
        self.dropout=args.dropout

        self.flash=hasattr(torch.nn.functional,'scaled_dot_product_attention') and args.flash_attention

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        bs, slen, _ = x.shape
        q, k ,v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = q.view(bs, slen, self.n_local_heads, self.head_dim)
        xk = k.view(bs, slen, self.n_local_kv_heads, self.head_dim)
        xv = v.view(bs, slen, self.n_local_kv_heads, self.head_dim)
        xq, xk = self.q_norm(xq), self.k_norm(xk)

        #调用RoPE
        cos, sin=position_embeddings
        xq, xk=apply_rotary_pos_emb(xq, xk, cos, sin)
        #KVcache
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None
        xq, xk, xv = (xq.transpose(1,2), repeat_kv(xk,self.n_rep).transpose(1,2), repeat_kv(xv, self.n_rep).transpose(1,2))
        #使用flash—attention，
        if self.flash and (slen > 1) and (past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1)):
            output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        else:
            scores = xq @ xk.transpose(-1,-2)/math.sqrt(self.head_dim)
            # 因果掩码
            scores[:, :, :, -slen:] += torch.full((slen, slen), float("-inf"), device=scores.device).triu(1)
            # 填充掩码
            if attention_mask is not None: 
                scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
            output = self.attn_drop(F.softmax(scores.float(), dim=-1).type_as(xq)) @ xv
        output = output.transpose(1, 2).reshape(bs, slen, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv
    
#FFN
class FeedForward(nn.Module):
    def __init__(self,args:BocchiMindConfig):
        super().__init__()
        if args.intermediate_size is None:
            intermediate_size=int(args.hidden_size*8/3)
            args.intermediate_size=64*((intermediate_size+64-1)//64)
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.act = ACT2FN[args.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act(self.gate_proj(x))*self.up_proj(x))

class MOEFeedForward(nn.Module):
    def __init__(self, args: BocchiMindConfig):
        super().__init__()
        self.args = args
        # MoE gate
        self.gate = nn.Linear(args.hidden_size, args.num_experts, bias=False)
        self.experts = nn.ModuleList([FeedForward(args, intermediate_size=args.moe_intermediate_size) for _ in range(args.num_experts)])
        self.act_fn = ACT2FN[args.hidden_act]

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)
        scores = F.softmax(self.gate(x_flat), dim=-1)
        topk_weight, topk_idx = torch.topk(scores, k=self.args.num_experts_per_tok, dim=-1, sorted=False)
        # 将权重归一化
        if self.args.norm_topk_prob: 
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
        y = torch.zeros_like(x_flat) # 初始化输出张量
        for i, expert in enumerate(self.experts):
            mask = (topk_idx == i)
            # 判断该专家是否有token选择
            if mask.any():
                # 将每一个token对应expert的索引取出
                token_idx = mask.any(dim=-1).nonzero().flatten()
                weight = topk_weight[mask].view(-1, 1)
                y.index_add_(0, token_idx, (expert(x_flat[token_idx]) * weight).to(y.dtype))
            elif self.training:
                y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())
        if self.training and self.args.router_aux_loss_coef > 0:
            load = F.one_hot(topk_idx, self.args.num_experts).float().mean(0)
            self.aux_loss = (load * scores.mean(0)).sum() * self.args.num_experts * self.args.router_aux_loss_coef
        else:
            self.aux_loss = scores.new_zeros(1).squeeze()
        return y.view(batch_size, seq_len, hidden_dim)
#Minimind模块
class BocchiMindBlock(nn.Module):
    def __init__(self, layer_id:int, args:BocchiMindConfig):
        super().__init__()
        self.self_attn = Attention(args)
        self.input_layernorm = RMSnorm(args.hidden_size, args.rms_norm_eps)
        self.post_attention_layernorm = RMSnorm(args.hidden_size, args.rms_norm_eps)
        self.mlp = FeedForward(args) if not args.use_moe else MOEFeedForward(args)\
        
    def forward(self, hidden_states, position_embedding, past_key_value=None, use_cache=False, attention_mask=None):
        residual = hidden_states
        hidden_states, past_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embedding,
            past_key_value, use_cache, attention_mask
            )
        hidden_states += residual
        hidden_states += self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, past_key_value
        
class BocchiMindModel(nn.Module):
    def __init__(self, args:BocchiMindConfig):
        super().__init__()
        self.args = args
        self.num_hidden_layers = args.num_hidden_layers
        self.vocab_size = args.vocab_size
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.dropout = nn.Dropout(args.dropout)
        self.layers = nn.ModuleList([BocchiMindBlock(i,args) for i in range(self.num_hidden_layers)])
        self.norm = RMSnorm(args.hidden_size, args.rms_norm_eps)
        freqs_cos, freqs_sin = precompute_freqs_cis(
            args.hidden_size//args.num_attention_heads, args.max_position_embeddings,
            args.rope_theta, args.rope_scaling
        ) 

        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(
            self,
            input_ids,
            attention_mask=None,
            past_key_values=None,
            use_cache=None,
            **kwargs,
    ):
        batch_size, seq_len = input_ids.shape
        if hasattr(past_key_values, "layers"): 
            past_key_values = None
        past_key_values = past_key_values or [None]*len(self.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        hidden_states = self.dropout(self.embed_tokens(input_ids))

        position_embeddings = (
            self.freqs_cos[start_pos:start_pos+seq_len], 
            self.freqs_sin[start_pos:start_pos+seq_len]
        )

        presents = []
        for layer, past_key_value in zip(self.layers, past_key_values):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                attention_mask=attention_mask,
                use_cache=use_cache
            )
            presents.append(present)
        hidden_states = self.norm(hidden_states)
        aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze())
        return hidden_states, presents, aux_loss

class BocchiMindForCausalLM(PreTrainedModel, GenerationMixin):
    cofig_class = BocchiMindConfig

    def __init__(self, config:BocchiMindConfig):
        self.config=config
        super().__init__(config)
        self.model = BocchiMindModel(self.config)
        # 将预测映射回到词表
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        # 权重共享
        # 输出层的权重和嵌入层的权重共享
        self.model.embed_tokens.weight = self.lm_head.weight

        self.Out = MoeCausalLMOutputWithPast()

    def forward(self, 
                input_ids,
                attention_mask=None,
                past_key_values=None,
                use_cache=False,
                logits_keep=0,
                labels=None,
                **kwargs):
        hidden_states, past_key_values, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )
        # logits to keep是整数，那就保留最后n个位置
        #生成的时候只需要最后的logits来预测下一个token
        slice_indices = (
            slice(-logits_keep, None)
            if isinstance(logits_keep, int)
            else logits_keep
            )
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        if labels is not None:
            x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
        return MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits, past_key_values=past_key_values, hidden_states=hidden_states)

    # # https://github.com/jingyaogong/minimind/discussions/611
    # @torch.inference_mode()
    # def generate(self, inputs=None, attention_mask=None, max_new_tokens=8192, temperature=0.85, top_p=0.85, top_k=50, eos_token_id=2, streamer=None, use_cache=True, num_return_sequences=1, do_sample=True, repetition_penalty=1.0, **kwargs):
    #     input_ids = kwargs.pop("input_ids", inputs).repeat(num_return_sequences, 1)
    #     attention_mask = attention_mask.repeat(num_return_sequences, 1) if attention_mask is not None else None
    #     past_key_values = kwargs.pop("past_key_values", None)
    #     finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
    #     if streamer: 
    #         streamer.put(input_ids.cpu())
    #     for _ in range(max_new_tokens):
    #         past_len = past_key_values[0][0].shape[1] if past_key_values else 0
    #         outputs = self.forward(input_ids[:, past_len:], attention_mask, past_key_values, use_cache=use_cache, **kwargs)
    #         attention_mask = torch.cat([attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)], -1) if attention_mask is not None else None
    #         logits = outputs.logits[:, -1, :] / temperature
    #         if repetition_penalty != 1.0:
    #             for i in range(input_ids.shape[0]): 
    #                 logits[i, torch.unique(input_ids[i])] /= repetition_penalty
    #         if top_k > 0: 
    #             logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float('inf')
    #         if top_p < 1.0:
    #             sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    #             mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
    #             mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0
    #             logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')
    #         next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1) if do_sample else torch.argmax(logits, dim=-1, keepdim=True)
    #         if eos_token_id is not None: 
    #             next_token = torch.where(finished.unsqueeze(-1), next_token.new_full((next_token.shape[0], 1), eos_token_id), next_token)
    #         input_ids = torch.cat([input_ids, next_token], dim=-1)
    #         past_key_values = outputs.past_key_values if use_cache else None
    #         if streamer: 
    #             streamer.put(next_token.cpu())
    #         if eos_token_id is not None:
    #             finished |= next_token.squeeze(-1).eq(eos_token_id)
    #             if finished.all(): 
    #                 break
    #     if streamer: 
    #         streamer.end()
    #     if kwargs.get("return_kv"): 
    #         return {'generated_ids': input_ids, 'past_kv': past_key_values}
    #     return input_ids