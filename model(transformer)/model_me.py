#更新于2026.6.14(加上了RMSNorm，SwiGLU,RoPE,pre-norm)
#2026.6.21 加上了GQA
#2026.6.27 加上了flash_attention
#2026.7.16 加上了MoE_FFN,加入config支持
#2026.7.24 加上了KV Cache
import torch.nn as nn
import torch.nn.functional as F
import torch as torch
import torch
import torch.nn as nn
       
# class PositionEncoder(nn.Module):
#     def __init__(self, max_len, d_model):
#         super().__init__()

#         pe = torch.zeros(max_len, d_model)

#         position = torch.arange(0, max_len).unsqueeze(1)

#         div_term = torch.exp(
#             torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
#         )

#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)

#         self.register_buffer("pe", pe)

#     def forward(self, x):
#         seq_len = x.size(1)
#         return x + self.pe[:seq_len].unsqueeze(0)
#class PositionEncoder:
#   def __init__(self,max_len,d_model):
#        self.d_model=d_model
#        self.max_len=max_len
#        self.p=torch.zeros(max_len,self.d_model)
#        for i in range(max_len):
#            for j in range(self.d_model):
#                if j%2==0:
#                    self.p[i][j]=torch.sin(i/(10000**(j/self.d_model)))
#                else:
#                    self.p[i][j]=torch.cos(i/(10000**(j/self.d_model)))
        
#    def forward(self,x):#x=(max_len,d_model)
#        seq_len=x.shape[1]
#        return x+self.p[:seq_len].unsqueeze(0)
    
class MaskMultiHeadAttention(nn.Module):
    def __init__(self,num_head,d_model,flash_attention,kv):
        super().__init__()
        assert d_model%num_head==0
        assert num_head%kv==0
        self.d_k=d_model//num_head
        self.num_head=num_head
        self.num_kv_head=num_head//kv
        self.d_model=d_model
        self.flash_attention=flash_attention
        self.kv=kv
        
        self.q=nn.Linear(d_model,d_model)#如果把batch_size也定了就不好搞了，太大了
        self.k=nn.Linear(d_model,self.num_kv_head*self.d_k)#flash attention
        self.v=nn.Linear(d_model,self.num_kv_head*self.d_k)
        self.out=nn.Linear(d_model,d_model)
        self.dropout=nn.Dropout(0.1)
    
    def RoPE(self,m,max_len,start_pos):#(batch,num_head,max_len,d_k)
        device=m.device
        inv_freq=1.0/(10000**(torch.arange(0,self.d_k,2,device=device).float()/self.d_k))
        pos=torch.arange(start_pos,start_pos+max_len,device=device).float()
        freqs=torch.outer(pos,inv_freq)
        sin=freqs.sin().unsqueeze(0).unsqueeze(0)
        cos=freqs.cos().unsqueeze(0).unsqueeze(0)
        #half=self.d_k//2
        x=m[...,0::2] #取偶数对
        y=m[...,1::2] #取奇数对
        x_new=x*cos-y*sin
        y_new=x*sin+y*cos
        out=torch.empty_like(m)
        out[...,0::2]=x_new
        out[...,1::2]=y_new
        return out
        
    def forward(self,x,past_kv=None,is_causal=False):#x=(batch_size,max_len,d_model)  mask=(mask_len,max_len)
        batch_size,max_len,_=x.size()
        
        Q=self.q(x)
        new_K=self.k(x)
        new_V=self.v(x)
        
        Q=Q.view(batch_size,max_len,self.num_head,self.d_k).transpose(1,2)
        new_K=new_K.view(batch_size,max_len,self.num_kv_head,self.d_k).transpose(1,2)
        new_V=new_V.view(batch_size,max_len,self.num_kv_head,self.d_k).transpose(1,2)#(batch,num_kv_head,max_len,d_k)
        
        if past_kv is not None:
            past_K,past_V=past_kv
            past_len=past_K.size(2)
        else:
            past_K=None
            past_V=None
            past_len=0
            
        Q=self.RoPE(Q,max_len,past_len)
        new_K=self.RoPE(new_K,max_len,past_len)
        #拼接缓存
        if past_kv is not None:
            K=torch.cat([past_K,new_K],dim=2)
            V=torch.cat([past_V,new_V],dim=2)
        else:
            K=new_K
            V=new_V
        cache=(K,V)
        
        K=K.repeat_interleave(self.kv,dim=1)
        V=V.repeat_interleave(self.kv,dim=1)
        
        if self.flash_attention==True:
            attention_out=F.scaled_dot_product_attention(
            Q,K,V,
            dropout_p=0.1 if self.training else 0.0,
            is_causal=is_causal  
            )
            
        else:
            qk=torch.matmul(Q,K.transpose(2,3))
            qk=qk/(self.d_k**0.5)
            if is_causal and max_len>1:
                mask=torch.zeros(max_len,past_len+max_len,device=x.device)
                causal=torch.triu(torch.ones(max_len,max_len,device=x.device)*-1e9,diagonal=1)
                mask[:,past_len:]=causal
                mask=mask.to(x.device).unsqueeze(0).unsqueeze(0)
                qk=qk+mask
            
            score=self.dropout(F.softmax(qk,dim=-1))
            attention_out=torch.matmul(score,V)#(batch,num,d_k)
        
        attention_out=attention_out.transpose(1,2)
        attention=attention_out.contiguous().view(batch_size,max_len,self.d_model)
        o=self.out(attention)
        o=self.dropout(o)
        return o,cache
#MoE混合专家模型
class Expert(nn.Module):
    def __init__(self,d_model):
        super().__init__()
        hidden=int(d_model*8/3)
        self.gate_proj=nn.Linear(d_model,hidden)
        self.up_proj=nn.Linear(d_model,hidden)
        self.down_proj=nn.Linear(hidden,d_model)
        
    def forward(self,x):
        gate=F.silu(self.gate_proj(x))
        up=self.up_proj(x)
        down=self.down_proj(gate*up)
        return down
    
class MoE_FFN(nn.Module):
    def __init__(self,d_model,num_experts,topk):
        super().__init__()
        self.num_experts=num_experts
        self.topk=topk
        self.d_model=d_model
        #Router
        self.gate=nn.Linear(d_model,num_experts,bias=False)
        #Router Experts
        self.experts=nn.ModuleList([Expert(d_model) for _ in range(num_experts)])
        #Shared Experts
        self.shared_expert=Expert(d_model)
    
    def forward(self,x):#x=(batch,sentence,d_model)
        batch,sen,_=x.shape
        shared_output=self.shared_expert(x)
        output=torch.zeros_like(batch*sen,self.d_model,device=x.device)
        score=self.gate(x)#score=(batch,sentence,num_experts)
        expert_gate,index=torch.topk(score,self.topk,dim=-1)#expert_gate,index=(batch,sentence,topk)
        expert_gate=torch.softmax(expert_gate,dim=-1)
        
        x=x.view(batch*sen,self.d_model)
        expert_gate=expert_gate.view(batch*sen,self.topk)
        index=index.view(batch*sen,self.topk)
        
        for ids in range(self.num_experts):
            mask=(index==ids)#mask=(batch*sen,topk)
            token_idx,topk_idx=torch.where(mask)
            gate=expert_gate[mask,topk_idx] #gate=(sentence)
            expert_input=x[token_idx] #把同一个expert下面的token取出来
            expert_output=self.experts[ids](expert_input)#E_output=(len(token_idx),d_model)
            expert_output*=gate.unsqueeze(-1)
            output[token_idx]+=expert_output
        
        output=output.view(batch,sen,self.d_model)
        return output+shared_output
    
class FeedForwardNetwork(nn.Module):
    def __init__(self,d_model):
        super().__init__()
        hidden=int(d_model*8/3)
        self.w1=nn.Linear(d_model,hidden)
        self.w2=nn.Linear(d_model,hidden)
        self.w3=nn.Linear(hidden,d_model)
        self.dropout=nn.Dropout(0.1)
        
    def forward(self,x):
        out=self.w3(F.silu(self.w1(x))*self.w2(x))
        return self.dropout(out)

class gpt_decoder(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.mmha=MaskMultiHeadAttention(config.num_head,config.d_model,config.flash_attention,config.kv)
        self.norm1=nn.RMSNorm(config.d_model)
        self.norm2=nn.RMSNorm(config.d_model)
        if config.moe is not None:
            self.ffn=MoE_FFN(config.d_model,**config.moe)
        else:
            self.ffn=FeedForwardNetwork(config.d_model)
            
    def forward(self,x,past_kv=None):#(batch_size,seq_len,d_model)
        if past_kv is None:
            is_causal=True
        else:
            is_causal=False
        
        out,cache=self.mmha(self.norm1(x),past_kv,is_causal)
        x=x+out
        
        output=x+self.ffn(self.norm2(x))
        return output,cache

class Transformer(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.out=nn.Linear(config.d_model,config.vocab_size,bias=False)
        self.blocks=nn.ModuleList([gpt_decoder(config)for _ in range(config.n_layer)])
        self.norm=nn.RMSNorm(config.d_model)
        self.emd=nn.Embedding(config.vocab_size,config.d_model)
        self.out.weight=self.emd.weight
    
    def forward(self,x,past_kv=None):#(batch_size,seq_len,d_model)所有处理都是动态的
        x=self.emd(x)
        new_cache=[]
        
        for i,block in enumerate(self.blocks):
            if past_kv is None:
                layer_cache=None
            else:
                layer_cache=past_kv[i]
            x,cache=block(x,layer_cache)
            new_cache.append(cache)
        
        x=self.norm(x)
        out=self.out(x)
        return out,new_cache