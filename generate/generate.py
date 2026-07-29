import sys
import torch.nn.functional as F
import torch as torch
import time

sys.path.append("F:/ai/python_ai/llm/model(transformer)")
sys.path.append("F:/ai/python_ai/llm/tokenizer")
sys.path.append("F:/ai/python_ai/llm")

from config import modelconfig
from model_me import Transformer
from tokenizer import mytokenizer

tokenizer=mytokenizer("F:/ai/python_ai/llm/tokenizer/vocab/tokenizer.json")
config=modelconfig()

def load_checkpoint(path,model,optimizer,device):
    checkpoint=torch.load(path,map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    epoch=checkpoint["epoch"]
    step=checkpoint["step"]
    print("成功加载checkpoint")
    return epoch,step

def generate(temperature,top_k):
    device="cuda" if torch.cuda.is_available() else "cpu"
    max_len=2048
    
    model=Transformer(config).to(device)
    
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=0.01)
    #读取checkpoints
    epoch,step=load_checkpoint(
        "X:/A checkpoints Hub/Genesis test v0(2026.7.9)/A-grpotrain-checkpoints/GRPO-epoch1-step799.pth",
        model,
        optimizer,
        device
    )
    model.eval()
    #打印模型参数量
    total_params=sum(p.numel()for p in model.parameters())
    print(total_params)
    
    pr=input("请输入提示词：")
    prompt=[tokenizer.user_id]
    pr=tokenizer.encode(pr)
    prompt.extend(pr)
    prompt+=[tokenizer.think_id]
    for _ in range(max_len-len(prompt)):
        b=torch.tensor(prompt,dtype=torch.long,device=device).unsqueeze(0)
        with torch.no_grad():
            out,_=model(b,past_kv=None)#out=(batch,seq,vocab_size)
            out=out[:,-1,:] #取batch中每个seq的最后一个word
            out=out/temperature#out=(batch,vocab_size)
            score=F.softmax(out,dim=-1)
            topk_token,topk_id=torch.topk(score,top_k,dim=-1)#toke_id是词的下标
            idx=torch.multinomial(topk_token,1)#取出的是topk里面的序号
            next_token=topk_id.gather(-1,idx)#映射回原词表
            next_token=next_token.item()#转成值，值就是原词表的token位置
            print(tokenizer.decode([next_token]),end="")
            if next_token==tokenizer.eos_id:
                break
            time.sleep(0.05)
            prompt.append(next_token)

generate(0.9,50)