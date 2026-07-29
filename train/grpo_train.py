#初次编写2026.7.27
import os
import sys
import torch
import random
import time
import json
import torch.nn.functional as F
import torch.nn as nn

sys.path.append("/root/autodl-tmp/.autodl/llm/model(transformer)")
sys.path.append("/root/autodl-tmp/.autodl/llm/tokenizer")
sys.path.append("/root/autodl-tmp/.autodl/llm")

from torch.utils.data import IterableDataset
from torch.utils.data import DataLoader
from model_me import Transformer
from tokenizer import mytokenizer
from config import modelconfig

config=modelconfig()
tokenizer=mytokenizer("/root/autodl-tmp/.autodl/llm/tokenizer/vocab/tokenizer.json")

def get_txt_files(folder):
    txt_files=[]
    for file_name in os.listdir(folder):
        if file_name.endswith(".jsonl"):
            txt_files.append(
                os.path.join(
                    folder,
                    file_name
                )
            )
    return txt_files

def load_checkpoint(path,model,optimizer,device):
    checkpoint=torch.load(path,map_location=device)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    epoch=checkpoint["epoch"]
    step=checkpoint["step"]
    print("成功加载checkpoint")
    return epoch,step

class GRPODatasetream(IterableDataset):
    def __init__(self,tokenizer,files,max_len,index_len=10,slices=20):
        self.tokenizer=tokenizer
        self.files=files
        self.slice=slices
        self.max_len=max_len
        self.index_len=index_len
        
    def process_grpo(self,data):#为了让多轮对话保持记忆
        result=[]
        prompt=[]
        conversation=data["conversations"]
        for item in conversation:
            if item["role"]=="system":
                continue
            if item["role"]=="user":
                prompt.extend(self.tokenizer.encode(item["content"]))
            if item["role"]=="assistant":
                break
        prompt.extend([self.tokenizer.assistant_id])
        target=self.tokenizer.encode(data["gt"][0])
        target.append(self.tokenizer.eos_id)
        result.append((prompt,target))
        return result
        # conversation=data["conversations"]
        # result=[]
        # history=[]
        # for item in conversation:
        #     if item["role"]=="user":
        #         history.append(item)#保存每个user的历史值
            
        #     elif item["role"]=="assistant":
        #         prompt=[]
        #         for h in history:#遍历每轮对话，使每轮对话都保存在history中
        #             if h["role"]=="user":
        #                 prompt.append(self.tokenzier.user_id)
        #                 prompt.append(self.tokenzier.encode(h["content"]))
        #             else:
        #                 prompt.append(self.tokenzier.assistant_id)
        #                 prompt.extend(self.tokenzier.encode(h["content"]))
        #         prompt.append(self.assistant_id)
        #         label=self.tokenzier.encode(item["content"])
        #         label.append(self.tokenzier.eos_id)
        #         result.append((prompt,label))
        #         history.append(item)#保存当前轮次的对话
        return result
    def data_file(self):
        files=self.files.copy()
        self.file_index={}
        for file in files:
            name=os.path.basename(file)#取文件名
            if file.endswith(".jsonl"):
                index=[]
                with open(file,"rb") as f:
                    while True:
                        pos=f.tell()
                        line=f.readline()
                        if not line:
                            break
                        index.append(pos)
                        for _ in range(self.slice-1):
                            if not f.readline():
                                break
                random.shuffle(index)
                self.file_index[name]=index
        
    def __iter__(self):
        self.data_file()
        files=self.files.copy()
        file_len=len(files)
        alive=0
        while alive!=file_len:
            random.shuffle(files)
            for file in files:
                name=os.path.basename(file)#取文件名
                if len(self.file_index[name])==0:
                    alive+=1
                    continue
                with open(file,"rb") as f:
                    for _ in range(min(self.index_len,len(self.file_index[name]))):
                        idx=self.file_index[name].pop()
                        f.seek(idx)
                        for _ in range(self.slice):
                            line=f.readline()
                            if not line:
                                break
                            line=line.decode("utf-8")
                            obj=json.loads(line)
                            results=self.process_grpo(obj)
                            for inputs,label in results:
                                inputs=inputs[:self.max_len]
                                label=label[:self.max_len]
                                input_pad=self.max_len-len(inputs)
                                label_pad=self.max_len-len(label)
                                inputs.extend([self.tokenizer.pad_id]*input_pad)
                                label.extend([self.tokenizer.pad_id]*label_pad)
                                x=torch.tensor(inputs,dtype=torch.long)
                                y=torch.tensor(label,dtype=torch.long)
                                yield x,y
             
class GRPOtrain(nn.Module):
    def __init__(self,groups,clip_coef,model,tokenizer,max_len,kl_b=0.04,temperature=0.9,topk=50):
        super().__init__()
        self.groups=groups
        self.clip=clip_coef
        self.model=model
        self.tokenizer=tokenizer
        self.temperature=temperature
        self.topk=topk
        self.kl_b=kl_b
        self.max_len=max_len
    
    def get_answer_probs(self,texts,device):#texts=(batch,seq)
        batch,_=texts.shape
        prompts=[]
        answer=[[] for _ in range(batch)]
        answer_mask=[[] for _ in range(batch)] #加掩码防止probs的值影响计算
        probs=[[] for _ in range(batch)]
        finished=[False for _ in range(batch)]
        #m=0计数句子结束的个数
        for text in texts:
            text=text.tolist()
            prompt=torch.tensor(text,dtype=torch.long,device=device)
            prompts.append(prompt)
        prompts=torch.stack(prompts)
        self.model.eval()#关闭训练模式，进行推理  
        cache=None
        for _ in range(self.max_len):
            with torch.no_grad():
                out,cache=self.model(prompts,cache)#out=(batch,seq,vocab_size)
                out=out[:,-1,:] #取batch中每个seq的最后一个word
                out=out/self.temperature
                log_score=F.log_softmax(out,dim=-1)#score=(batch.vocab_size)
                score=log_score.exp()
                
                topk_token,topk_id=torch.topk(score,self.topk,dim=-1)#toke_id是词的下标
                topk_token=topk_token/topk_token.sum(dim=-1,keepdim=True)
                idx=torch.multinomial(topk_token,1)#按概率采样取出的是topk里面的序号
                next_token=topk_id.gather(-1,idx)#映射回原词表
                prompts=next_token
                for i in range(batch):
                    if finished[i]:
                        answer[i].append(self.tokenizer.pad_id)
                        probs[i].append(torch.tensor(0.,device=device))
                        answer_mask[i].append(0)
                        continue
                    
                    token=next_token[i].item()
                    
                    if token==self.tokenizer.eos_id:
                        answer[i].append(token)
                        probs[i].append(log_score[i,token])
                        answer_mask[i].append(1)
                        finished[i]=True #m+=1
                    # elif m==batch:
                    #     return probs,answer
                    # elif  len(answer[i])>0 and (answer[i][-1]==-100 or answer[i][-1]==self.tokenizer.eos_id):
                    #     answer[i].extend([-100])
                    #     probs[i].extend([-100])
                    #     continue
                    else:
                        answer[i].append(token)
                        probs[i].append(log_score[i,token])#probs=(batch,seq)
                        answer_mask[i].append(1)
                if all(finished):
                    for j in range(batch):#就算所有都提前结束了都要pad
                        while len(answer[j])<self.max_len:
                            answer[j].append(self.tokenizer.pad_id)
                            probs[j].append(torch.tensor(0.,device=device))
                            answer_mask[j].append(0)
                    self.model.train()#开启训练模式
                    return answer,probs,answer_mask
        self.model.train()#开启训练模式
        return answer,probs,answer_mask#防止每个句子都没输出eos
                
    def Reward_get(self,prompt,answer,target):#answer=(batch,answer_len) 
        device=prompt.device
        rewards=[]
        for ans,tgt in zip(answer,target):#依据每个batch进行循环ans是未decode过的 tgt同样是没有decode过的
            tgt=tgt.tolist()#huggingface tokenizer要求传入list 但Dataloader传入的是tensor所以需要转换
            ans_text=self.tokenizer.decode(ans)
            tgt_text=self.tokenizer.decode(tgt)
            reward=0
            if tgt_text.strip() in ans_text.strip():#strip去掉字符串两侧的换行符号(如果有)
                reward+=1.0
            if len(ans_text)>0:
                reward+=0.1
            if self.tokenizer.eos_id in ans:
                reward+=0.05
            rewards.append(reward)
        return torch.tensor(rewards,dtype=torch.float32,device=device)
        
    def get_new_probs(self,texts,answer,device,model=None):#answer=(batch,answer_len)=probs
        answer=torch.tensor(answer,dtype=torch.long,device=device)
        _,answer_len=answer.shape
        prompts=[]
        
        for text in texts:
            text=text.tolist()
            prompt=torch.tensor(text,dtype=torch.long,device=device) 
            prompts.append(prompt)
        prompts=torch.stack(prompts)
        inputs=torch.cat([prompts,answer],dim=1)
        
        if model is None:
            model=self.model
        out,_=model(inputs)#out=(batch,seq,vocab_size)
        log_out=F.log_softmax(out[:,:-1,:],dim=-1)
        answer_probs=log_out[:,-answer_len:,:] #answer_probs=(batch,answer_len,vocab) 去最后answer_len个值
        new_probs=answer_probs.gather(-1,answer.unsqueeze(-1)).squeeze(-1)#answer升维度再去掉维度
        return new_probs
        
    def collect_data(self,prompt,target):#有batch个prompt
        device=prompt.device
        all_data=[]
        group_answer=[]
        group_logprobs=[]
        group_reward=[]
        for _ in range(self.groups):
            #生成回答和评分
            answer,probs,mask=self.get_answer_probs(prompt,device)#answer=(batch,answer_len)=probs
            reward=self.Reward_get(prompt,answer,target)
            
            group_logprobs.append(torch.tensor(probs,device=device))#group=(group,batch,answer_len) 把probs的list转为tensor防止后面出错
            group_answer.append(answer)
            group_reward.append(reward)#reward=(group,batch)=list[tensor]
        #计算组内相对优势  list变tensor计算，再变回list
        reward_tensor=torch.stack(group_reward)#把多个由tensor组成的list变为更高维的tensor
        mean=reward_tensor.mean(dim=0)#计算均值mean=(batch)
        std=reward_tensor.std(dim=0)#计算标准差
        group_advantage=[(r-mean)/(std+1e-8) for r in reward_tensor] #组回list advantage=(group,batch)
        #组成Data  data=[tensor,tensor]后面需要变为data=tensor(tensor,tensor)
        for i in range(self.groups):
            all_data.append({
                "prompt":prompt,
                "answer":group_answer[i],
                "advantage":group_advantage[i],
                "old_logprobs":group_logprobs[i],
                "reward":group_reward[i],
                "mask":torch.tensor(mask,dtype=torch.float32,device=device),#mask=(group,batch,answer_len)因为mask是list(list)所以需要变为list(tensor)
            })
        return all_data
    
    def compute_loss(self,all_data,ref_logprobs=None):#ref_logprobs通常是冻结的SFTmodel 使用时要先冻结一个
        device=all_data[0]["prompt"].device
        #取数据
        prompts=all_data[0]["prompt"]
        answer=[all_data[i]["answer"] for i in range(self.groups)]
        old_logprobs=[all_data[i]["old_logprobs"] for i in range(self.groups)] #group=(group,batch,answer_len)
        advantage=[all_data[i]["advantage"] for i in range(self.groups)] #advantage=(group,batch)
        mask=[all_data[i]["mask"] for i in range(self.groups)] #mask=(group,batch,answer_len)
        reward=[all_data[i]["reward"] for i in range(self.groups)]
        #计算新的轨迹
        new_logprobs=[]
        for i in range(self.groups):
            new_probs=self.get_new_probs(prompts,answer[i],device)#用old answer 在new model中寻找概率
            new_logprobs.append(new_probs)
        #用tensor加进去，防止计算图断掉
        new_logprobs=torch.stack(new_logprobs)
        old_logprobs=torch.stack(old_logprobs)
        advantage=torch.stack(advantage)
        mask=torch.stack(mask)
        reward=torch.stack(reward).float()
        #计算ratio
        ratio=torch.exp(new_logprobs-old_logprobs)
        #计算loss
        loss_pg=ratio*advantage.unsqueeze(-1)
        loss_clip=torch.clamp(ratio,1-self.clip,1+self.clip)*advantage.unsqueeze(-1)
        loss_ppo=torch.min(loss_pg,loss_clip)
        #计算KL散度(包括πold和πref) 使用时自己选择
        if ref_logprobs is not None:
            ratio_ref=torch.exp(ref_logprobs-new_logprobs)
            kl=ratio_ref-torch.log(ratio_ref)-1
            
        else:
            kl=0
        total_loss=loss_ppo-self.kl_b*kl#total_loss=(group,batch,answer_len)
        total_loss=total_loss*mask
        loss=-total_loss.sum()/mask.sum()
        kl_mean=(kl*mask).sum()/mask.sum()
        R=reward.mean()
        return loss,kl_mean,R
        
def train(epoch_num,sft_resume,rl_resume=None):
    loss_record=[]
    start_epoch=0
    
    device="cuda" if torch.cuda.is_available() else "cpu"
    scaler=torch.amp.GradScaler("cuda")
    files=get_txt_files("/root/autodl-tmp/.autodl/data")
    
    vocab_size=config.vocab_size
    model=Transformer(config).to(device)
    ref_model=Transformer(config).to(device)
    #冻结SFTmodel
    for p in ref_model.parameters():
        p.requires_grad=False
    ref_model.eval()
    max_len=512
    groups=4
    optimizer=torch.optim.AdamW(model.parameters(),lr=5e-7,weight_decay=0.01)
    
    if rl_resume is not None:
        start_epoch,_=load_checkpoint(
            rl_resume,
            model,
            None,
            device
        )
        _,_=load_checkpoint(
            sft_resume,
            ref_model,
            None,
            device
        )
    else:
        start_epoch,_=load_checkpoint(
            sft_resume,
            model,
            None,
            device
        )
        _,_=load_checkpoint(
            sft_resume,
            ref_model,
            None,
            device
        )
    
    dataset=GRPODatasetream(tokenizer,files,max_len)
    loader=DataLoader(dataset,batch_size=12,num_workers=0,pin_memory=True)
    grpo=GRPOtrain(groups=groups,clip_coef=0.2,model=model,tokenizer=tokenizer,max_len=max_len)
    
    for epoch in range(start_epoch,epoch_num):
        start=time.time()
        step_loss=0
        epoch_loss=0
        for step,(x,y) in enumerate(loader):
            x=x.to(device)
            y=y.to(device)
            all_data=grpo.collect_data(x,y)
            with torch.amp.autocast("cuda"):
                answer=[all_data[i]["answer"] for i in range(groups)]
                ref_logprobs=[]
                with torch.no_grad():
                    for i in range(groups):
                        ref_probs=grpo.get_new_probs(x,answer[i],device,ref_model)
                        ref_logprobs.append(ref_probs)
                ref_logprobs=torch.stack(ref_logprobs)
                loss,kl,reward=grpo.compute_loss(all_data,ref_logprobs)
            
            optimizer.zero_grad()#清空上一轮梯度  防止累计
            #scale是防止fp16训练中梯度消失的情况
            scaler.scale(loss).backward()#为了防止梯度太小fp16表示不了，所以用scale参数放大梯度进行backward
            
            scaler.unscale_(optimizer)#把梯度还原为原始大小
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )#把所有梯度放到1这个区间内，用正则化，防止梯度爆炸
            
            scaler.step(optimizer)#检查有无NaN，如果有就跳过，没有就更新参数
            scaler.update()#根据情况调整scale大小，如果很多步没有nan，就放大scale
            step_loss+=loss.item()
            epoch_loss+=loss.item()
            if (step+1)%1==0:
                loss_record.append(step_loss/1)
                print(f"epoch={epoch} | step={step} | loss={loss.item():.4f} | 1step_loss={step_loss/1:.4f} | kl={kl.item()} | reward={reward.item()}")
                step_loss=0
                end=time.time()
                print(f"1step耗时:{end-start:.2f}s")
                start=time.time()
                
            
            if (step+1)%200==0:
                    torch.save(
                {
                    "epoch":epoch,
                    "model":model.state_dict(),
                    "optimizer":optimizer.state_dict(),
                    "step":step
                },
                f"/root/autodl-tmp/.autodl/checkpoints/GRPO-epoch{epoch}-step{step}.pth"
            )
                    
        with open("/root/autodl-tmp/.autodl/checkpoints/GRPO-loss.json","w") as f:#以写入模式打开，如果文件不存在就创建
            json.dump(loss_record,f)#写入json文件
        epoch_loss=epoch_loss/(step+1)
        
        print(f"epoch_loss={epoch_loss:.4f}")
        
    torch.save(
        {
            "epoch":epoch,
            "model":model.state_dict(),
            "optimizer":optimizer.state_dict(),
            "step":step
        },
        f"/root/autodl-tmp/.autodl/checkpoints/GRPO-last-checkpoint.pth"
    )#所有epoch跑完进行记录checkpoints
            
if __name__ == "__main__":
    train(2,sft_resume="/root/autodl-tmp/.autodl/checkpoints/minimind-dataset-sft-checkpoints.pth")