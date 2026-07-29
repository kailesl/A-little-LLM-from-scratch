#初次编写2026.7.8
import os
import sys
import torch.nn.functional as F
import torch as torch
import random
import time
import json

sys.path.append("")
sys.path.append("")
sys.path.append("")

from torch.utils.data import IterableDataset
from torch.utils.data import DataLoader
from model_me import Transformer
from tokenizer import mytokenizer
from config import modelconfig

config=modelconfig
tokenizer=mytokenizer("")

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

class SFTDatastream(IterableDataset):
    def __init__(self,tokenizer,files,max_len,index_len=10,slices=20):
        self.tokenizer=tokenizer
        self.files=files
        self.slice=slices
        self.max_len=max_len
        self.index_len=index_len
        
    def process_sft(self,data):
        inputs=[]
        label=[]
        conversation=data["conversations"]
        for item in conversation:
            if item["role"]=="user":
                content=item["content"]
                idm=[self.tokenizer.user_id]
                idm+=self.tokenizer.encode(content)
                inputs.extend(idm)
                label.extend([-100]*len(idm))
            elif item["role"]=="assistant":
                content=item["content"]
                if item.get("reasoning_content") is not None:
                    reasoning_content=item["reasoning_content"]
                    id_re=[self.tokenizer.think_id]
                    id_re+=self.tokenizer.encode(reasoning_content)
                    id_re.append(self.tokenizer.think_sla)
                    inputs.extend(id_re)
                    label.extend(id_re)
                id_con=[self.tokenizer.assistant_id]
                id_con+=self.tokenizer.encode(content)
                id_con.append(self.tokenizer.eos_id)
                inputs.extend(id_con)
                label.extend(id_con)
        return inputs,label
    
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
                            inputs,label=self.process_sft(obj)
                            if len(inputs)>self.max_len:
                                inputs=inputs[:self.max_len]
                                label=label[:self.max_len]
                                
                            pad_len=self.max_len-len(inputs)
                            inputs.extend([self.tokenizer.pad_id]*pad_len)
                            label.extend([-100]*pad_len)
                            x=torch.tensor(inputs[:-1],dtype=torch.long)
                            y=torch.tensor(label[1:],dtype=torch.long)
                            yield x,y
                            
def train(epoch_num,resume):
    loss_history=[]
    start_epoch=0
    #超参数设置
    device="cuda" if torch.cuda.is_available() else "cpu"
    
    max_len=2048
    vocab_size=config.vocab_size
    
    model=Transformer(config).to(device)
    
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-6,weight_decay=0.01)
    
    if resume is not None:
        #读取checkpoints
        start_epoch,_=load_checkpoint(
            resume,
            model,
            None,#SFT阶段，不要加载优化器
            device
        )
    scaler=torch.amp.GradScaler("cuda")
    #读取文件夹
    files=get_txt_files(
        ""
    )
    
    #dataset处理
    dataset=SFTDatastream(tokenizer,files,max_len)
    loader=DataLoader(dataset,batch_size=16,num_workers=0,pin_memory=True)
    #打印模型参数量
    total_params=sum(p.numel()for p in model.parameters())
    print(total_params)
    
    for epoch in range(start_epoch,epoch_num):
        start=time.time()
        step_loss=0
        epoch_loss=0
        for step,(x,y) in enumerate(loader):
            x=x.to(device)
            y=y.to(device)
            with torch.amp.autocast("cuda"):
                out,_=model(x,past_kv=None)
                
                loss=F.cross_entropy(
                    out.view(-1,vocab_size),
                    y.view(-1),
                    ignore_index=-100
                )
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )#梯度裁剪放置爆炸
            
            scaler.step(optimizer)
            scaler.update()
            step_loss+=loss.item()
            epoch_loss+=loss.item()
            if (step+1)%100==0:
                loss_history.append(step_loss/100)
                print(f"epoch={epoch} | step={step} | loss={loss.item():.4f} | 100step_loss={step_loss/100:.4f}")
                step_loss=0
                end=time.time()
                print(f"100 step耗时:{end-start:.2f}s")
                start=time.time()
                
            
            if (step+1)%20000==0:
                    torch.save(
                {
                    "epoch":epoch,
                    "model":model.state_dict(),
                    "optimizer":optimizer.state_dict(),
                    "step":step
                },
                f""
            )
        with open("","w") as f:
            json.dump(loss_history,f)
        epoch_loss=epoch_loss/(step+1)
        print(f"epoch_loss={epoch_loss:.4f}")
    torch.save(
        {
            "epoch":epoch,
            "model":model.state_dict(),
            "optimizer":optimizer.state_dict(),
            "step":step
        },
        f""
    )
            
if __name__ == "__main__":
    train(2,resume="")                                
