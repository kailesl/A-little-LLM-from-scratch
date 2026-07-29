#最新更改日期2026.6.27
#2026.7.3 dataset读入改为shuffle slice
#2026.7.4 加入了加载checkpoints
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

config=modelconfig()
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
#dataset处理+流式切块
class tokenstreamdataset(IterableDataset):
    def __init__(self,tokenizer,files,max_len,index_len=10,stride=128,slices=20):
        self.tokenizer=tokenizer
        self.max_len=max_len
        self.files=files
        self.stride=stride
        self.slice=slices
        self.index_len=index_len#每个文件读多少个slice
    
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
                        ids=self.file_index[name].pop()
                        f.seek(ids)
                        token_buffer=[]
                        for _ in range(self.slice):
                            line=f.readline()
                            if not line:
                                break
                            line=line.decode("utf-8")
                            obj=json.loads(line)
                            text=obj["text"]
                            tokens=self.tokenizer.encode(text)
                            tokens.append(self.tokenizer.eos_id)
                            token_buffer.extend(tokens)
                            if len(token_buffer)>1000000:
                                print("token_buffer=",len(token_buffer))
                        while len(token_buffer)>=self.max_len+1:
                            x=torch.tensor(token_buffer[:self.max_len],dtype=torch.long)
                            y=torch.tensor(token_buffer[1:self.max_len+1],dtype=torch.long)
                            yield x,y
                            token_buffer=token_buffer[self.stride:]
                            
def load_checkpoint(path,model,optimizer,device):
    checkpoint=torch.load(path,map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    epoch=checkpoint["epoch"]
    step=checkpoint["step"]
    print("成功加载checkpoint")
    return epoch,step

def train(epoch_num,resume):
    start_epoch=0
    #超参数设置
    device="cuda" if torch.cuda.is_available() else "cpu"
    max_len=256
    model=Transformer(config).to(device)
    
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=0.01)
    
    if resume is not None:
        #读取checkpoints
        start_epoch,_=load_checkpoint(
            resume,
            model,
            optimizer,
            device
        )
    scaler=torch.amp.GradScaler("cuda")
    #读取文件夹
    files=get_txt_files(
        ""
    )
    
    #dataset处理
    dataset=tokenstreamdataset(tokenizer,files,max_len)
    loader=DataLoader(dataset,batch_size=80,num_workers=0,pin_memory=True)
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
                    out.view(-1,config.vocab_size),
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
    train(4,resume="")
#注释代码区

#data切块
#def token_process(tokens,max_len):
#    data=[]
#    for i in range(0,len(tokens),max_len//2):
#        chunk=tokens[i:i+max_len]
#        data.append(chunk)
#    return data
 
#def file_read(folder,window_size):
#    with open(folder,"r",encoding="utf-8") as f:
#        text=f.read
        
    # for epoch in range(epoch_num):
    #     for step,(x,y) in enumerate(loader):
    #         x=x.to(device)
    #         y=y.to(device)
            
    #         out=model(x)
            
    #         loss=F.cross_entropy(
    #             out.view(-1,vocab_size),
    #             y.view(-1),
    #             ignore_index=-100
    #         )
            
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         if step%100==0:
    #             print(f"epoch={epoch} | step={step} | loss={loss.item():.4f}")
            
    #         if step%1000==0:
    #             torch.save(
    #                 {
    #                     "epoch":epoch,
    #                     "model":model.state_dict(),
    #                     "optimizer":optimizer.state_dict(),
    #                     "step":step
    #                 },
    #                 f""
    #             )
    
    # #读取文件
    # tokens=[]
    # for i,file in enumerate(files):
    #     with open(file,"r",encoding="utf-8") as f:
    #         text=f.read()
    #         text=tokenizer.encode(text)
    #         tokens.extend(text)
    #         print(f"text{i}")
    # #dataset处理
    # dataset=GPTDataset(tokens,max_len)
    # loader=DataLoader(dataset,batch_size=8,shuffle=True)
    # text=json.loads(line)
                        # text=text["text"]
                        # tokens=self.tokenizer.encode(text)
                        # tokens.append(self.tokenizer.eos_id)
                        # token_buffer.extend(tokens)
                        # while len(token_buffer)>=self.max_len+1:
                        #     x=token_buffer[:self.max_len]
                        #     y=token_buffer[1:self.max_len+1]
                        #     shuffle_buffer.append((x,y))
                        #     if len(shuffle_buffer)>=self.buffer_size:
                        #         idx=random.randrange(len(shuffle_buffer))
                        #         fo,g=shuffle_buffer.pop(idx)
                        #         yield torch.tensor(fo,dtype=torch.long),torch.tensor(g,dtype=torch.long)
                        #     token_buffer=token_buffer[self.stride:]
            # else:
            #     with open(file,"r",encoding="utf-8") as f:
            #         while True:
            #             text=f.read(self.chunk_size)
            #             if not text:
            #                 break
            #             tokens=self.tokenizer.encode(text)
            #             token_buffer.extend(tokens)
            #             while len(token_buffer)>=self.max_len+1:
            #                 x=torch.tensor(token_buffer[:self.max_len])
            #                 y=torch.tensor(token_buffer[1:self.max_len+1])
            #                 shuffle_buffer.append((x,y))
            #                 if len(shuffle_buffer)>=self.buffer_size:
            #                     idx=random.randrange(len(shuffle_buffer))
            #                     fo,g=shuffle_buffer.pop(idx)
            #                     yield fo,g
            #                 token_buffer=token_buffer[self.stride:]
        # random.shuffle(shuffle_buffer)
        # while shuffle_buffer:
        #     fo,g=shuffle_buffer.pop()
        #     yield fo,g
        #for file in files:
        #     file_name=os.path.basename(file)#取文件名
        #     index=self.file_index[file_name]
        #     with open(file,"rb") as f:
        #         for ids in index:
        #             f.seek(ids)
        #             token_buffer=[]
        #             for i in range(self.slice):
        #                 line=f.readline()
        #                 if not line:
        #                     break
        #                 line=line.decode("utf-8")
        #                 obj=json.loads(line)
        #                 text=obj["text"]
        #                 tokens=self.tokenizer.encode(text)
        #                 tokens.append(self.tokenizer.eos_id)
        #                 token_buffer.extend(tokens)
        #                 if len(token_buffer)>1000000:
        #                     print("token_buffer=",len(token_buffer))
        #             while len(token_buffer)>=self.max_len+1:
        #                 x=torch.tensor(token_buffer[:self.max_len],dtype=torch.long)
        #                 y=torch.tensor(token_buffer[1:self.max_len+1],dtype=torch.long)
        #                 yield x,y
        #                 token_buffer=token_buffer[self.stride:]
        #             self.file_index[file_name].pop()
