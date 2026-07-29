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
# 数据样式
# {
#   "conversations": [
#     {
#       "role": "user",
#       "content": "人工智能如何重塑人类创造力边界？跨学科视角下，算法生成内容与人类艺术表达的共生关系如何演变？"
#     },
#     {
#       "role": "assistant",
#       "content": "人工智能通过算法优化创作流程、提供新表现形式，重塑人类创造力边界。在艺术领域，AI生成图像、音乐和文字成为创作工具，但人类艺术家仍主导审美判断与情感表达。跨学科协作下，算法与人类创作形成共生关系：AI可作为灵感启发者，如DALL·E生成艺术，而人类则确保作品的哲学深度与人文价值。这种互动推动艺术形式突破传统限制，但需警惕技术对原创性的挑战。未来趋势显示，AI将更深入参与创作过程，但人类创造力的核心仍在于情感与思想的独创性。",
#       "reasoning_content": "好的，用户问的是人工智能如何重塑人类创造力边界，以及算法生成内容与人类艺术表达的共生关系。首先，我需要明确这两个问题的核心。人工智能对创造力的影响可能涉及工具、启发、甚至重新定义创作本身。然后，跨学科视角下，算法和人类艺术的互动，可能需要讨论技术如何辅助艺术创作，或者两者如何相互影响。\n\n用户可能希望了解AI在艺术中的应用，比如生成艺术、音乐、文学，以及这些技术如何改变艺术家的创作方式。同时，共生关系可能涉及艺术家如何利用AI工具，或者AI如何与人类合作，共同创作。\n\n需要考虑用户可能的背景，可能是学生、艺术家或研究者，他们想了解AI在艺术领域的最新发展。回答要准确，但也要简洁，控制在200-300字。需要提到具体的例子，比如GANs生成图像，AI辅助创作，或者AI作为创作伙伴。同时，要指出挑战，比如原创性问题，以及未来趋势，比如AI与人类协作的深化。\n\n还要注意避免过于技术化的术语，保持易懂。可能需要强调共生关系中的相互影响，比如AI提供新工具，人类保持创意方向。最后，总结AI如何扩展创造力边界，同时保持人类艺术的独特性。"
#     },
#     {
#       "role": "user",
#       "content": "你今天在做什么？"
#     },
#     {
#       "role": "assistant",
#       "content": "我正在处理用户的问题，提供关于人工智能与创造力的深度分析。如果您有其他问题，欢迎随时提出。"
#     }
#   ]
# }
