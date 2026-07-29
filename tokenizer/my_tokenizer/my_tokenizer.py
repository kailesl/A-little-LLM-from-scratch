import os
import random
import pickle

class my_tokenizer:
    def get_stats(self,ids):#输入的是一长串的数字串
        counts={}
        for i in range(len(ids)-1):
            if (ids[i],ids[i+1]) not in counts:
                counts[(ids[i],ids[i+1])]=0
            counts[(ids[i],ids[i+1])]+=1
        return counts#假设output=(254,256):2
    
    def merge(self,ids,vocab,merges):#merge是词对编码，用于encode;vocab是编码对词，用于decode
        counts=self.get_stats(ids)
        if not counts:
            return vocab,merges,ids
        pair=max(counts,key=counts.get)
        new_id=256+len(merges)
        merges[pair]=new_id
        vocab[new_id]=vocab[pair[0]]+vocab[pair[1]]#对字节进行拼接
        i=0
        new_ids=[]
        while i<len(ids):
            if i+1<len(ids) and (ids[i],ids[i+1])==pair:
                new_ids.append(merges[pair])
                i+=2
            else:
                new_ids.append(ids[i])
                i+=1
        return vocab,new_ids,merges
    
    def get_txt_files(self,folder):
        txt_files=[]
        for file_name in os.listdir(folder):
            if file_name.endswith((".txt",".jsonl")):
                txt_files.append(
                    os.path.join(
                        folder,
                        file_name
                    )
                )
        return txt_files

    def train(self,vocab_size,data_folder,chunk_size=8192):
        vocab={idx:bytes([idx]) for idx in range(256)}
        merges={}
        files=self.get_txt_files(data_folder)
        sum_idx=[]
        #read_len=(20//len(files))*1048576
        read_len=(20*1024*1024)//len(files)
        for file in files:
            len_rb=0
            with open(file,"rb") as f:#字节阅读，utf编码
                size=os.path.getsize(file)
                if size<=chunk_size:
                    continue#放置文件过小
                while len_rb<=read_len:
                    random_pos=random.randint(0,size-chunk_size)
                    f.seek(random_pos)#随机挑选位置开始阅读
                    text=f.read(chunk_size)
                    len_rb+=chunk_size
                    sum_idx.extend(text) 
        #显示训练进度
        step=0  
        while len(vocab)<vocab_size:
            vocab,sum_idx,merges=self.merge(sum_idx,vocab,merges)
            step+=1
            print(
                f"\r训练进度: {len(vocab)}/{vocab_size} "
                f"({len(vocab)/vocab_size*100:.4f}%)",
                end=""
            )
        self.vocab=vocab
        self.merges=merges
        return vocab,merges
    
    def encode(self,text):#merge是词对编码，用于encode;vocab是编码对词，用于decode
        ids=list(text.encode("utf-8"))
        for (pair,new_id) in self.merges.items():
            while True:
                new_ids=[]
                i=0
                found=False
                while i<len(ids):
                    if i+1<len(ids) and (ids[i],ids[i+1])==pair:
                        new_ids.append(new_id)
                        i+=2
                        found=True
                    else:
                        new_ids.append(ids[i])
                        i+=1
                ids=new_ids#扫完一遍赋值一遍，做到反复扫
                if found==False:
                    break
        return ids
    
    def decode(self,ids):#merge是词对编码，用于encode;vocab是编码对词，用于decode
        text_bytes=[]
        for id in ids:
            text_bytes.append(self.vocab[id])
        return b''.join(text_bytes).decode("utf-8")
    
    def save(self,path):#保存tokenizer
        with open(path,"wb") as f:
            pickle.dump(
                {
                "vocab":self.vocab,
                "merges":self.merges
                },
                f        
            )
    
    def load(self,path):#加载
        with open(path,"rb") as f:
            data=pickle.load(f)
        self.vocab=data["vocab"]
        self.merges=data["merges"]

    def __len__(self):
        return len(self.vocab)