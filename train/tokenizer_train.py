from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers import decoders
import os
import json
import random

tokenizer = Tokenizer(
    BPE(
        unk_token="<unk>"
    )
)
tokenizer.pre_tokenizer = ByteLevel()
tokenizer.decoder=decoders.ByteLevel()
trainer=BpeTrainer(
    vocab_size=10000,
    special_tokens=[
        "<pad>",
        "<unk>",
        "<bos>",
        "<eos>",
        "<user>",
        "<assistant>",
        "<think>",
        "</think>"
    ]
)

def get_txt_files(folder):
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
def iter_text():
    data_folder=""
    files=get_txt_files(data_folder)
    data_len=len(files)
    max_bytes=1024*1024*1024//data_len
    all_line=[]
    for file in files:
        read_b=0
        with open(file,"r",encoding="utf-8") as f:
            for line in f:
                if random.random()>0.5:
                    continue
                read_b+=len(line.encode("utf-8"))
                if read_b>=max_bytes:
                    break
                if file.endswith(".jsonl"):
                    try:
                        obj=json.loads(line)
                        all_line.append(obj["text"]) 
                    except:
                        pass
                else:
                    text = line.strip()
                    if text:
                        all_line.append(text)
    for line in all_line:
        yield line
            
tokenizer.train_from_iterator(iter_text(),trainer=trainer)

tokenizer.save(
    ""
)
