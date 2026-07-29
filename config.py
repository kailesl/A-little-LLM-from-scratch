import sys
sys.path.append("")
from tokenizer import mytokenizer
tokenizer=mytokenizer("")
class modelconfig:
    def __init__(self):
        #tokenizer
        self.vocab_size=len(tokenizer)
        
        #transformer
        self.n_layer=8
        self.num_head=8
        self.d_model=768
        
        #attention
        self.flash_attention=True
        self.kv=4
        
        #MoE
        self.moe={
            "num_experts":8,
            "topk":2
        }
        
        self.moe=None
