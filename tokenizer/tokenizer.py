from tokenizers import Tokenizer

class mytokenizer:
    def __init__(self,path):
        self.tokenizer=Tokenizer.from_file(path)
        self.eos_id=self.tokenizer.token_to_id("<eos>")
        self.pad_id=self.tokenizer.token_to_id("<pad>")
        self.unk_id=self.tokenizer.token_to_id("<unk>")
        self.user_id=self.tokenizer.token_to_id("<user>")
        self.assistant_id=self.tokenizer.token_to_id("<assistant>")
        self.think_id=self.tokenizer.token_to_id("<think>")
        self.think_sla=self.tokenizer.token_to_id("</think>")
    def encode(self,text):
        return self.tokenizer.encode(text).ids
    def decode(self,ids):
        return self.tokenizer.decode(ids)
    def __len__(self):
        return self.tokenizer.get_vocab_size()