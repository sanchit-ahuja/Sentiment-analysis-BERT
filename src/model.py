import transformers
import torch.nn as nn
import config

class BERTBaseUncased(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH)
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(768,1) #Binary classification problem

    def forward(self,ids,mask,token_type_ids):
        _,o2 = self.bert(ids,attention_mask = mask, token_type_ids = token_type_ids)
        #out1 sequence of hidden states for each and every hidden token 512 vectors of size 768
        # sequence of hidden-states at the output of the last layer shape = [batch_size,sequence_length, hidden_size]
        #out2 pooler output shape = [batch_size, hidden_size] last layer hidden-state of the first token of the sequence further processed
        bo = self.bert_drop(o2)
        output = self.out(bo)
        return output
