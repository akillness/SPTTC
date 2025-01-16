import torch
import torch.nn as nn

from mySelfAttention import SelfAttention

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, multi_heads):
        super().__init__()
        attent_dim = embed_dim // multi_heads # h ( muti-head 의 개수 ) 에 따라 attentnion layer의 linear dimension 설정
        self.attentions = nn.ModuleList([SelfAttention(embed_dim,attent_dim) for _ in range(multi_heads)])
        self.fc = nn.Linear(embed_dim,embed_dim) # attentions 를 하나의 linear 로 concatnate 하기 위한 fullyconnectied Layer 설정

    def forward(self,x):
        head_outputs = []
        for attention in self.attentions:
            head_outputs.append(attention(x)) # attention feature

        concatenated_head = torch.cat(head_outputs)
        multiHeadAttentions = self.fc(concatenated_head) # fully connected layer 로 설정

        return multiHeadAttentions

