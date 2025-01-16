import torch
import torch.nn as nn

embed_dim = 20
attent_dim = 5

# 요약 : 하나의 문장에서 단어들간의 관계를 파악하고, 단어의 Vector Representation을 update 하여 재배치 한다.
# ㄴ 가능한 이유는 SelfAttention 방법론 
class SelfAttention(nn.Module):
    def __init__(self,embed_dim, attent_dim):
        super().__init__()
        
        # self-attention의 query, key, value 를 linear(재배치) 하여 선언
        self.query = torch.nn.Linear(embed_dim, attent_dim,bias=False)  # LayerNorm 연산시 bias는 의미없는 값이 되므로, bias 는 false로 함
        self.key = torch.nn.Linear(embed_dim, attent_dim,bias=False)
        self.value = torch.nn.Linear(embed_dim, attent_dim,bias=False)
        # step.1 ) q*k^T = score matrix 
        # step.2 ) query 의 element 와 관련된 key element 간의 관계(Attention) 에 따라 high score 또는 low score
        # step.3 ) score matrix 를 row의 방향으로 softmax ( row의 총합을 1로 만들기 위함 )
        # step.4 ) softmax 적용된 q*k^T matrix 에 value 를 합성곱하여 word 마다의 vector representation 이 값을 갖게됨
        
        ## tip ) 1/dk^(1/2) 의 값은 Attention(Q,K,V) 의 값이 극단적이지 않도록 smoothing 하는 역할 값
    
    def forward(self,x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        score = torch.matmul(q,k.transpose(-2,-1)) # step.1, 2
        score = score / k.size(-1) ** 0.5 # smoothing

        attention_weights = torch.nn.Softmax(score,dim=-1) # step.3
        weight_values = torch.matmul(attention_weights,v) # step.4

        return weight_values # SelfAttention(Q,K,V)