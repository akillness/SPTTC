
import torch
import torch.nn as nn
import myMultiHeadAttention

class FeedForwrad(nn.Module):
    def __init__(self, embed_dim, ff_dim):
        super().__init__()

        # Feedforward 개념 구현
        self.net = nn.Sequential(
            nn.Linear(embed_dim,ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim,embed_dim)
        )

    def forward(self,x):
        return self.net(x)

class TransformBlock(nn.Module): 
    def __init__(self, embed_dim, multi_heads):
        super().__init__()

        self.layer_norm1 = nn.LayerNorm(embed_dim) # Layer Norm을 통해 scale, shift 학습
        self.multihead_attenstion = myMultiHeadAttention.MultiHeadAttention(embed_dim,multi_heads)

        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.feed_forward = FeedForwrad(embed_dim, 4*embed_dim)

    def forward(self,x):
        x = x + self.multihead_attenstion(self.layer_norm1(x))
        x = x + self.feed_forward(self.layer_norm2(x))
        return x