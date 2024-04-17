import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

import torch.nn.functional as F

# XOR 문제 1개의 input/hidden/output 레이어의 MLP를 classification으로 활용하여 XOR 문제를 푼다

X = np.array([[0,0],[1,0],[0,1],[1,1]]) # 입력
Y = np.array([[1,0],[0,1],[0,1],[1,0]]) # 출력

print('numpy:',X,Y)

X = torch.tensor(X, dtype=torch.float32) # numpy -> torch (double)
Y = torch.tensor(Y, dtype=torch.float32) # numpy -> torch (double)

print('torch:',X,Y)

# Custorm Peceptron
class XOR(nn.Module):
    def __init__(self, input_dim = 2, output_dim=1):
        super(XOR, self).__init__()
        self.lin1 = nn.Linear(input_dim, 2)
        self.lin2 = nn.Linear(2, output_dim)
    
    def forward(self, x):
        x = self.lin1(x)
        x = F.sigmoid(x)
        x = self.lin2(x)
        x = F.sigmoid(x)
        return x


print('''[참고]
      신경망 weight 초기화를 위한 권장되는 방법
      xavier uniform 이나 normal을 사용하는 것.(코드 참고)
      XOR 문제도 local minima가 있음을 기억할 것.
      hyperparameter 변경을 통해 극복 가능할지도?''')

model = XOR()
# 학습을 위한 준비
opt = torch.optim.Adam(model.parameters(),lr=0.09)
loss_func = torch.nn.MSELoss()

# [학습 100000회 반복]
for i in (pbar := tqdm(range(50000))):
    for s in range(X.size(0)):
        opt.zero_grad()
    
        # 네트워크 동적 계산
        yhat = model(X[s])

        loss= loss_func.forward(yhat,Y) # mean-square loss

        loss.backward()
        opt.step()

    if i%1000 == 0:
        pbar.set_description(f'loss:{loss.item()}')

print('입력 데이터:',X)
print('희망 데이터:',Y)

with torch.no_grad(): # 학습 할 필요가 없어서 gradient를 적용하지 않고 계산만 하길 원하는 경우 no_grad 사용
    yhat = model(X)
    xor_classifier_value = yhat

print('XOR 신경망 결과:',xor_classifier_value)

