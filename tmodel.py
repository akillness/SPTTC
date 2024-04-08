import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

# XOR 문제 1개의 input/hidden/output 레이어의 MLP를 classification으로 활용하여 XOR 문제를 푼다

X = np.array([[0,0],[1,0],[0,1],[1,1]]) # 입력
Y = np.array([[1,0],[0,1],[0,1],[1,0]]) # 출력

print('numpy:',X,Y)

X = torch.tensor(X, dtype=torch.float64) # numpy -> torch (double)
Y = torch.tensor(Y, dtype=torch.float64) # numpy -> torch (double)

print('torch:',X,Y)

# Custorm Peceptron
class CLinear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # self.W = torch.FloatTensor(input_dim,output_dim)
        # self.B = torch.FloatTensor(output_dim)

        # nn.Module을 상속받아 선형 계층(linear layer)를 구현여 계산을 수행할 수 있지만 이 방법으로는 학습을 진행할 수 없습니다. 
        # W(weight)와 b(bias)가 학습이 가능한 parameter로 설정되어있지 않기 때문입니다. 
        # nn.Parmater을 활용하여 학습이 가능한 parameter로 인식시켜줄 수 있습니다.
        self.W = nn.Parameter(torch.FloatTensor(input_dim,output_dim))
        self.B = nn.Parameter(torch.FloatTensor(output_dim))
    
    # nn.Module을 상속받은 객체는 __call__ 함수가 forward 함수와 mapping 되어 있어 foward 를 따로 호출할 필요가 없다는 것입니다.
    def forward(self,x:torch.tensor):
        y = x@self.W+self.B # or torch.matmul(x,self.W)+B
        return y

class MyLinear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim,output_dim)
        self.activation = nn.Sigmoid()

    def forward(self,x):
        y = self.linear(x)
        out = self.activation(y)
        return out


w1 = torch.tensor(np.random.normal(size=(2,2)), requires_grad=True) # 학습을 위해 requires_grad 옵션 추가
b1 = torch.tensor(np.random.normal(size=(1,2)), requires_grad=True) # 학습을 위해 requires_grad 옵션 추가

print('w1:',w1) #  tensor([[ 0.8241, -1.8039], [ 0.6684, -0.6701]], dtype=torch.float64, requires_grad=True)

w2 = torch.tensor(np.random.normal(size=(2,2))) # 학습 불가
b2 = torch.tensor(np.random.normal(size=(1,2))) # 학습 불가

print(w2) # tensor([[ 1.2678,  0.4371], [-1.4530, -0.9344]], dtype=torch.float64) # require_grad가 없음

# 학습을 위해 requires_grad 추가
w2.requires_grad_(True)

print(w2) # tensor([[ 1.2678,  0.4371], [-1.4530, -0.9344]], dtype=torch.float64, requires_grad=True) # requires_grad가 있음

# torch 함수 특징, 함수 끝에 '_'가 있으면 변수 자체에 바로 적용됨. (ex: require_grad, require_grad_)
print(b2) # tensor([[-0.7115], [-0.0490]], dtype=torch.float64)

try:
    b2.requires_grad(True) # 함수 명령어는 있으나 이렇게 사용하면 에러
except Exception as e:
    print(e)

print('b2:',b2) # tensor([[-0.7115], [-0.0490]], dtype=torch.float64) # 아직 적용 안됨
b2.requires_grad_(True)
print('b2:',b2) # tensor([[-0.7115], [-0.0490]], dtype=torch.float64, requires_grad=True) # 적용 됨

print('''[참고]
      신경망 weight 초기화를 위한 권장되는 방법
      xavier uniform 이나 normal을 사용하는 것.(코드 참고)
      XOR 문제도 local minima가 있음을 기억할 것.
      hyperparameter 변경을 통해 극복 가능할지도?''')

# torch.nn.init.xavier_normal_(w1)
# torch.nn.init.xavier_normal_(b1)
# torch.nn.init.xavier_normal_(w2)
# torch.nn.init.xavier_normal_(b2)

# 학습을 위한 준비
opt = torch.optim.Adam([w1,b1,w2,b2]) # optimizer는 Adam, hyperparameter는 임의 입력이 가능하지만, 일단 default값 사용 (learning rate:0.001, beta = (0.9, 0.999), eps=1e-8)

# [학습 1회]
opt.zero_grad()

# 네트워크 동적 계산
xor_mlp_input_layer = X
xor_mlp_hidden_layer = torch.sigmoid(xor_mlp_input_layer@w1+b1)
xor_mlp_output_layer = torch.sigmoid(xor_mlp_hidden_layer@w2+b2)

print('out:',xor_mlp_output_layer) # 현재 모델의 output

loss = torch.mean((xor_mlp_output_layer-Y)**2) # mean-square loss

print('loss:',loss) # 현재 모델의 loss
print('loss(value only):',loss.item())

loss.backward()
print('b2(before):',b2)
opt.step()
print('b2(after):',b2)

# [학습 100000회 반복]
for i in (pbar := tqdm(range(100000))):
    opt.zero_grad()

    # 네트워크 동적 계산
    xor_mlp_input_layer = X
    xor_mlp_hidden_layer = torch.sigmoid(xor_mlp_input_layer@w1+b1)
    xor_mlp_output_layer = torch.sigmoid(xor_mlp_hidden_layer@w2+b2)

    loss = torch.mean((xor_mlp_output_layer-Y)**2) # mean-square loss

    loss.backward()
    opt.step()

    if i%1000 == 0:
        pbar.set_description(f'loss:{loss.item()}')

print('입력 데이터:',X)
print('희망 데이터:',Y)

with torch.no_grad(): # 학습 할 필요가 없어서 gradient를 적용하지 않고 계산만 하길 원하는 경우 no_grad 사용
    xor_mlp_input_layer = X
    xor_mlp_hidden_layer = torch.sigmoid(xor_mlp_input_layer@w1+b1)
    xor_mlp_output_layer = torch.sigmoid(xor_mlp_hidden_layer@w2+b2)
    xor_classifier_value = xor_mlp_output_layer

print('XOR 신경망 결과:',xor_classifier_value)

