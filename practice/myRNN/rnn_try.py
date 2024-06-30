import pandas as pd

import torch
import torch.nn as nn
from torch.optim import Adam, SGD

df = pd.read_csv('./name_gender_filtered.csv')
unique_chars = set()

for name in df['Name']:
    unique_chars.update(name)
    
unique_chars = sorted(list(unique_chars))
unique_chars = ''.join(unique_chars)
print(unique_chars)


#Try other arch
class MyRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size) # nn.Linear 를 통해 mat 차원변경
        self.h2o = nn.Linear(hidden_size, output_size)


    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1) # input vector 와 hidden(t-1) state vector 를 hidden(t) state 로 연산하기 위한 concatrate
        hidden = torch.tanh(self.i2h(combined)) # activation function 으로 tanh -1 < h(x) < 1
        output = self.h2o(hidden) # hidden(t) state 로부터 output vector 
        # output vector는 예제에서 F, M 로 사용 ( Binary classification problem )
        return output, hidden

    def get_hidden(self): # hidden state 는 초기 0 tensor로 구성
        return torch.zeros(1, self.hidden_size) # batch size 1 로 고정
    


n_hidden = 32
rnn_model = MyRNN(n_letters, n_hidden, 2)


loss_fn = nn.CrossEntropyLoss() # CrossEntropyLoss term 설정
optimizer = Adam(rnn_model.parameters(), lr=0.0001) # Optimizer 설정


rnn_model.train()
for epoch_idx in range(200):
    shuffled_df = df.sample(frac=1).reset_index(drop=True)

    total_loss = 0.
    correct_predictions = 0
    total_predictions = 0

    for index, row in shuffled_df.iterrows():
        input_tensor = nameToTensor(row['Name'])
        target_tensor = torch.tensor([gen2num[row['Gender']]], dtype=torch.long)

        hidden = rnn_model.get_hidden()

        rnn_model.zero_grad()

        for char_idx in range(input_tensor.size()[0]):
            char_tensor = input_tensor[char_idx]
            output, hidden = rnn_model(char_tensor[None,:], hidden)

        loss = loss_fn(output, target_tensor)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predicted_index = torch.argmax(output, 1)
        correct_predictions += (predicted_index == target_tensor).sum().item()
        total_predictions += 1

    average_loss = total_loss / total_predictions
    accuracy = 100 * correct_predictions / total_predictions
    print(f'Epoch: {epoch_idx}, Loss: {average_loss:.4f}, Accuracy: {accuracy:.2f}%')
