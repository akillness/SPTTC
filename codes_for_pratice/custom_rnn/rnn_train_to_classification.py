import pandas as pd

import torch
import torch.nn as nn

from tqdm import tqdm
from torch.optim import Adam, SGD

from torch.utils.tensorboard import SummaryWriter 


# Setting device for using gpu operator
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")

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
        return torch.zeros(1, self.hidden_size,device=device) # batch size 1 로 고정
    
def nameToTensor(name, unique_chars):
    n_letters = len(unique_chars)
    tensor = torch.zeros(len(name), n_letters)
    for char_idx, char in enumerate(name):
        letter_idx = unique_chars.find(char)
        assert letter_idx != -1, f"char is {name}, {char}"
        tensor[char_idx][letter_idx] = 1
    return tensor

def main():
    # import os

    # print(os.getcwd())
    # df = pd.read_csv('./practice/myRNN/name_gender_filtered.csv')
    df = pd.read_csv('./name_gender_filtered.csv')
    unique_chars = set()

    for name in df['Name']:
        unique_chars.update(name)
        
    unique_chars = sorted(list(unique_chars))
    unique_chars = ''.join(unique_chars)
    print(unique_chars) # 일종의 tokenizer를 위한 token 타입을 찾는 과정

    n_letters = len(unique_chars)
    
    # tokenizer for labeled data 
    # Classification problem 을 위해 prediction dictionary 설정
    gen2num = {'F':0, 'M':1}
    num2gen = {0:'F', 1:'M'}

    n_hidden = 32
    rnn_model = MyRNN(n_letters, n_hidden, 2)
    rnn_model = rnn_model.to(device)

    loss_fn = nn.CrossEntropyLoss() # CrossEntropyLoss term 설정 
    # Cross Entropy 는 Multinomial 에 대한 예측을 위해, logit (sigmoid) score를 probability 로 softmax 한 값을 이용해
    # Cumulative Sum about Logit element wise -log(Ground Truth) 을 이용해 Entropy 가 0 인 값을 추정하는 방법
    optimizer = Adam(rnn_model.parameters(), lr=0.0001) # Optimizer 설정
    # Loss function 의 값을 미세 조정하기 위한 Parameters (Weights) 편미분할 Optimizer 설정


    ''' Summary writer for tensorboard '''
    writer = SummaryWriter('./practice/myRNN/logs/')

    ''' Section of traning '''

    rnn_model.train() # torch module 의 state 설정
    for epoch_idx in tqdm(range(200)): # training 하기 위한 epoch 수 설정
        shuffled_df = df.sample(frac=1).reset_index(drop=True) # dataframe row를 shuffle 하고 index reset

        total_loss = 0.
        correct_predictions = 0
        total_predictions = 0

        for index, row in shuffled_df.iterrows(): # data 의 row 수 만큼 iteration 설정 ( 현재, 전체 데이터 수를 하나의 batch 로 사용 - deteministic )
            input_tensor = nameToTensor(row['Name'],unique_chars).to(device)
            target_tensor = torch.tensor([gen2num[row['Gender']]], dtype=torch.long).to(device)

            hidden = rnn_model.get_hidden() # 매 epoch 마다 hidden state 초기화

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

        # Write scalar datas
        writer.add_scalar("Train/Loss", average_loss, epoch_idx) 
        writer.add_scalar("Train/Accuracy", accuracy, epoch_idx) 
        
        ''' Save model by using static_dic() '''
        if epoch_idx % 20 == 0:
            # Save
            torch.save(rnn_model.state_dict(), './practice/myRNN/myRNN.pt')
            # Load
            # loaded_model = MyRNN()
            # loaded_model.load_state_dict(torch.load('myRNN.pt'))

    writer.flush()
    # Call close() to save data of writer.
    writer.close()
    # Command line to visualize at localhost : tensorboard --logdir ./logs

    


if __name__ == "__main__":
    main()
