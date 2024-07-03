import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F 

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

# Representation from tonkens to one hot encoding
def nameToOneHot(name,tokens):
    # Add start and end tokens to the name
    tokenized_name = ['<S>'] + list(name) + ['<E>']
    int_tensor = torch.tensor([tokens[char] for char in tokenized_name]) # token의 index 값 설정
    one_hot_encoded = F.one_hot(int_tensor,num_classes=len(tokens)).float() # indices로부터 생성하기 위한 vector 설정
    return one_hot_encoded

@torch.no_grad()
def generateName(model, input_tokens, output_tokens):
    model.eval()
    start_token_idx = torch.tensor(input_tokens['<S>'])
    one_hot_encode = F.one_hot(start_token_idx,num_classes=len(input_tokens)).float()
    hidden = model.get_hidden()
    char_list = [] #
    for i in range(20):
        out_score, hidden = model(one_hot_encode[None,:].to(device),hidden) # [None,:] 하면 2차원 배열로 표현
        score_probability = F.softmax(out_score[0],dim=-1) # logit score를 probability score로 softmax 변환
        out_idx = torch.multinomial(score_probability,1).item() # multinomial 은 input row의 distibution을 따라 sample 결과 반환
        if out_idx == input_tokens['<E>']:
            break
        char_list.append(output_tokens[out_idx])
        one_hot_encode = F.one_hot(torch.tensor(out_idx),num_classes=len(input_tokens)).float() # RNN의 다음 input encode 값으로 사용하기 위해 update
    print(''.join(char_list))

def main():
    # import os

    # print(os.getcwd())
    df = pd.read_csv('./practice/myRNN/name_gender_filtered.csv')
    unique_chars = set()

    for name in df['Name']:
        unique_chars.update(name)
        
    unique_chars = sorted(list(unique_chars))
    unique_chars = ''.join(unique_chars)
    sorted_chars = sorted(set(unique_chars))

    # predefine tokens to generate
    stoi = {s:i for i,s in enumerate(sorted_chars)}
    stoi['<S>'] = len(stoi)
    stoi['<E>'] = len(stoi)
    itos = {i:s for s,i in stoi.items()}
    print(itos) 

    n_letters = len(stoi)
    n_hidden = 1024
    rnn_model = MyRNN(n_letters, n_hidden, n_letters)
    rnn_model = rnn_model.to(device)

    loss_fn = nn.CrossEntropyLoss() # CrossEntropyLoss term 설정 
    # Cross Entropy 는 Multinomial 에 대한 예측을 위해, logit (sigmoid) score를 probability 로 softmax 한 값을 이용해
    # Cumulative Sum about Logit element wise -log(Ground Truth) 을 이용해 Entropy 가 0 인 값을 추정하는 방법
    optimizer = Adam(rnn_model.parameters(), lr=0.0001) # Optimizer 설정
    # Loss function 의 값을 미세 조정하기 위한 Parameters (Weights) 편미분할 Optimizer 설정


    ''' Summary writer for tensorboard '''
    writer = SummaryWriter('./practice/myRNN/logs_generate/')

    ''' Section of traning '''
    for epoch_idx in tqdm(range(100)): # training 하기 위한 epoch 수 설정
        shuffled_df = df.sample(frac=1).reset_index(drop=True) # dataframe row를 shuffle 하고 index reset

        crnt_loss = 0.        
        rnn_model.train() # torch module 의 state 설정
        for _, row in shuffled_df.iterrows(): # data 의 row 수 만큼 iteration 설정 ( 현재, 전체 데이터 수를 하나의 batch 로 사용 - deteministic )
            name_one_hot = nameToOneHot(row['Name'],stoi).to(device)
            hidden = rnn_model.get_hidden() # 매 epoch 마다 hidden state 초기화
            rnn_model.zero_grad()

            losses= []
            for char_idx in range(len(name_one_hot)-1):
                input_tensor = name_one_hot[char_idx].to(device)
                target_char = name_one_hot[char_idx+1].to(device)
                target_class = torch.argmax(target_char,-1) # target char index 출력
                out_score,hidden = rnn_model(input_tensor[None,:].to(device),hidden)
                losses.append(loss_fn(out_score[0],target_class)) # score 와 target token vector 값을 이용해 loss 설정


            loss = sum(losses) # input d
            loss.backward()
            optimizer.step()

            crnt_loss += loss.item()

        generateName(rnn_model,stoi,itos)
        average_loss = crnt_loss / len(df)

        print(f'Iter idx {epoch_idx}, Loss: {average_loss:.4f}')

        # Write scalar datas
        writer.add_scalar("Train/Loss", average_loss, epoch_idx) 
        
        ''' Save model by using static_dic() '''
        if (epoch_idx+1) % 20 == 0:
            # Save
            torch.save(rnn_model.state_dict(), './practice/myRNN/myRNN_generate.pt')
            # Load
            # loaded_model = MyRNN()
            # loaded_model.load_state_dict(torch.load('myRNN.pt'))

    writer.flush()
    # Call close() to save data of writer.
    writer.close()
    # Command line to visualize at localhost : tensorboard --logdir ./logs

    


if __name__ == "__main__":
    main()
