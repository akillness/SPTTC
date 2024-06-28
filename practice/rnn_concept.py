import torch
import torch.nn as nn

class MyRNN(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size+hidden_size,hidden_size)
        self.h2o = nn.Linear(hidden_size,output_size)

    def forward(self,input, hidden):
        combine = torch.cat((input,hidden),1)
        hidden = torch.tanh(self.i2h(combine))
        output = self.h2o(hidden)
        return output, hidden

    # Return to zero tensor 
    # It mean state tensor inner NN.
    def get_hidden_size(self):
        # Because it follow paradigm of torch programming
        # As possible, A Neural Network class have weight parameters only
        return torch.zeros(1,self.hidden_size) # Fixed batch size -> 1



# Classificate positive or negative of input sequence
rnn_model = MyRNN(input_size=4,hidden_size=4,output_size=2)
hidden  = rnn_model.get_hidden_size()

# # input sequence : The food is good 
# _,hidden = rnn_model(input_tensor0,hidden) # 0: The
# _,hidden = rnn_model(input_tensor1,hidden) # 1: food
# _,hidden = rnn_model(input_tensor2,hidden) # 2: is
# output_tensor,_ = rnn_model(input_tensor3,hidden) # 3: good


