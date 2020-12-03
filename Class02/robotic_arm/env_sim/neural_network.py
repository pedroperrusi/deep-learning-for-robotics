import torch
import torch.nn as nn

'''Feed Forward Neural Network For Regression'''

############################################# FOR STUDENTS #####################################
class FullyConnectedNetwork(nn.Module):
    def __init__(self, input_dim, num_hidden_neurons, dropout_rte):
        super(FullyConnectedNetwork, self).__init__()
        ###### Define Linear Layer 0 ######

        self.h_0 = nn.Linear(input_dim, num_hidden_neurons[0])
        self.h_1 = nn.Linear(num_hidden_neurons[0], num_hidden_neurons[1])
        self.h_2 = nn.Linear(num_hidden_neurons[1], num_hidden_neurons[2])
        self.h_3 = nn.Linear(num_hidden_neurons[2], num_hidden_neurons[3])
        self.h_4 = nn.Linear(num_hidden_neurons[3], num_hidden_neurons[4])

        self.drop = nn.Dropout(dropout_rte)

    def forward(self, x):
        out_0 = self.drop(torch.tanh(self.h_0(x)))
        out_1 = self.drop(torch.tanh(self.h_1(out_0)))
        out_2 = self.drop(torch.tanh(self.h_2(out_1)))
        out_3 = self.drop(torch.tanh(self.h_3(out_2)))
        out_4 = self.drop(torch.tanh(self.h_4(out_3)))

        ###### Using The Defined Layers and F.tanh As The Nonlinear Function Between Layers Define Forward Function ######

        return out_4
#################################################################################################
