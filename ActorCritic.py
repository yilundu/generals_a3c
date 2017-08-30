import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class ActorCritic(nn.Module):

    def __init__(self, on_gpu = False):
        # Current architecture for policy is 3 5x5 convolutions
        # followed by 2 LSTM layers followed by 2 5x5 convolutions
        # and a final 1x1 convolution
        # This architecture if fully convolutional with no max pooling
        super(ActorCritic, self).__init__()
        self.lstm_layer = 3
        self.hidden_dim = 200
        self.on_gpu = on_gpu

        self.conv1 = nn.Conv2d(11, self.hidden_dim, 5, padding=2)

        self.conv2 = nn.Conv2d(self.hidden_dim, self.hidden_dim, 5, padding=2)
        self.conv3 = nn.Conv2d(self.hidden_dim, self.hidden_dim, 5, padding=2)

        self.pre_lstm_bn = nn.BatchNorm2d(self.hidden_dim)

        self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim, self.lstm_layer)

        self.lstm_batch_norm = nn.BatchNorm2d(self.hidden_dim)

        self.conv4 = nn.Conv2d(self.hidden_dim, self.hidden_dim, 5, padding=2)
        self.conv5 = nn.Conv2d(self.hidden_dim, self.hidden_dim, 5, padding=2)

        self.move_conv = nn.Conv2d(self.hidden_dim, 8, 1)
        self.value_conv = nn.Conv2d(self.hidden_dim, self.hidden_dim, 1)
        self.value_linear = nn.Linear(self.hidden_dim, 1)

    def init_hidden(self, height, width):
        self.height = height
        self.width = width
        self.batch = height * width

        self.cell_state = Variable(torch.zeros(self.lstm_layer, self.batch, self.hidden_dim))
        self.hidden_state = Variable(torch.zeros(self.lstm_layer, self.batch, self.hidden_dim))

        if self.on_gpu:
            self.cell_state = self.cell_state.cuda()
            self.hidden_state = self.hidden_state.cuda()

    def reset_hidden(self):
        # Zero gradients on hidden states
        self.cell_state = Variable(self.cell_state.data)
        self.hidden_state = Variable(self.hidden_state.data)

    def forward(self, input):
        x = F.elu(self.conv1(input))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))

        # Next flatten the output to be batched into LSTM layers
        # The shape of x is batch_size, channels, height, width
        x = self.pre_lstm_bn(x)

        x = torch.transpose(x, 1, 3)
        x = torch.transpose(x, 1, 2)
        x = x.contiguous()

        x = x.view(x.size(0), self.batch, self.hidden_dim)
        x, hidden = self.lstm(x, (self.hidden_state, self.cell_state))
        self.hidden_state, self.cell_state = hidden

        x = torch.transpose(x, 2, 1)
        x = x.contiguous()
        x = x.view(x.size(0), self.hidden_dim, self.height, self.width)

        x = self.lstm_batch_norm(x)

        x = F.elu(self.conv4(x))
        x = F.elu(self.conv5(x))

        logit = self.move_conv(x)
        logit = logit.view(logit.size(0), -1)

        x = self.value_conv(x)
        x = x.view(x.size(0), self.hidden_dim, self.batch)
        x = F.max_pool1d(x, self.batch)
        x = x.squeeze()
        val = self.value_linear(x)

        return val, logit
