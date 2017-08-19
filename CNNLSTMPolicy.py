import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class CNNLSTMPolicy(nn.Module):

    def __init__(self):
        # Current architecture for policy is 3 5x5 convolutions
        # followed by 2 LSTM layers followed by 2 5x5 convolutions
        # and a final 1x1 convolution
        # This architecture if fully convolutional with no max pooling
        super(CNNLSTMPolicy, self).__init__()
        self.conv1 = nn.Conv2d(10, 50, 5, padding=2)
        self.conv2 = nn.Conv2d(50, 50, 5, padding=2)
        self.conv3 = nn.Conv2d(50, 50, 5, padding=2)

        self.lstm = nn.LSTM(50, 50, 2)

        self.conv4 = nn.Conv2d(50, 50, 5, padding=2)
        self.conv5 = nn.Conv2d(50, 50, 5, padding=2)

        self.begin_conv = nn.Conv2d(50, 1, 1)
        self.end_conv = nn.Conv2d(50, 2, 1)

        self.cell_state = Variable(torch.zeros(2, 400, 50))
        self.hidden_state = Variable(torch.zeros(2, 400, 50))

    def forward(self, input, reset_state=False):
        if reset_state:
            self.cell_state = Variable(torch.zeros(2, 400, 50))
            self.hidden_state = Variable(torch.zeros(2, 400, 50))

        # TODO perhaps add batch normalization or layer normalization

        x = F.relu(self.conv1(input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Next flatten the output to be batched into LSTM layers
        # The shape of x is batch_size, channels, height, width

        x = torch.transpose(x, 1, 3)
        x = x.contiguous()

        x = x.view(x.size(0), 400, 50)
        x, hidden = self.lstm(x, (self.hidden_state, self.cell_state))
        self.hidden_state, self.cell_state = hidden

        x = torch.transpose(x, 2, 1)
        x = x.contiguous()
        x = x.view(x.size(0), 50, 20, 20)

        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))

        o_begin = self.begin_conv(x)
        o_end = self.end_conv(x)

        o_begin = o_begin.view(o_begin.size(0), -1)
        o_end = o_end.view(o_end.size(0), -1)

        o_begin = F.log_softmax(o_begin)
        o_end = F.log_softmax(o_end)

        return o_begin, o_end
