import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class MultiGPUTrain(nn.Module):

    def __init__(self, module, n_gpu):
        super(MultiGPUTrain, self).__init__()
        self.device_ids = range(n_gpu)
        if n_gpu > 1:
            self.replicas = nn.parallel.replicate(module, self.device_ids)
        else:
            self.replicas = [module]

        self.n_gpu = n_gpu

    def init_hidden(self, l_dims):
        # l_dims should be a list of tuples of form (h, w)
        # Perhaps make this parallel?
        for i, dim in enumerate(l_dims):
            self.replicas[i].init_hidden(dim[0], dim[1])

    def forward(self, inputs):
        # inputs should be variables
        if self.n_gpu > 1:
            inputs = list(map(lambda x: (x,), inputs))
            outputs = nn.parallel.parallel_apply(self.replicas, inputs, devices=self.device_ids)
        else:
            outputs = [self.replicas[0].forward(inputs[0])]
        return outputs

    def zero_grad(self):
        for replica in self.replicas:
            replica.zero_grad()


class CNNLSTMPolicy(nn.Module):

    def __init__(self, on_gpu = False):
        # Current architecture for policy is 3 5x5 convolutions
        # followed by 2 LSTM layers followed by 2 5x5 convolutions
        # and a final 1x1 convolution
        # This architecture if fully convolutional with no max pooling
        super(CNNLSTMPolicy, self).__init__()
        self.lstm_layer = 3
        self.hidden_dim = 100
        self.on_gpu = on_gpu

        self.conv1 = nn.Conv2d(11, self.hidden_dim, 5, padding=2)

        self.conv2 = nn.Conv2d(self.hidden_dim, self.hidden_dim, 5, padding=2)
        self.conv3 = nn.Conv2d(self.hidden_dim, self.hidden_dim, 5, padding=2)

        self.pre_lstm_bn = nn.BatchNorm2d(self.hidden_dim)

        self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim, self.lstm_layer)

        self.lstm_batch_norm = nn.BatchNorm2d(self.hidden_dim)

        self.conv4 = nn.Conv2d(self.hidden_dim, self.hidden_dim, 5, padding=2)
        self.conv5 = nn.Conv2d(self.hidden_dim, self.hidden_dim, 5, padding=2)

        self.begin_conv = nn.Conv2d(self.hidden_dim, 1, 1)
        self.end_conv = nn.Conv2d(self.hidden_dim, 2, 1)

    def init_hidden(self, height, width):
        self.height = height
        self.width = width
        self.batch = height * width

        self.cell_state = Variable(torch.zeros(self.lstm_layer, self.batch, self.hidden_dim))
        self.hidden_state = Variable(torch.zeros(self.lstm_layer, self.batch, self.hidden_dim))

        if self.on_gpu:
            self.cell_state = self.cell_state.cuda()
            self.hidden_state = self.hidden_state.cuda()

    def forward(self, input):
        # TODO perhaps add batch normalization or layer normalization

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

        o_begin = self.begin_conv(x)
        o_end = self.end_conv(x)

        o_begin = o_begin.view(o_begin.size(0), -1)
        o_end = o_end.view(o_end.size(0), -1)

        o_begin = F.log_softmax(o_begin)
        o_end = F.log_softmax(o_end)

        return o_begin, o_end
