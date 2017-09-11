import CNNLSTMPolicy
import torch.nn as nn
import torch.optim as opt
from torch.autograd import Variable
import torch
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--on-gpu", type=bool, default=True)
parser.add_argument("--num-epochs", type=int, default=2,
                    help="number of epochs to train network")
parser.add_argument("--data", type=str, default="",
                    help="directory containing data directory")
parser.add_argument("--lr", type=float, default=1e-5,
                    help="LR")
args = parser.parse_args()

# Currently only 1 GPU works
num_gpu = 1
model = CNNLSTMPolicy.CNNLSTMPolicy(on_gpu=args.on_gpu)
multi_gpu_trainer = CNNLSTMPolicy.MultiGPUTrain(model, num_gpu)

loss_function = nn.NLLLoss()
optimizer = opt.Adam(model.parameters(), lr=args.lr)

num_epochs = args.num_epochs

print("Loading all data...")
data_x = np.load(args.data + "data_x.npz")['arr_0']
data_y = np.load(args.data + "data_y.npz")['arr_0']
data_z = np.load(args.data + "data_z.npz")['arr_0']

counter = 0
inputs = range(num_gpu)
label1 = range(num_gpu)
label2 = range(num_gpu)
l_dim = range(num_gpu)


def backward_output(outputs, ll1, ll2):
    lloss_1, lloss_2 = [], []
    for output, l_1, l_2 in zip(outputs, ll1, ll2):
        o_1, o_2 = output
        loss_1, loss_2 = loss_function(o_1, l_1), loss_function(o_2, l_2)
        loss_1.backward(retain_graph=True)
        loss_2.backward(retain_graph=True)

        lloss_1.append(loss_1.data[0])
        lloss_2.append(loss_2.data[0])

    return sum(lloss_1) / len(lloss_1), sum(lloss_2) / len(lloss_2)


loss = []
print("Training network...")
for _ in range(num_epochs):
    for j in xrange(len(data_x)):
        for k in xrange(len(data_x[j])):
            # Shapes of replay is given by batch_size * num_channels * h * w
            input_player = data_x[j][k]
            if len(input_player.shape) < 4:
                continue

            counter += 1
            index = counter % num_gpu

            l_dim[index] = input_player.shape[2], input_player.shape[3]
            inputs[index] = Variable(
                torch.Tensor(
                    input_player.astype(
                        np.float64)))
            label1[index] = Variable(torch.LongTensor(data_y[j][k]))
            label2[index] = Variable(torch.LongTensor(data_z[j][k]))

            if args.on_gpu:
                inputs[index].cuda(index)
                label1[index].cuda(index)
                label2[index].cuda(index)

            if counter % num_gpu == 0:
                multi_gpu_trainer.init_hidden(l_dim)
                outputs = multi_gpu_trainer.forward(inputs)
                loss_1, loss_2 = backward_output(outputs, label1, label2)
                nn.utils.clip_grad_norm(model.parameters(), 1)
                optimizer.step()
                optimizer.zero_grad()

                loss.append(loss_1 + loss_2)

np.save('loss', loss)
torch.save(model.cpu().state_dict(), 'policy.mdl')
