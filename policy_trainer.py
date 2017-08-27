import CNNLSTMPolicy
from termcolor import cprint
import torch.nn as nn
import torch.optim as opt
import numpy as np
from torch.autograd import Variable
import torch


# cprint("Testing save model...", 'green')
# model = CNNLSTMPolicy.CNNLSTMPolicy().cpu()
# torch.save(model.cpu().state_dict(), '/tmp/test.mdl')
# cprint("Testing load model...", 'green')
# model = CNNLSTMPolicy.CNNLSTMPolicy().cpu()
# model.load_state_dict(torch.load('2_epoch.mdl'))
# model.init_hidden(20, 20)
# model.eval()
# model.forward(Variable(torch.randn(1, 11, 20, 20)))

# Currently only 1 GPU works
num_gpu = 1
model = CNNLSTMPolicy.CNNLSTMPolicy(on_gpu=True).cuda(0)
multi_gpu_trainer = CNNLSTMPolicy.MultiGPUTrain(model, num_gpu)

loss_function = nn.NLLLoss()
optimizer = opt.Adam(model.parameters(), lr=1e-5)

num_epochs = 2

cprint("Loading all data...", "red")
data_x = np.load("/mnt/homedir/yilundu/data_x.npz")['arr_0']
data_y = np.load("/mnt/homedir/yilundu/data_y.npz")['arr_0']
data_z = np.load("/mnt/homedir/yilundu/data_z.npz")['arr_0']

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

    return sum(lloss_1)/len(lloss_1), sum(lloss_2)/len(lloss_2)


loss = []
cprint("Training network...", "green")
for _ in range(num_epochs):
    for j in xrange(len(data_x)):
        for k in xrange(len(data_x[j])):
            # Shapes of replay is given by batch_size * num_channels * h * w
            input_player = data_x[j][k]
            if len(input_player.shape) < 4:
                continue

            counter += 1
            index = counter % num_gpu

            inputs[index] = Variable(torch.Tensor(input_player.astype(np.float64)).cuda(index))
            l_dim[index] = input_player.shape[2], input_player.shape[3]
            label1[index] = Variable(torch.LongTensor(data_y[j][k]).cuda(index))
            label2[index] = Variable(torch.LongTensor(data_z[j][k]).cuda(index))

            if counter %  num_gpu == 0:
                multi_gpu_trainer.init_hidden(l_dim)
                outputs = multi_gpu_trainer.forward(inputs)
                loss_1, loss_2 = backward_output(outputs, label1, label2)
                nn.utils.clip_grad_norm(model.parameters(), 1)
                optimizer.step()
                optimizer.zero_grad()

                cprint("Finished processing batch {}".format(counter))

                cprint("Loss for the first layer is {}".format(loss_1))
                cprint("Loss for the second layer is {}".format(loss_2))
                loss.append(loss_1 + loss_2)

loss = np.array(loss)
np.save('loss_2e-6_2', loss)
torch.save(model.cpu().state_dict(), '2_epoch.mdl')

