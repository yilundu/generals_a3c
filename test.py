from collections import deque
import time

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from generalsenv import GeneralEnvironment
from ActorCritic import ActorCritic


def test(rank, args, shared_model):
    torch.manual_seed(args.seed + rank)

    env = GeneralEnvironment('2_epoch.mdl')

    model = ActorCritic()
    model.eval()

    state = env.reset()
    model.init_hidden(env.map_height, env.map_width)
    state = torch.Tensor(state)
    reward_sum = 0
    done = True

    start_time = time.time()

    # a quick hack to prevent the agent from stucking
    actions = deque(maxlen=100)
    episode_length = 0
    while True:
        episode_length += 1
        # Sync with the shared model
        if done:
            model.load_state_dict(shared_model.state_dict())

        value, logit = model(Variable(
            state.unsqueeze(0), volatile=True))
        prob = F.softmax(logit)

        # Set the probability of all items that not owned by user to 
        # 0
        army_map = state[0, ...]
        label_map = (army_map > 0)
        label_map = label_map.view(1, env.map_height, env.map_width)
        label_map = label_map.expand(8, env.map_height, env.map_width)
        label_map = label_map.contiguous()
        label_map = label_map.view(-1)
        prob = prob * Variable(label_map.float())

        action = prob.max(1, keepdim=True)[1].data.numpy()

        state, reward, done, _ = env.step(action[0, 0])
        done = done or episode_length >= args.max_episode_length
        reward_sum += reward

        # a quick hack to prevent the agent from stucking
        actions.append(action[0, 0])
        if actions.count(actions[0]) == actions.maxlen:
            done = True

        if done:
            print("Time {}, episode reward {}, episode length {}".format(
                time.strftime("%Hh %Mm %Ss",
                              time.gmtime(time.time() - start_time)),
                reward_sum, episode_length))
            reward_sum = 0
            episode_length = 0
            actions.clear()
            state = env.reset()
            model.init_hidden(env.map_height, env.map_width)
            time.sleep(60)

        state = torch.Tensor(state)
