import gym
import numpy as np
import torch
import torch.optim as optim
from utils_main import make_env, save_files
from neural_network import ActorCritic
from ppo_method import ppo
from common.multiprocessing_env import SubprocVecEnv
from itertools import count

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


num_envs = 2
env_name = "CustomEnv-v0"


envs = [make_env(env_name) for i in range(num_envs)]
envs = SubprocVecEnv(envs)


num_inputs = envs.observation_space.shape[0]
num_outputs = envs.action_space.shape[0]

# Hyper params:
hidden_size = 400
lr = 3e-6
num_steps = 20
mini_batch_size = 5
ppo_epochs = 4
threshold_reward = -0.01

model = ActorCritic(num_inputs, num_outputs, hidden_size).to(device)
env = gym.make(env_name)

my_ppo = ppo(model, env)
optimizer = optim.Adam(model.parameters(), lr=lr)
max_frames = 1_500_0000
frame_idx = 0
test_rewards = []
save_iteration = 1000
model_save_iteration = 1000
state = envs.reset()
early_stop = False


def trch_ft_device(input, device):
    output = torch.FloatTensor(input).to(device)
    return output


saver_model = save_files()

while frame_idx < max_frames and not early_stop:

    log_probs = []
    values = []
    states = []
    actions = []
    rewards = []
    masks = []
    entropy = 0

    for _ in range(num_steps):
        state = trch_ft_device(state, device)
        dist, value = model(state)

        action = dist.sample()
        next_state, reward, done, _ = envs.step(action.cpu().numpy())

        log_prob = dist.log_prob(action)
        entropy += dist.entropy().mean()
        # appending
        log_probs.append(log_prob)
        values.append(value)
        rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device))
        masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(device))
        states.append(state)
        actions.append(action)
        # next iteration init.
        state = next_state
        frame_idx += 1

        if frame_idx % save_iteration == 0:
            test_reward = np.mean([my_ppo.test_env() for _ in range(num_envs)])
            test_rewards.append(test_reward)
            # plot(frame_idx, test_rewards)
            if test_reward > threshold_reward:
                early_stop = True
            if frame_idx % model_save_iteration == 0:
                saver_model.model_save(model)
    next_state = trch_ft_device(next_state, device)
    _, next_value = model(next_state)
    returns = my_ppo.compute_gae(next_value, rewards, masks, values)

    returns = torch.cat(returns).detach()
    log_probs = torch.cat(log_probs).detach()
    values = torch.cat(values).detach()
    states = torch.cat(states)
    actions = torch.cat(actions)
    advantage = returns - values

    my_ppo.ppo_update(ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantage, optimizer)

max_expert_num = 50000
num_steps = 0
expert_traj = []
# building an episode based on the current model.
for i_episode in count():
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        dist, _ = model(state)
        action = dist.sample().cpu().numpy()[0]
        next_state, reward, done, _ = env.step(action)
        state = next_state
        total_reward += reward
        expert_traj.append(np.hstack([state, action]))
        num_steps += 1

    print("episode:", i_episode, "reward:", total_reward)

    if num_steps >= max_expert_num:
        break

expert_traj = np.stack(expert_traj)
print()
print(expert_traj.shape)
print()
np.save("expert_traj.npy", expert_traj)
