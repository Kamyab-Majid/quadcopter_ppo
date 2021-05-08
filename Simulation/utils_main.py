import gym
import envs
from IPython.display import clear_output
import matplotlib.pyplot as plt
import os
from datetime import datetime
import torch
import numpy as np
import csv


def make_env(env_name):
    def _thunk():
        env = gym.make(env_name)
        return env

    return _thunk


def plot(frame_idx, rewards):
    clear_output(True)
    plt.figure(figsize=(20, 5))
    plt.subplot(131)
    plt.title("frame %s. reward: %s" % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    plt.show()


class save_files:
    def __init__(self):
        self.date = datetime.now().strftime("%Y_%m_%d_%I_%M_%S_%p")
        self.current_dir = os.getcwd()
        self.path_step_reward = "results/reward_step"
        self.path_best_reward = f"results/bestreward{self.date}"
        self.path_model = f"results/model{self.date}"
        self._save_init(self.path_step_reward)
        self._save_init(self.path_best_reward)
        self._save_init(self.path_model)
        self.index = 1
        fields = ['counter', 'step', 'reward']
        with open(f"{self.path_step_reward}/reward_step{self.date}.csv", "a") as f:
            writer = csv.writer(f)
            writer.writerow(fields)

    def _save_init(self, directory):
        self.path = os.path.join(self.current_dir, directory)
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def best_reward_save(self, all_t, all_actions, all_obs, all_rewards, control_rewards, header):
        date = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
        np.savetxt(
            f"{self.path_best_reward}/best_rewards{date}.csv",
            np.c_[all_t, all_actions, all_obs, all_rewards, control_rewards],
            delimiter=",",
            header=header,
        )

    def reward_step_save(self, best_rew, longest_step, curr_tot_rew, curr_step):
        print(f"best reward: {best_rew}, longest step: {longest_step}, reward: {curr_tot_rew}, step: {curr_step} ")
        fields = [self.index, curr_step, float(curr_tot_rew)]
        with open(f"{self.path_step_reward}/reward_step{self.date}.csv", "a") as f:
            writer = csv.writer(f)
            writer.writerow(fields)
        self.index += 1

    def model_save(self, model):
        date = datetime.now().strftime("%Y_%m_%d_%I_%M_%S_%p")
        torch.save(model.state_dict(), f"{self.path_model}/model{date}.pt")
