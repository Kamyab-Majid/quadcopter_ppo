import csv
import gym
import envs
ENV_ID = "CustomEnv-v0"
my_env = gym.make(ENV_ID)
done = False
my_env.current_states = my_env.reset()
while not done:
    # current_action = my_contr.Controller_model(my_env.current_states, my_env.dt * my_env.counter, action=sl_action)   
    my_env.current_states, b, done, _ = my_env.step((0,0,0,0))
# if my_env.best_reward > current_best_rew:
#     current_best_rew = my_env.best_reward
    # with open("reward_step.csv", "a") as f:
    #     writer = csv.writer(f)
    #     writer.writerow(sl_action)
