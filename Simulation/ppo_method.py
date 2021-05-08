
import numpy as np
import torch


class ppo:
    def __init__(self, model, env):
        self.model = model
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.env = env

    def test_env(self, vis=False):
        state = self.env.reset()
        if vis:
            self.env.render()
        done = False
        total_reward = 0
        while not done:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            dist, _ = self.model(state)
            next_state, reward, done, _ = \
                self.env.step(dist.sample().cpu().numpy()[0])
            state = next_state
            if vis:
                self.env.render()
            total_reward += reward
        return total_reward

    @staticmethod
    def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
        values = values + [next_value]
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * values[step + 1] * masks[step] - \
                values[step]  # value: model value,
            gae = delta + gamma * tau * masks[step] * gae
            returns.insert(0, gae + values[step])
        return returns

    @staticmethod
    def ppo_iter(mini_batch_size, states, actions, log_probs,
                 returns, advantage):
        batch_size = states.size(0)  # number of states in batch
        for _ in range(batch_size // mini_batch_size):
            rand_ids = np.random.randint(0, batch_size, mini_batch_size)
            # random integer in a range
            yield states[rand_ids, :], actions[rand_ids, :],\
                log_probs[rand_ids, :], returns[rand_ids, :], \
                advantage[rand_ids, :]

    def ppo_update(
        self, ppo_epochs, mini_batch_size, states, actions, log_probs, returns,
            advantages, optimizer, clip_param=0.2):
        for _ in range(ppo_epochs):
            for state, action, old_log_probs, return_, advantage in\
                ppo.ppo_iter(mini_batch_size, states, actions, log_probs,
                             returns, advantages):
                dist, value = self.model(state)
                entropy = dist.entropy().mean()
                new_log_probs = dist.log_prob(action)

                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param)\
                    * advantage

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = (return_ - value).pow(2).mean()

                loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
