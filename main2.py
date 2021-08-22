import gym
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tpd
from tf_agents.distributions.gumbel_softmax import GumbelSoftmax
import continuous_policy_network as cpn
import sac as sac
import numpy as np
import replay_buffer as replay

def makeT(i):
    t = tf.constant([5.0 + i, 7.0 - i, 100 + i])
    print(t)
    return t

def convert_gym_action_space(gym_action_space):
    num_actions = gym_action_space.shape[0]
    lows = gym_action_space.low
    highs = gym_action_space.high
    dtype = gym_action_space.dtype
    bounds = []
    for i in range(0, num_actions):
        bounds[i] = (lows[i], highs[i], dtype)

    return bounds

if __name__ == '__main__':

    env = gym.make("Pendulum-v0")
    action_space = env.action_space

    policy = cpn.ContinuousPolicyNetwork()
    q1 = cpn.QNetwork()
    q2 = cpn.QNetwork()

    s = sac.SAC(action_space=action_space, policy=policy, Qs=[q1, q2],
                reward_scale=1.0, target_update_interval=1)

    buf = replay.ReplayBuffer(env.observation_space.shape, state_type=np.float32,
                              action_shape=env.action_space.shape, action_type=np.float32,
                              reward_type=np.float32)

    max_steps = 199
    batch_size = 128
    training_count = 0

    action_low = action_space.low
    action_high = action_space.high
    action_high_minus_action_low = action_space.high - action_space.low

    def scale_actions(actions):
        scaled_action = action_low + (actions + 1) * 0.5 * action_high_minus_action_low
        return scaled_action

    last_episode = -1
    for episode in range(100000):
        state = env.reset()
        episode_reward = 0
        for step in range(max_steps):
            action = s.get_action(state)
            if tf.math.is_nan(action):
                print("stop")
            next_state, reward, done, _ = env.step(scale_actions(action))
            buf.append(state, action, reward, next_state, done)
            episode_reward += reward
            if buf.len > 3000:
                batch = buf.sample(batch_size)
                Qs_values, Qs_losses, policy_losses, alpha_losses, alpha, log_pis = s.train(batch, training_count)
                if last_episode != episode:
                    print("losses - policy: {}, Qs: {}, alpha loss: {}, alpha: {}, log_pis: {}".format(tf.nn.compute_average_loss(policy_losses),
                                                                          tf.nn.compute_average_loss(Qs_losses),
                                                                          tf.nn.compute_average_loss(alpha_losses),
                                                                                          alpha, log_pis.numpy()))

                    last_episode = episode
                training_count+=1

            if done or step == max_steps-1:
                print("done: {} episode: {} reward is {}".format(done, episode, episode_reward))
                break

            state = next_state



