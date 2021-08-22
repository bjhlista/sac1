from copy import deepcopy

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

import numpy as np


def hte(action_space):
    return -np.prod(action_space.shape)


class SAC():

    @tf.function(experimental_relax_shapes=True)
    def copy_to_target(self, tau):
        for Q, Q_target in zip(self._Qs, self._Q_targets):
            for source_weight, target_weight in zip(
                    Q.trainable_variables, Q_target.trainable_variables):
                target_weight.assign(
                    tau * source_weight + (1.0 - tau) * target_weight)

    def __init__(self,
                 action_space,
                 policy: tf.keras.Model,
                 Qs,
                 # Q_targets,
                 policy_lr=0.0005,
                 Q_lr=0.0005,
                 alpha_lr=0.005,
                 reward_scale=1.0,
                 gamma=0.99,
                 tau=0.008,
                 target_update_interval: int = 1,
                 target_entropy="auto"
                 ):


        self._policy = policy
        self._Qs = Qs
        self._Q_targets = tuple(deepcopy(Q) for Q in Qs)
        self.copy_to_target(tau=tf.constant(1.0))

        self._policy_lr = policy_lr
        self._Q_lr = Q_lr
        self._alpha_lr = alpha_lr
        self._reward_scale = tf.convert_to_tensor(reward_scale)
        self._gamma = tf.convert_to_tensor(gamma)
        self._tau = tau
        self._target_update_interval = target_update_interval

        self._policy_optimizer = tf.optimizers.Adam(learning_rate=self._policy_lr, name="policy_optimizer", clipvalue=1.0)

        self._Q_optimizers = tuple(
            tf.optimizers.Adam(learning_rate=self._Q_lr, name=f'Q_{i}_optimizer', clipvalue=1.0) for i, Q in enumerate(self._Qs))

        self._log_alpha = tf.Variable(0.1)
        self._alpha = tfp.util.DeferredTensor(self._log_alpha, tf.exp)
        self._target_entropy = (
            hte(action_space)
            if target_entropy == 'auto'
            else target_entropy
        )
        self._alpha_optimizer = tf.optimizers.Adam(learning_rate=self._alpha_lr, name="alpha_optimizer", clipvalue=1.0)


    def get_action(self, state):
        state = tf.expand_dims(state, axis=0)
        actions, _ = self._policy(state)
        action = tf.squeeze(actions, axis=0).numpy()
        return action

    @tf.function(experimental_relax_shapes=True)
    def update_alpha(self, states):
        actions, log_pis = self._policy(states)

        with tf.GradientTape() as tape:
            alpha_losses0 = -self._alpha * tf.stop_gradient(-log_pis - self._target_entropy)
            alpha_losses = -1.0 * (self._alpha * tf.stop_gradient(log_pis + self._target_entropy))
            alpha_loss = tf.nn.compute_average_loss(alpha_losses)

        alpha_gradients = tape.gradient(alpha_loss, [self._log_alpha])
        self._alpha_optimizer.apply_gradients(zip(alpha_gradients, [self._log_alpha]))

        return alpha_losses

    @tf.function(experimental_relax_shapes=True)
    def _update_policy(self, states):
        entropy_scale = tf.convert_to_tensor(self._alpha)

        with tf.GradientTape() as tape:
            actions, log_pis = self._policy(states)
            Qs_log_goals = tuple(
                Q(states, actions) for Q in self._Qs
            )
            Qs_log_goals = tf.reduce_mean(Qs_log_goals, axis=0)
            policy_losses = entropy_scale * log_pis - Qs_log_goals
            policy_loss = tf.nn.compute_average_loss(policy_losses)

        policy_gradients = tape.gradient(policy_loss, self._policy.trainable_variables)
        self._policy_optimizer.apply_gradients(zip(policy_gradients, self._policy.trainable_variables))

        return policy_losses, log_pis

    @tf.function(experimental_relax_shapes=True)
    def compute_Q_targets(self, rewards, dones, next_states):
        next_actions, next_log_pis = self._policy(next_states)
        next_Qs_values = tuple(
            Q(next_states, next_actions) for Q in self._Q_targets
        )
        next_Qs_values = tf.reduce_min(next_Qs_values, axis=0)
        entropy_scale = tf.convert_to_tensor(self._alpha)
        next_values = next_Qs_values - entropy_scale * next_log_pis
        Q_targets = self._reward_scale * rewards + self._gamma * (1-dones)*next_values
        return tf.stop_gradient(Q_targets)

    @tf.function(experimental_relax_shapes=True)
    def _update_critic(self, states, actions, rewards, next_states, dones):
        Q_targets = self.compute_Q_targets(rewards, dones, next_states)

        Qs_values = []
        Qs_losses = []
        for Q, optimizer in zip(self._Qs, self._Q_optimizers):
            with tf.GradientTape() as tape:
                Q_values = Q(states, actions)
                Q_losses = 0.5 * (tf.losses.MSE(y_true=Q_targets, y_pred=Q_values))
                Q_loss = tf.nn.compute_average_loss(Q_losses)

            gradients = tape.gradient(Q_loss, Q.trainable_variables)
            optimizer.apply_gradients(zip(gradients, Q.trainable_variables))
            Qs_losses.append(Q_losses)
            Qs_values.append(Q_values)

        return Qs_values, Qs_losses

    @tf.function(experimental_relax_shapes=True)
    def _do_updates(self, states, actions, rewards, next_states, dones):
        Qs_values, Qs_losses = self._update_critic(states, actions, rewards, next_states, dones)
        policy_losses, log_pis = self._update_policy(states)
        alpha_losses = self.update_alpha(states)
        return Qs_values, Qs_losses, policy_losses, alpha_losses, self._alpha, log_pis

    def time_to_update(self, iteration: int) -> bool:
        return (iteration % self._target_update_interval) == 0

    def train(self, batch, iteration: int):
        t_batch = to_tensors(batch)
        Qs_values, Qs_losses, policy_losses, alpha_losses, alpha, log_pis = self._do_updates(t_batch["states"],
                                                                             t_batch["actions"],
                                                                             t_batch["rewards"],
                                                                             t_batch["next_states"],
                                                                             t_batch["dones"])
        if self.time_to_update(iteration):
            self.copy_to_target(tau=self._tau)

        return Qs_values, Qs_losses, policy_losses, alpha_losses, alpha, log_pis


def to_tensors(batch):
    t_batch = {}
    for key in batch:
        t_batch[key] = tf.convert_to_tensor(batch[key])
    return t_batch

# import tf_agents.distributions.gumbel_softmax as gs


# class DiscretePolicyNetwork(tf.keras.Model):
#     def __init__(self, linear_layers: tf.keras.Sequential, action_space: [int]):
#         super(DiscretePolicyNetwork, self).__init__()
#         self.linear_layers = linear_layers
#         action_space = [3,2,3]
#
#     def call(self, states):
#         logits = self.linear_layers(states)
#         dist = gs.GumbelSoftmax(temperature=.9, logits=logits)  # todo: set hyperparameter temperature
#         samples = dist.sample()
#         action_one_hots = dist.convert_to_one_hot(samples)
#         log_pis = dist.log_prob(action_one_hots)
#         actions = tf.argmax(action_one_hots, axis=1)  # convert to action
#         return actions, log_pis
