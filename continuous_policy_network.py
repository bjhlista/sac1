import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


class ContinuousPolicyNetwork(tf.keras.Model):
    def __init__(self):
        super(ContinuousPolicyNetwork, self).__init__()
        self.fc1 = tf.keras.layers.Dense(256, activation="relu",
                                         kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
                                         bias_initializer=tf.keras.initializers.Zeros())
        self.fc2 = tf.keras.layers.Dense(256, activation="relu",
                                         kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
                                         bias_initializer=tf.keras.initializers.Zeros())

        self.mu = tf.keras.layers.Dense(1, activation="linear")
        self.sigma = tf.keras.layers.Dense(1, activation="linear")

    def call(self, states):
        fc1 = self.fc1(states)
        fc2 = self.fc2(fc1)

        mean = self.mu(fc2)
        log_std = self.sigma(fc2)
        log_std = tf.clip_by_value(log_std, -20.0, 2.0)
        std = tf.math.exp(log_std)

        probabilities = tfp.distributions.Normal(mean, std)
        z = probabilities.sample()
        actions = tf.math.tanh(z)
        log_p = probabilities.log_prob(z)
        correction = tf.reduce_sum(-2.0 * (tf.math.log(2.0) - z - tf.math.softplus(-2.0 * z)), axis=1, keepdims=True)
        log_pis = log_p + correction
        # old log_probs = probabilities.log_prob(z) - tf.math.log(1-tf.math.pow(actions,2)+.0001)
        # old log_pis = tf.math.reduce_sum(log_probs, axis=1, keepdims=True)
        return actions, log_pis


class QNetwork(tf.keras.Model):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc1 = tf.keras.layers.Dense(256, activation="relu",
                                         kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
                                         bias_initializer=tf.keras.initializers.Zeros())
        self.fc2 = tf.keras.layers.Dense(256, activation="relu",
                                         kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
                                         bias_initializer=tf.keras.initializers.Zeros())
        self.q = tf.keras.layers.Dense(1, activation="linear")

    def call(self, state, action):
        fc1 = self.fc1(tf.concat([state, action], axis=1))
        fc2 = self.fc2(fc1)
        q = self.q(fc2)
        return q
