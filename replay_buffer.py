from collections import deque
import numpy as np
import tensorflow as tf

class ReplayBuffer:
    def __init__(self, state_shape, state_type: type, action_type: type,
                 maxlen: int = 10000, action_shape=None, reward_type: type = None):
        self.buffer = deque(maxlen=maxlen)
        self.state_shape = state_shape
        self.state_type = state_type
        self.action_shape = action_shape
        self.action_type = action_type
        if reward_type is None:
            self.reward_type = int
        else:
            self.reward_type = reward_type

        self.states_shape_list = []
        self.states_shape_list.append(0)
        for s in state_shape:
            self.states_shape_list.append(s)

        if action_shape is not None:
            self.actions_shape_list = []
            self.actions_shape_list.append(0)
            for a in action_shape:
                self.actions_shape_list.append(a)

    def append(self, state, action, reward, next_state, done):
        tuple = (state, action, reward, next_state, done)
        self.buffer.append(tuple)

    def getFirst(self):
        tuple = self.buffer[0]
        return tuple[0], tuple[1], tuple[2], tuple[3], tuple[4]

    def sample(self, batch_size: int):
        max = min(self.len, batch_size)
        indices = sorted(np.random.choice(max, batch_size, replace=False))

        self.states_shape_list[0] = max
        states_shape = tuple(self.states_shape_list)

        if self.action_shape is not None:
            self.actions_shape_list[0] = max
            actions_shape = tuple(self.actions_shape_list)
            actions = np.zeros(shape=(actions_shape), dtype=self.action_type)
        else:
            actions = np.zeros(shape=(max,), dtype=self.action_type)

        states = np.zeros(shape=(states_shape), dtype=self.state_type)
        rewards = np.zeros(shape=(max,1), dtype=self.reward_type)
        next_states = np.zeros(shape=(states_shape), dtype=self.state_type)
        dones = np.zeros(shape=(max,1), dtype=np.float32)

        tuples = self.buffer
        for i in indices:
            row = tuples[i]
            states[i] = (row)[0]
            actions[i] = (row)[1]
            rewards[i] = (row)[2]
            next_states[i] = (row)[3]
            dones[i] = (row)[4]

        batch = {}
        batch["states"] = states
        batch["actions"] = actions
        batch["rewards"] = rewards
        batch["next_states"] = next_states
        batch["dones"] = dones

        return batch


    @property
    def len(self):
        return len(self.buffer)


def to_tensors(batch):
    t_batch = {}
    for key in batch:
        t_batch[key] = tf.convert_to_tensor(batch[key])
    return t_batch


if __name__ == '__main__':

    state = np.array([3.4, 5.1, 10.3])

    action = [1.3, 3.5]
    reward = 3.9
    next_state = [3.6, 5.8, 11.0]
    done = False

    replayBuffer = ReplayBuffer(maxlen=10, state_shape=(3,), state_type=np.float32,
                                action_shape=(2,), action_type=np.float32,
                                reward_type=np.float32)

    replayBuffer.append(state, action, reward, next_state, done)
    replayBuffer.append(state, action, np.float32(10.8), next_state, True)

    print("len ==", replayBuffer.len)

    batch = replayBuffer.sample(2)

    for i in range(batch["states"].shape[0]):
        print("{} {} {:.2f} {} {}".format(batch["states"][i],
                                          batch["actions"][i],
                                          batch["rewards"][i],
                                          batch["next_states"][i],
                                          bool(batch["dones"][i])))

    t_batch = to_tensors(batch)


    print(done)
