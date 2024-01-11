import numpy as np
import tensorflow as tf

from stocktypes import Experience

from .sum_tree import SumTree


class PrioritizedReplayBuffer:
    def __init__(
        self,
        capacity: int = 10000,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
        epsilon: float = 0.01,
    ):
        self.tree = SumTree(capacity)

        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon

        self.error_upper = 1.0
        self.max_priority = 1.0

    def put(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        experience = Experience(state, action, reward, next_state, done)
        priority = self.max_priority**self.alpha
        self.tree.add(priority, experience)

    def sample(self, batch_size: int, time_steps: int):
        indices = []
        priorities = []
        samples = []

        segment = self.tree.total() / batch_size
        self.beta = np.min([1.0, self.beta + self.beta_increment])

        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)
            value = np.random.uniform(a, b)
            index, priority, experience = self.tree.get(value)
            assert isinstance(experience, Experience)
            indices.append(index)
            priorities.append(priority)
            samples.append(experience)

        states, actions, rewards, next_states, dones = map(np.asarray, zip(*samples))
        states = np.array(states).reshape(batch_size, time_steps, -1)
        next_states = np.array(next_states).reshape(batch_size, time_steps, -1)

        probabilities = priorities / self.tree.total()
        probabilities = tf.convert_to_tensor(probabilities, dtype=tf.float32)
        weights = np.power(self.tree.n_entries * probabilities, -self.beta)
        weights /= weights.max()

        return (indices, weights, states, actions, rewards, next_states, dones)

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        clipped_errors = np.minimum(np.abs(td_errors) + self.epsilon, self.error_upper)
        priorities = np.power(clipped_errors, self.alpha)
        for index, priority in zip(indices, priorities):
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(index, priority)

    def __len__(self) -> int:
        return self.tree.n_entries
