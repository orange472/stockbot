import argparse
import os
import random
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import tensorflow as tf
from keras.layers import LSTM, BatchNormalization, Dense, Input
from keras.optimizers.legacy import Adam
from tensorflow import keras

from stocktypes.StockTypes import DRQNAgentProps, DRQNModelProps
from utils import PrioritizedReplayBuffer

tf.keras.backend.set_floatx("float64")

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--time_steps", type=int, default=4)
parser.add_argument("--gamma", type=float, default=0.95)
parser.add_argument("--lr", type=float, default=0.005)
parser.add_argument("--eps", type=float, default=1.0)
parser.add_argument("--eps_decay", type=float, default=0.995)
parser.add_argument("--eps_min", type=float, default=0.01)
args = parser.parse_args()


class Model:
    def __init__(self, state_dim: int, action_dim: int, props: DRQNModelProps):
        self.load_path = props["load_path"]
        self.save_path = props["save_path"]
        self.output_file = "model.h5" if props["model_type"] == "model" else "target.h5"

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = args.eps
        self.opt = Adam(args.lr)
        self.loss_function = tf.keras.losses.MeanSquaredError()
        self.model = self.load_model()

    def save_model(self):
        if self.save_path is None:
            return
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        if os.path.isfile(self.save_path):
            self.model.save(self.save_path)
        elif os.path.isdir(self.save_path):
            self.model.save(os.path.join(self.save_path, self.output_file))

    def load_model(self):
        try:
            if self.load_path is not None and os.path.exists(self.load_path):
                print("Loading model from {}...".format(self.load_path))
                model: tf.keras.Sequential = keras.models.load_model(self.load_path)
                if model is None:
                    raise Exception("Failed to load model, creating model instead...")
                model.compile(optimizer=self.opt, loss=self.loss_function)
                return model
            raise Exception("No pretrained model provided, creating model...")
        except Exception as e:
            print(e)
            return self.create_model()

    def create_model(self) -> tf.keras.Sequential:
        model = keras.Sequential()
        model.add(Input((args.time_steps, self.state_dim)))
        model.add(LSTM(32, activation="tanh"))
        model.add(Dense(16, activation="relu"))
        model.add(Dense(self.action_dim))
        model.compile(optimizer=self.opt, loss=self.loss_function)
        return model

    def predict(self, state: np.ndarray) -> np.ndarray:
        return self.model.predict(state)

    def get_action(self, state: np.ndarray) -> Tuple[int, float]:
        state = np.reshape(state, [1, args.time_steps, self.state_dim])
        self.epsilon = max(self.epsilon * args.eps_decay, args.eps_min)
        # Want this to be between 0 and 1
        q_value = self.predict(state)[0]
        probabilities = tf.nn.softmax(q_value)
        if np.random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1), np.random.uniform(0, 1)
        action = np.argmax(q_value)
        return action, probabilities[action]

    def train(self, states, targets, weights):
        targets = tf.stop_gradient(targets)

        with tf.GradientTape() as tape:
            logits = self.model(states, training=True)
            loss = self.loss_function(targets, logits)
            loss = tf.reduce_mean(loss * weights)
            grads = tape.gradient(loss, self.model.trainable_variables)
            self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return logits


class Agent:
    def __init__(self, env, props: DRQNAgentProps):
        self.env = env
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n

        self.states = np.zeros([args.time_steps, self.state_dim])
        self.rewards = []

        model_props: DRQNModelProps = {
            "model_type": "model",
            "save_path": props["save_model"],
        }
        target_props: DRQNModelProps = {
            "model_type": "target",
            "save_path": props["save_target"],
        }
        if props["load_model"] is not None and props["load_target"] is not None:
            model_props["load_path"] = props["load_model"]
            target_props["load_path"] = props["load_target"]

        self.model = Model(self.state_dim, self.action_dim, model_props)
        self.target_model = Model(self.state_dim, self.action_dim, target_props)
        self.target_update()

        self.buffer = PrioritizedReplayBuffer()

    def target_update(self):
        w = self.model.model.get_weights()
        self.target_model.model.set_weights(w)

    def replay(self):
        # Get samples
        batch = self.buffer.sample(args.batch_size, args.time_steps)
        (indices, weights, states, actions, rewards, next_states, done) = batch

        # Compute targets through double-Q
        next_q_values = self.target_model.predict(next_states)
        best_actions = np.argmax(self.model.predict(next_states), axis=1)
        next_q_values = next_q_values[np.arange(args.batch_size), best_actions]
        targets = self.target_model.predict(states)
        targets[range(args.batch_size), actions] = (
            rewards + (1 - done) * next_q_values * args.gamma
        )

        # Train model
        logits = self.model.train(states, targets, weights)

        # Update priorities
        td_errors = (np.asarray(targets) - np.asarray(logits))[
            np.arange(args.batch_size), actions
        ]
        self.buffer.update_priorities(indices, td_errors)

    def update_states(self, next_state: np.ndarray):
        self.states = np.roll(self.states, -1, axis=0)
        self.states[-1] = next_state[0]

    def save_models(self):
        self.model.save_model()
        self.target_model.save_model()

    def train(self, max_episodes: int = 100):
        self.rewards = []

        for ep in range(max_episodes):
            done, total_reward = False, 0
            self.states = np.zeros([args.time_steps, self.state_dim])
            self.update_states(self.env.reset())
            while not done:
                action, fraction = self.model.get_action(self.states)
                next_state, reward, done, _ = self.env.step(action, fraction)
                reward = np.clip(reward, a_min=-100, a_max=100)
                prev_states = self.states
                self.update_states(next_state)
                self.buffer.put(prev_states, action, reward * 0.01, self.states, done)
                total_reward += reward

            if self.buffer.__len__() > args.batch_size:
                for _ in range(8):
                    self.replay()

            self.target_update()
            print("EP{} EpisodeReward={}".format(ep, total_reward))
            self.rewards.append(total_reward)
