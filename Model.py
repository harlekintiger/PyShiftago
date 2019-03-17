import numpy as np
import tensorflow as tf
import matplotlib.pylab as plt
import random
import math

import Game as game
import Board as board
import KerasPlayer as tensorPlayer
import Library as library

MAX_EPSILON = 1
MIN_EPSILON = 0.01
LAMBDA = 0.0001
GAMMA = 0.99
BATCH_SIZE = 50


class Model:
    def __init__(self, num_states, num_actions, batch_size):
        self._num_states = num_states
        self._num_actions = num_actions
        self._batch_size = batch_size
        # define the placeholders
        self._states = None
        self._actions = None
        # the output operations
        self._logits = None
        self._optimizer = None
        self._var_init = None
        # now setup the model
        self._define_model()

    def _define_model(self):
        self._states = tf.placeholder(shape=[None, self._num_states], dtype=tf.float32)
        self._q_s_a = tf.placeholder(shape=[None, self._num_actions], dtype=tf.float32)
        fc1 = tf.layers.dense(self._states, 100, activation=tf.nn.sigmoid)
        fc2 = tf.layers.dense(fc1, 100, activation=tf.nn.sigmoid)
        fc3 = tf.layers.dense(fc2, 75, activation=tf.nn.sigmoid)
        fc4 = tf.layers.dense(fc3, 75, activation=tf.nn.sigmoid)
        fc5 = tf.layers.dense(fc4, 50, activation=tf.nn.softmax)
        fc6 = tf.layers.dense(fc5, 50, activation=tf.nn.softmax)
        self._logits = tf.layers.dense(fc6, self._num_actions)
        loss = tf.losses.mean_squared_error(self._q_s_a, self._logits)
        self._optimizer = tf.train.AdamOptimizer().minimize(loss)
        self._var_init = tf.global_variables_initializer()

    def predict_one(self, state, sess):
        # print("state", state, "    ", type(state), "   ")
        return sess.run(self._logits, feed_dict={self._states: state.reshape(1, self.num_states)})

    def predict_batch(self, states, sess):
        return sess.run(self._logits, feed_dict={self._states: states})

    def train_batch(self, sess, x_batch, y_batch):
        sess.run(self._optimizer, feed_dict={self._states: x_batch, self._q_s_a: y_batch})

    def save_model(self, sess, path_name):
        saver = tf.train.Saver()
        save_path = saver.save(sess, "/tmp/model.ckpt")
        print("Model saved in path: %s" % save_path)

    def restore_model(self, sess, path_name):
        # saver = tf.train.Saver()
        # saver.restore(sess, "/tmp/model.ckpt")
        saver = tf.train.import_meta_graph('/tmp/model.ckpt.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./'))

    @property
    def num_states(self):
        return self._num_states

    @property
    def num_actions(self):
        return self._num_actions

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def var_init(self):
        return self._var_init


class Memory:
    def __init__(self, max_memory):
        self._max_memory = max_memory
        self._samples = []

    def add_sample(self, sample):
        self._samples.append(sample)
        if len(self._samples) > self._max_memory:
            self._samples.pop(0)

    def sample(self, no_samples):
        if no_samples > len(self._samples):
            return random.sample(self._samples, len(self._samples))
        else:
            return random.sample(self._samples, no_samples)


class GameRunner:
    def __init__(self, sess, model, memory, max_eps, min_eps,
                 decay, render=True):
        self._sess = sess
        self._model = model
        self._memory = memory
        self._render = render
        self._max_eps = max_eps
        self._min_eps = min_eps
        self._decay = decay
        self._eps = self._max_eps
        self._steps = 0
        self._reward_store = []
        self._max_x_store = []
        self._game_winner = []
        self._win_to_lose = []
        self._tensor_wins = 0
        self._random_wins = 0

    def run(self, cnt):
        random_only = False
        round_counter = 0
        state = library.to_np_array(game.initialize(False))
        tot_reward = 0
        max_x = -100
        while True:
            # if self._render:
            #     board.draw_board()
            #     print()

            action = self._choose_action(state, random_only)
            done, reward = game.compute_turn_loop(action)

            round_counter += 1
            if round_counter % 10000 == 0:
                random_only = True
                print("Round count:", round_counter, "; Curr action:", action)
                board.draw_board()

            if tensorPlayer.score > max_x:
                max_x = tensorPlayer.score
            # is the game complete? If so, set the next state to
            # None for storage sake
            next_state = library.to_np_array(board.board)
            if done:
                next_state = None

            self._memory.add_sample((state, action, reward, next_state))
            self._replay()

            # exponentially decay the eps value
            self._steps += 1
            self._eps = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) \
                        * math.exp(-LAMBDA * self._steps)

            # move the agent to the next state and accumulate the reward
            state = next_state
            tot_reward += reward
            self._reward_store.append(reward)

            # if the game is done, break the loop
            if done:
                if reward == 100:
                    self._tensor_wins += 1
                elif reward == -100:
                    self._random_wins += 1

                if cnt > 50 and not self.random_wins == 0:
                    self._win_to_lose.append(self._tensor_wins / float(self._random_wins))
                self._max_x_store.append(max_x)
                self._game_winner.append(reward)
                break

        print("Step {}, Total reward: {}, Eps: {}".format(self._steps, tot_reward, self._eps))

    def _choose_action(self, state, random_only=False):
        list_of_legal_moves = board.get_list_of_legal_moves()
        if random.random() < self._eps or random_only:
            while True:
                random_move = random.randint(0, self._model.num_actions - 1)
                if list_of_legal_moves[random_move] == 1:
                    return random_move
        else:
            return np.argmax(self._model.predict_one(state, self._sess) * list_of_legal_moves)

    def _replay(self):
        batch = self._memory.sample(self._model.batch_size)
        states = np.array([val[0] for val in batch])
        next_states = np.array([(np.zeros(self._model.num_states)
                                 if val[3] is None else val[3]) for val in batch])
        # predict Q(s,a) given the batch of states
        q_s_a = self._model.predict_batch(states, self._sess)
        # predict Q(s',a') - so that we can do gamma * max(Q(s'a')) below
        q_s_a_d = self._model.predict_batch(next_states, self._sess)
        # setup training arrays
        x = np.zeros((len(batch), self._model.num_states))
        y = np.zeros((len(batch), self._model.num_actions))
        for i, b in enumerate(batch):
            state, action, reward, next_state = b[0], b[1], b[2], b[3]
            # get the current q values for all actions in state
            current_q = q_s_a[i]
            # update the q value for action
            if next_state is None:
                # in this case, the game completed after action, so there is no max Q(s',a')
                # prediction possible
                current_q[action] = reward
            else:
                current_q[action] = reward + GAMMA * np.amax(q_s_a_d[i])
            x[i] = state
            y[i] = current_q
        self._model.train_batch(self._sess, x, y)

    @property
    def reward_store(self):
        return self._reward_store

    @property
    def max_x_store(self):
        return self._max_x_store

    @property
    def game_winner(self):
        return self._game_winner

    @property
    def win_to_lose(self):
        return self._win_to_lose

    @property
    def tensor_wins(self):
        return self._tensor_wins

    @property
    def random_wins(self):
        return self._random_wins


if __name__ == "__main__":

    num_states = board.board_dimension * board.board_dimension
    num_actions = board.board_dimension * 4

    model = Model(num_states, num_actions, BATCH_SIZE)
    mem = Memory(500000)

    with tf.Session() as sess:
        sess.run(model.var_init)
        gr = GameRunner(sess, model, mem, MAX_EPSILON, MIN_EPSILON, LAMBDA)
        num_episodes = 10000
        cnt = 0
        while cnt < num_episodes:
            if cnt % 10 == 0:
                print('Episode {} of {}'.format(cnt + 1, num_episodes))
            gr.run(cnt)
            cnt += 1
        print("Tensor wins:", gr.tensor_wins, "; Random wins:", gr.random_wins)
        plt.plot(gr.win_to_lose)
        plt.show()
        plt.close("all")
        # plt.plot(gr.reward_store)
        # plt.show()
        # plt.close("all")
        # plt.plot(gr.max_x_store)
        # plt.show()
        # plt.close("all")
        # plt.plot(gr.game_winner)
        # plt.show()
