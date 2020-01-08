import timeit

from tensorboardcolab import *

# tbc = TensorBoardColab()
import tensorboard
import gym
import numpy as np
import tensorflow as tf
import collections
from ModifiedTensorBoard import ModifiedTensorBoard
from datetime import datetime
import timeit
game = 'MountainCarContinuous-v0'
env = gym.make(game)
env._max_episode_steps = 7000
env.seed(1)
np.random.seed(1)


def print_tests_in_tensorboard(path_for_file_or_name_of_file=None, read_from_file=False, data_holder=None):
    if read_from_file:
        data_holder_to_visualize = np.load(path_for_file_or_name_of_file)
        name_of_log_dir = '{}-{}'.format(path_for_file_or_name_of_file.split("/")[3],
                                         path_for_file_or_name_of_file.split("/")[4])
        name_of_log_dir = name_of_log_dir.split('.')[0]
        name_of_log_dir += datetime.now().strftime("-%m-%d-%Y-%H-%M-%S")
    else:
        data_holder_to_visualize = data_holder
        name_of_log_dir = path_for_file_or_name_of_file

    tensorboard = ModifiedTensorBoard(log_dir="logs/{}".format(name_of_log_dir))

    for data_of_episode in data_holder_to_visualize:
        tensorboard.step = data_of_episode[0]
        tensorboard.update_stats(Number_of_steps=data_of_episode[1],
                                 average_rewards=data_of_episode[2])


class PolicyActorNetwork:
    def __init__(self, state_size, action_size, name='policy_network'):
        self.state_size = state_size
        self.action_size = action_size

        with tf.variable_scope(name):
            self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
            self.action = tf.placeholder(tf.int32, [self.action_size], name="action")
            self.R_t = tf.placeholder(tf.float32, name="total_rewards")
            self.learning_rate = tf.placeholder(tf.float32, name="lr")

            self.W1_1 = tf.get_variable("W1_1_A", [self.state_size, 12],
                                        initializer=tf.contrib.layers.xavier_initializer(seed=0), trainable=False)
            self.b1_1 = tf.get_variable("b1_1_A", [12], initializer=tf.zeros_initializer(), trainable=False)
            self.W2_1 = tf.get_variable("W2_1_A", [12, 12], initializer=tf.contrib.layers.xavier_initializer(seed=0),
                                        trainable=False)
            self.b2_1 = tf.get_variable("b2_1_A", [12], initializer=tf.zeros_initializer(), trainable=False)
            self.W3_1 = tf.get_variable("W3_1_A", [12, self.action_size],
                                        initializer=tf.contrib.layers.xavier_initializer(seed=0), trainable=False)
            self.b3_1 = tf.get_variable("b3_1_A", [self.action_size], initializer=tf.zeros_initializer(),
                                        trainable=False)
            ####
            self.W1_2 = tf.get_variable("W1_1_C", [self.state_size, 12],
                                        initializer=tf.contrib.layers.xavier_initializer(seed=0), trainable=False)
            self.b1_2 = tf.get_variable("b1_1_C", [12], initializer=tf.zeros_initializer(), trainable=False)
            self.W2_2 = tf.get_variable("W2_1_C", [12, 12], initializer=tf.contrib.layers.xavier_initializer(seed=0),
                                        trainable=False)
            self.b2_2 = tf.get_variable("b2_1_C", [12], initializer=tf.zeros_initializer(), trainable=False)
            self.W3_2 = tf.get_variable("W3_1_C", [12, self.action_size],
                                        initializer=tf.contrib.layers.xavier_initializer(seed=0), trainable=False)
            self.b3_2 = tf.get_variable("b3_1_C", [self.action_size], initializer=tf.zeros_initializer(),
                                        trainable=False)
            ####
            self.W1_3 = tf.get_variable("W1_3", [self.state_size, 12],
                                        initializer=tf.contrib.layers.xavier_initializer(seed=0))
            self.b1_3 = tf.get_variable("b1_3", [12], initializer=tf.zeros_initializer())
            self.W2_3 = tf.get_variable("W2_3", [12, 12], initializer=tf.contrib.layers.xavier_initializer(seed=0))
            self.b2_3 = tf.get_variable("b2_3", [12], initializer=tf.zeros_initializer())
            self.W3_3 = tf.get_variable("W3_3", [12, self.action_size],
                                        initializer=tf.contrib.layers.xavier_initializer(seed=0))
            self.b3_3 = tf.get_variable("b3_3", [self.action_size], initializer=tf.zeros_initializer())

            # v1: lines are w_i+b_i and squares are activation
            # w1_i + b1_i
            self.Z1_1 = tf.add(tf.matmul(self.state, self.W1_1), self.b1_1)  # from transfer
            self.Z1_2 = tf.add(tf.matmul(self.state, self.W1_2), self.b1_2)  # from transfer
            self.Z1_3 = tf.add(tf.matmul(self.state, self.W1_3), self.b1_3)
            # relu1
            self.A1_1 = tf.nn.relu(self.Z1_1)
            self.A1_2 = tf.nn.relu(self.Z1_2)
            self.A1_3 = tf.nn.relu(self.Z1_3)
            # w2_i+b2_i
            self.Z2_1 = tf.add(tf.matmul(self.A1_1, self.W2_1), self.b2_1)  # from transfer
            self.Z2_2 = tf.add(tf.matmul(self.A1_2, self.W2_2), self.b2_2)  # from transfer
            self.Z2_3 = tf.add(tf.matmul(self.A1_3, self.W2_3), self.b2_3)
            # relu2 + layers when i<j
            self.A2_1 = tf.nn.relu(self.Z2_1)
            self.A2_2 = tf.nn.relu(tf.add(self.Z2_1, self.Z2_2))
            self.A2_3 = tf.nn.relu(tf.add(tf.add(self.Z2_1, self.Z2_2), self.Z2_3))
            # outputs (w3_i+b3_i)
            self.output_1 = tf.add(tf.matmul(self.A2_1, self.W3_1), self.b3_1)  # from transfer
            self.output_2 = tf.add(tf.matmul(self.A2_2, self.W3_2), self.b3_2)  # from transfer
            self.output_3 = tf.add(tf.matmul(self.A2_3, self.W3_3), self.b3_3)
            # final output - softmax
            self.output_final = tf.nn.softmax(tf.add(tf.add(self.output_1, self.output_2), self.output_3))

            # Softmax probability distribution over actions
            self.actions_distribution = tf.squeeze(self.output_final)
            # Loss with negative log probability
            self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output_3, labels=self.action)
            self.loss = tf.reduce_mean(self.neg_log_prob * self.R_t)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)


class ValueCriticNetwork:
    def __init__(self, state_size, action_size, learning_rate, name='value_network'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name):
            self.state = tf.placeholder(tf.float32, [None, self.state_size], "state")
            self.R_t = tf.placeholder(dtype=tf.float32, name="total_rewards")
            self.W1 = tf.get_variable("W1_1", [self.state_size, 20],
                                      initializer=tf.contrib.layers.xavier_initializer(seed=0))
            self.b1 = tf.get_variable("b1_1", [20], initializer=tf.zeros_initializer())
            self.W2 = tf.get_variable("W2_1", [20, 20], initializer=tf.contrib.layers.xavier_initializer(seed=0))
            self.b2 = tf.get_variable("b2_1", [20], initializer=tf.zeros_initializer())
            self.W3 = tf.get_variable("W3_1", [20, 1], initializer=tf.contrib.layers.xavier_initializer(seed=0))
            self.b3 = tf.get_variable("b3_1", [1], initializer=tf.zeros_initializer())

            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            self.A1 = tf.nn.relu(self.Z1)
            self.Z2 = tf.add(tf.matmul(self.A1, self.W2), self.b2)
            self.A2 = tf.nn.relu(self.Z2)
            self.output = tf.add(tf.matmul(self.A2, self.W3), self.b3)

            self.value_estimate = tf.squeeze(self.output)
            self.loss = tf.squared_difference(self.value_estimate, self.R_t)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)


state_size = 6
action_size = 10
reward_t, average_rewards = 80, 0
max_episodes = 5000
max_steps = 1000000
max_speed = 0.07
max_position = -0.2
discount_factor = 0.99
lr_policy_network = 0.001
lr_value_network = 0.001
lr_decay = 0.999
policy_num_n = 12
value_num_n = 20
kernel_initializer = tf.contrib.layers.xavier_initializer(seed=0)

render = False

# Initialize the policy network
tf.reset_default_graph()
Policy_Network = PolicyActorNetwork(state_size, action_size)
Value_Network = ValueCriticNetwork(state_size, action_size, learning_rate=lr_value_network)

data_holder = []

# Start training the agent with REINFORCE algorithm
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # Load Acrobot
    saver = tf.train.Saver({
        'policy_network/W1_Acrobot-v1': Policy_Network.W1_2,
        'policy_network/W2_Acrobot-v1': Policy_Network.W2_2,
        'policy_network/W3_Acrobot-v1': Policy_Network.W3_2,
    })
    saver.restore(sess, "./models/Acrobot-v1.ckpt")
    # Load Mountain
    saver2 = tf.train.Saver({
        'policy_network/W1_CartPole-v1': Policy_Network.W1_1,
        'policy_network/W2_CartPole-v1': Policy_Network.W2_1,
        'policy_network/W3_CartPole-v1': Policy_Network.W3_1,
    })
    saver2.restore(sess, "./models/CartPole-v1.ckpt")

    solved = False
    transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
    episode_rewards = np.zeros(max_episodes)

    start = timeit.default_timer()
    for episode in range(max_episodes):
        state = env.reset()
        state = np.append(state, [0] * (state_size - len(state)))
        state = state.reshape([1, state_size])
        I = 1.0
        episode_transitions = []

        for step in range(max_steps):
            actions_distribution = sess.run(Policy_Network.actions_distribution, {Policy_Network.state: state})

            action = np.random.choice([-0.7, -0.6, -0.5, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.7], 1, p=actions_distribution)
            velocity = state[0][1]
            action += velocity

            next_state, reward, done, _ = env.step(action)

            next_state = np.append(next_state, [0] * (state_size - len(next_state)))
            next_state = next_state.reshape([1, state_size])

            if render:
                env.render()

            action_one_hot = np.zeros(action_size)
            action_one_hot[0] = 1

            episode_transitions.append(
                transition(state=state, action=action_one_hot, reward=reward, next_state=next_state, done=done))
            episode_rewards[episode] += reward

            value_curr = sess.run(Value_Network.value_estimate, {Value_Network.state: state})

            value_next = 0 if done else sess.run(Value_Network.value_estimate, {Value_Network.state: next_state})

            td_target = reward + discount_factor * value_next
            td_error = td_target - value_curr

            lr_policy_network = lr_policy_network * lr_decay ** episode if lr_policy_network > 0.0001 else 0.0001

            feed_dict_pol = {Policy_Network.state: state, Policy_Network.R_t: td_error * I,
                             Policy_Network.action: action_one_hot,
                             Policy_Network.learning_rate: lr_policy_network}
            _, loss = sess.run([Policy_Network.optimizer, Policy_Network.loss], feed_dict_pol)

            feed_dict_val = {Value_Network.state: state, Value_Network.R_t: td_target}
            _, loss = sess.run([Value_Network.optimizer, Value_Network.loss], feed_dict_val)

            if done:
                if episode > 98:
                    average_rewards = np.mean(episode_rewards[(episode - 99):episode + 1])
                data_holder.append([episode, step, average_rewards])

                print("Episode {} Reward: {} Average over 100 episodes: {}".format(episode, episode_rewards[episode],
                                                                                   round(average_rewards, 2)))
                if average_rewards > reward_t:
                    print_tests_in_tensorboard(
                        path_for_file_or_name_of_file="{}_{}".format(game, episode),
                        data_holder=data_holder)
                    solved = True
                    print('Running Time: ', timeit.default_timer() - start)

                break
            state = next_state
            I = I * discount_factor

        if solved:
            break