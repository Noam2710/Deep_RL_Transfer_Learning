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

            self.W1 = tf.get_variable("W1_" + game, [self.state_size, policy_num_n], initializer=kernel_initializer)
            self.b1 = tf.get_variable("b1_" + game, [policy_num_n], initializer=tf.zeros_initializer())
            self.W2 = tf.get_variable("W2_" + game, [policy_num_n, policy_num_n], initializer=kernel_initializer)
            self.b2 = tf.get_variable("b2_" + game, [policy_num_n], initializer=tf.zeros_initializer())
            self.W3 = tf.get_variable("W3_" + game, [policy_num_n, self.action_size], initializer=kernel_initializer)
            self.b3 = tf.get_variable("b3_" + game, [self.action_size], initializer=tf.zeros_initializer())

            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            self.A1 = tf.nn.relu(self.Z1)
            self.Z2 = tf.add(tf.matmul(self.A1, self.W2), self.b2)
            self.A2 = tf.nn.relu(self.Z2)
            self.output = tf.add(tf.matmul(self.A2, self.W3), self.b3)

            self.actions_distribution = tf.squeeze(tf.nn.softmax(self.output))
            self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output, labels=self.action)
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
            self.W1 = tf.get_variable("W1_" + game, [self.state_size, value_num_n], initializer=kernel_initializer)
            self.b1 = tf.get_variable("b1_" + game, [value_num_n], initializer=tf.zeros_initializer())
            self.W2 = tf.get_variable("W2_" + game, [value_num_n, value_num_n], initializer=kernel_initializer)
            self.b2 = tf.get_variable("b2_" + game, [value_num_n], initializer=tf.zeros_initializer())
            self.W3 = tf.get_variable("W3_" + game, [value_num_n, 1], initializer=kernel_initializer)
            self.b3 = tf.get_variable("b3_" + game, [1], initializer=tf.zeros_initializer())

            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            self.A1 = tf.nn.relu(self.Z1)
            self.Z2 = tf.add(tf.matmul(self.A1, self.W2), self.b2)
            self.A2 = tf.nn.relu(self.Z2)
            self.output = tf.add(tf.matmul(self.A2, self.W3), self.b3)

            self.value_estimate = tf.squeeze(self.output)
            self.loss = tf.squared_difference(self.value_estimate, self.R_t)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)


render = False
tf.reset_default_graph()

Policy_Network = PolicyActorNetwork(state_size, action_size)
Value_Network = ValueCriticNetwork(state_size, action_size, learning_rate=lr_value_network)
saver = tf.train.Saver()

data_holder = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    solved = False
    transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
    episode_rewards = np.zeros(max_episodes)

    start = timeit.default_timer()
    for episode in range(max_episodes):
        state = env.reset()
        state = np.pad(state, [(0, (state_size - len(state)))], mode='constant')
        state = state.reshape([1, state_size])
        I = 1.0
        episode_transitions = []

        for step in range(max_steps):
            actions_distribution = sess.run(Policy_Network.actions_distribution, {Policy_Network.state: state})

            action = np.random.choice([-0.7, -0.6, -0.5, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.7], 1, p=actions_distribution)
            velocity = state[0][1]
            action += velocity


            next_state, reward, done, _ = env.step(action)

            next_state = np.pad(next_state, [(0, (state_size - len(next_state)))], mode='constant')
            next_state = next_state.reshape([1, state_size])

            if render:
                env.render()

            action_one_hot = np.zeros(action_size)
            action_one_hot[0] = 1

            episode_transitions.append(
                transition(state=state, action=action_one_hot, reward=reward, next_state=next_state, done=done))
            episode_rewards[episode] += reward

            value_next = 0 if done else sess.run(Value_Network.value_estimate, {Value_Network.state: next_state})

            td_target = reward + discount_factor * value_next
            td_error = td_target - sess.run(Value_Network.value_estimate, {Value_Network.state: state})

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

                    path_to_save = saver.save(sess,"./models/{}.ckpt".format(game))
                    print("Model saved in {}".format(path_to_save))

                    print('Running Time: ', timeit.default_timer() - start)

                break
            state = next_state
            I = I * discount_factor

        if solved:
            break