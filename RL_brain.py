"""
The Dueling DQN based on this paper: https://arxiv.org/abs/1511.06581

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.8.0
"""

import numpy as np
import tensorflow as tf
from collections import deque
import random

#np.random.seed(1)
#tf.set_random_seed(1)


class DuelingDQN:
    def __init__(
            self,
            action_size,
            state,
            epsilon,
            learning_rate=0.0001,
            reward_decay=0.95,
            e_greedy=0.01,
            replace_target_iter=100,
            memory_size=500,
            batch_size=32,
            e_greedy_decay=None,
            output_graph=True,
            dueling=True,
            cnn = None,
            fcn = True,
            sess=None,
    ):
        self.actions = action_size
        self.state = state
        self.epsilon = epsilon
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_min = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_decay = e_greedy_decay
        self.action_size = action_size
        self.cnn = cnn              # decide using which neural network
        self.fcn = fcn
        self.dueling = dueling      # decide to use dueling DQN or not
        self.learn_step_counter = 0
        self.memory = deque()
        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]
        #assign the value of e_params to t_params

        if sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
        else:
            self.sess = sess
        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)
        self.cost_his = []

    def _build_net(self):
        def dqn_layers(s,net_param):
            #choose the network type as cnn
            if self.cnn:
                s = tf.reshape(s,[-1,self.state,1,1])
                with tf.variable_scope('conv1'):
                    conv1=tf.contrib.layers.conv2d(
                        inputs = s,
                        num_outputs = 4,
                        kernel_size = [3, 1],
                        stride=1,
                        padding='SAME',
                        data_format=None,
                        rate=1,
                        activation_fn=tf.nn.relu,
                        variables_collections=net_param,
                        trainable=True,
                        scope=None
                    )
                with tf.variable_scope('pool1'):
                    pool1=tf.contrib.layers.max_pool2d(
                        inputs = conv1,
                        kernel_size = [2,1],
                        stride=[2,1],
                        padding='VALID',
                        scope=None
                    )
                with tf.variable_scope('conv2'):
                    conv2=tf.contrib.layers.conv2d(
                        inputs = pool1,
                        num_outputs = 4,
                        kernel_size = [3, 1],
                        stride=1,
                        padding='SAME',
                        data_format=None,
                        rate=1,
                        activation_fn=tf.nn.relu,
                        variables_collections=net_param,
                        trainable=True,
                        scope=None
                    )
                with tf.variable_scope('pool2'):
                    pool2=tf.contrib.layers.max_pool2d(
                        inputs = conv2,
                        kernel_size = [2,1],
                        stride=[2,1],
                        padding='VALID',
#                        outputs_collections=net_param,
                        scope=None
                    )
                with tf.variable_scope('conv3'):
                    conv3=tf.contrib.layers.conv2d(
                        inputs = pool2,
                        num_outputs = 4,
                        kernel_size = [3, 1],
                        stride=1,
                        padding='SAME',
                        data_format=None,
                        rate=1,
                        activation_fn=tf.nn.relu,
                        variables_collections=net_param,
#                        outputs_collections=net_param,
                        trainable=True,
                        scope=None
                    )
                with tf.variable_scope('pool3'):
                    pool3=tf.contrib.layers.max_pool2d(
                        inputs = conv3,
                        kernel_size = [2,1],
                        stride=[2,1],
                        padding='VALID',
#                        outputs_collections=net_param,
                        scope=None
                    )
                feature = tf.reshape(pool3, [-1, 25 * 4])
                fully_connected_hidden = 50#the number of each 
            #choose the network type as fully-connected layers
            if self.fcn:
                with tf.variable_scope('fc1'):
                    fc1 = tf.contrib.layers.fully_connected(
                        inputs = s,
                        num_outputs = 100,
                        activation_fn=tf.nn.relu,
                        variables_collections=net_param,
#                        outputs_collections=net_param,
                        trainable=True,
                        scope=None
                    )
                with tf.variable_scope('fc2'):
                    fc2 = tf.contrib.layers.fully_connected(
                        inputs = fc1,
                        num_outputs = 50,
                        activation_fn=tf.nn.relu,
                        variables_collections=net_param,
#                        outputs_collections=net_param,
                        trainable=True,
                        scope=None
                    )
                feature = fc1
                fully_connected_hidden = 20

            if self.dueling:
                # Dueling DQN
                with tf.variable_scope('Value'):
                    fc2 = tf.contrib.layers.fully_connected(
                        inputs = feature,
                        num_outputs = fully_connected_hidden,
                        activation_fn=tf.nn.relu,
                        variables_collections=net_param,
#                        outputs_collections=net_param,
                        trainable=True,
                        scope=None
                    )
                    self.Val = tf.contrib.layers.fully_connected(
                        inputs = fc2,
                        num_outputs = 1,
                        activation_fn=None,
                        variables_collections=net_param,
#                        outputs_collections=net_param,
                        trainable=True,
                        scope=None
                    )
                    
                with tf.variable_scope('Advantage'):
                    fc2 = tf.contrib.layers.fully_connected(
                        inputs = feature,
                        num_outputs = fully_connected_hidden,
                        activation_fn=tf.nn.relu,
                        variables_collections=net_param,
#                        outputs_collections=net_param,
                        trainable=True,
                        scope=None
                    )
                    self.Adv = tf.contrib.layers.fully_connected(
                        inputs = fc2,
                        num_outputs = self.actions,
                        activation_fn=None,
                        variables_collections=net_param,
#                        outputs_collections=net_param,
                        trainable=True,
                        scope=None
                    )

                with tf.variable_scope('Q'):
                    out = self.Val + (self.Adv - tf.reduce_mean(self.Adv, axis=1, keep_dims=True))     # Q = V(s) + A(s,a)
            else:
                with tf.variable_scope('dqn'):
                    fc2 = tf.contrib.layers.fully_connected(
                        inputs = feature,
                        num_outputs = fully_connected_hidden,
                        activation_fn=tf.nn.relu,
                        variables_collections=net_param,
#                        outputs_collections=net_param,
                        trainable=True,
                        scope=None
                    )
                    self.Adv = tf.contrib.layers.fully_connected(
                        inputs = fc2,
                        num_outputs = self.actions,
                        activation_fn=None,
                        variables_collections=net_param,
#                        outputs_collections=net_param,
                        trainable=True,
                        scope=None
                    )
            return out

        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.state], name='s')
        self.temp = tf.placeholder(shape=None,dtype=tf.float32, name='temp')# input
        self.q_target = tf.placeholder(tf.float32, [None, self.actions], name='Q_target')  # for calculating loss
        with tf.variable_scope('eval_net'):
            self.q_eval = dqn_layers(self.s,['eval_net_params',tf.GraphKeys.GLOBAL_VARIABLES])
            self.q_dist = tf.nn.softmax(self.q_eval/self.temp)

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.state], name='s_')    # input
        with tf.variable_scope('target_net'):
            self.q_next = dqn_layers(self.s_, ['target_net_params',tf.GraphKeys.GLOBAL_VARIABLES])


    def remember(self, state, action, reward, next_state,done):
        self.memory.append((state, action, reward, next_state,done))
        if len(self.memory) > self.memory_size:
            self.memory.popleft()

    def act(self, state,exploration):
        if exploration == "e_greedy":
            if np.random.rand() <= self.epsilon:
                return random.randrange(self.action_size)
            act_values = self.sess.run(self.q_eval, feed_dict={self.s: state})
            return np.argmax(act_values[0])  # returns action
        if exploration == "boltzmann":
            act_values = self.sess.run(self.q_dist, feed_dict={self.s: state, self.temp: self.epsilon})
            a = np.random.choice(act_values[0],p=act_values[0])
            a = np.argmax(act_values[0] == a)
            return a
    def epsilon_decay_per_epoch(self):
        self.epsilon = self.epsilon * self.epsilon_decay if self.epsilon > self.epsilon_min else self.epsilon_min
    
    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            q_next = self.sess.run(self.q_next, feed_dict={self.s_: next_state}) # next observation
            q_eval = self.sess.run(self.q_eval, {self.s: state})   
            q_target = q_eval.copy()
            target = reward + self.gamma * np.max(q_next, axis=1)
            q_target[0][action] = target
        
            _, self.cost = self.sess.run([self._train_op, self.loss],
                                         feed_dict={self.s: state,
                                                    self.q_target: q_target})
    #modified here
            self.cost_his.append(self.cost)
        self.learn_step_counter += 1





