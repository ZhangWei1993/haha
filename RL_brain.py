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
import matplotlib as mp
import matplotlib.pyplot as plt
import math
import matplotlib.gridspec as gridspec

#np.random.seed(1)
#tf.set_random_seed(1)


class DuelingDQN:
    def __init__(
            self,
            action_size,
            state,
            epsilon,
            learning_rate=0.00001,
            reward_decay=0.95,
            e_greedy=0.01,
            replace_target_iter=500,
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
        self.laser = 50
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
        def dqn_layers(s,laser,net_param):
            #choose the network type as cnn
            if self.cnn:
                s = tf.reshape(s,[-1,48,64,1])
                # add the input of laser data
#                laser = tf.shape(laser,[-1,50])
                self.input_image = s
#                self.input_laser = laser
#                tf.image.summary('input1',tf.reshape(s[:,:,:,0],[-1,32,32,1]))
                with tf.variable_scope('conv1'):
                    conv1=tf.contrib.layers.conv2d(
                        inputs = s,
                        num_outputs = 16,
                        kernel_size = [3, 3],
                        stride=2,
                        padding='SAME',
                        data_format=None,
                        rate=1,
                        activation_fn=tf.nn.relu,
                        variables_collections=net_param,
                        trainable=True,
                        scope=None
                    )
                    self.conv1 = conv1
#                with tf.variable_scope('pool1'):
#                    pool1=tf.contrib.layers.max_pool2d(
#                        inputs = conv1,
#                        kernel_size = [2,2],
#                        stride=[2,2],
#                        padding='VALID',
#                        scope=None
#                    )
                with tf.variable_scope('conv2'):
                    conv2=tf.contrib.layers.conv2d(
                        inputs = conv1,
                        num_outputs = 16,
                        kernel_size = [3, 3],
                        stride=2,
                        padding='SAME',
                        data_format=None,
                        rate=1,
                        activation_fn=tf.nn.relu,
                        variables_collections=net_param,
                        trainable=True,
                        scope=None
                    )
                    self.conv2 = conv2
#                with tf.variable_scope('pool2'):
#                    pool2=tf.contrib.layers.max_pool2d(
#                        inputs = conv2,
#                        kernel_size = [2,2],
#                        stride=[2,2],
#                        padding='VALID',
##                        outputs_collections=net_param,
#                        scope=None
#                    )
                with tf.variable_scope('conv3'):
                    conv3=tf.contrib.layers.conv2d(
                        inputs = conv2,
                        num_outputs = 32,
                        kernel_size = [3, 3],
                        stride=2,
                        padding='SAME',
                        data_format=None,
                        rate=1,
                        activation_fn=tf.nn.relu,
                        variables_collections=net_param,
#                        outputs_collections=net_param,
                        trainable=True,
                        scope=None
                    )
                with tf.variable_scope('conv4'):
                    conv4=tf.contrib.layers.conv2d(
                        inputs = conv3,
                        num_outputs = 32,
                        kernel_size = [3, 3],
                        stride=1,
                        padding='VALID',
                        data_format=None,
                        rate=1,
                        activation_fn=tf.nn.relu,
                        variables_collections=net_param,
#                        outputs_collections=net_param,
                        trainable=True,
                        scope=None
                    )
                feature1 = tf.reshape(conv4, [-1, 6 * 4*32])
                feature = tf.contrib.layers.fully_connected(
                        inputs = feature1,
                        num_outputs = 50,
                        activation_fn=tf.nn.relu,
                        variables_collections=net_param,
#                        outputs_collections=net_param,
                        trainable=True,
                        scope=None
                    )
                fully_connected_hidden = 200#the number of each 
            #choose the network type as fully-connected layers

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
                with tf.variable_scope('laser_regulation'):
                    self.Regulation = tf.contrib.layers.fully_connected(
                        inputs = feature,
                        num_outputs = self.laser,
                        activation_fn=None,
                        variables_collections=net_param,
#                        outputs_collections=net_param,
                        trainable=True,
                        scope=None
                    )
                    regulation_max = tf.reduce_max(self.Regulation, axis=1, keep_dims=True)
                    reulation_normal = tf.div(self.Regulation,regulation_max)
                    self.Regulation = reulation_normal
                with tf.variable_scope('Q'):
                    out = self.Val + (self.Adv - tf.reduce_mean(self.Adv, axis=1, keep_dims=True))     # Q = V(s) + A(s,a)
                    regulation = self.Regulation
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
            return out, regulation

        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, 48,64,1], name='s')
        self.temp = tf.placeholder(shape=None,dtype=tf.float32, name='temp')# input
        self.laser = tf.placeholder(tf.float32,[None,50],name = 'laser')
        self.q_target = tf.placeholder(tf.float32, [None, self.actions], name='Q_target')  # for calculating loss
        with tf.variable_scope('eval_net'):
            self.q_eval, self.regulation_1 = dqn_layers(self.s,['eval_net_params',tf.GraphKeys.GLOBAL_VARIABLES])
            self.q_dist = tf.nn.softmax(self.q_eval/self.temp)

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval)) + 0.01*tf.reduce_mean(tf.squared_difference(self.laser, self.regulation_1))
        with tf.variable_scope('train'):
            self._train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, 48,64,1], name='s_')    # input
        with tf.variable_scope('target_net'):
            self.q_next, self.regulation_next = dqn_layers(self.s_, ['target_net_params',tf.GraphKeys.GLOBAL_VARIABLES])

    def remember(self, state, action, reward, next_state,laser_state, done):
        self.memory.append((state, action, reward, next_state,laser_state,done))
        if len(self.memory) > self.memory_size:
            self.memory.popleft()     
    def plot_image(self,state,kind):
        def plotNNFilter(input_image,conv1,conv2,kind):
            filters = input_image.shape[3]
            fig = plt.figure(1,(40,40))
#            gridspec.GridSpec(8,8)
            n_columns = 1
            n_rows = 5
            for i in range(filters):
                fig.add_subplot(n_rows, n_columns, i+1)
                plt.title('input_image ' + str(i))
                plt.imshow(input_image[0,:,:,i], interpolation="nearest", cmap="gray") 
#            fig.savefig('input_image.png')            
            filters1 = conv1.shape[3]
#            fig1 = plt.figure(1, figsize=(20,20))
            n_columns = 8
            n_rows = 5
            for i in range(filters1):
                fig.add_subplot(n_rows, n_columns, i+9)
                plt.title('conv1 ' + str(i))
                plt.imshow(conv1[0,:,:,i], interpolation="nearest", cmap="gray") 
#            fig1.savefig('conv1.png')
            filters2 = conv2.shape[3]
#            fig2 = plt.figure(2, figsize=(20,20))
            n_columns = 8
            n_rows = 5
            for i in range(filters2):
                fig.add_subplot(n_rows, n_columns, i+25)
                plt.title('conv2 ' + str(i))
                plt.imshow(conv2[0,:,:,i], interpolation="nearest", cmap="gray") 
            fig.savefig('state'+kind+'.png')            
        input_image,conv1,conv2 = self.sess.run([self.input_image,self.conv1, self.conv2],feed_dict={self.s_: state})
        plotNNFilter(input_image,conv1,conv2,kind)
        
#        conv2_show = self.sess.run(self.conv2, feed_dict={self.s: state})
#        plotNNFilter(conv2_show)

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
        for state, action, reward, next_state, laser_state, done in minibatch:
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





