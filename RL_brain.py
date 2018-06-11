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
            learning_rate=0.0001,
            reward_decay=0.99,
            e_greedy=0.0001,
            replace_target_iter=1500,
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
        def dqn_layers(s,target,net_param):
            #choose the network type as cnn
            if self.cnn:
                s = tf.reshape(s,[-1,50,1,2])
                target = tf.reshape(target,[-1,2])
                self.input_image = s
#                tf.image.summary('input1',tf.reshape(s[:,:,:,0],[-1,32,32,1]))
                with tf.variable_scope('conv1'):
                    conv1=tf.contrib.layers.conv2d(
                        inputs = s,
                        num_outputs = 16,
                        kernel_size = [3, 1],
                        stride=2,
                        padding='VALID',
                        data_format=None,
                        rate=1,
                        activation_fn=tf.nn.relu,
                        variables_collections=net_param,
                        trainable=True,
                        scope=None
                    )
                feature = tf.reshape(conv1, [-1, 24 * 16])
                fcn_0 = tf.contrib.layers.fully_connected(
                        inputs = feature,
                        num_outputs = 30,
                        activation_fn=None,
                        variables_collections=net_param,
#                        outputs_collections=net_param,
                        trainable=True,
                        scope=None
                    )
                fcn_0 = tf.concat([fcn_0,target],1)
                fully_connected_hidden = 10
                

            if self.dueling:
                # Dueling DQN
                with tf.variable_scope('Value'):
                    fc2_val = tf.contrib.layers.fully_connected(
                        inputs = fcn_0,
                        num_outputs = fully_connected_hidden,
                        activation_fn=tf.nn.relu,
                        variables_collections=net_param,
#                        outputs_collections=net_param,
                        trainable=True,
                        scope=None
                    )
                    self.Val = tf.contrib.layers.fully_connected(
                        inputs = fc2_val,
                        num_outputs = 1,
                        activation_fn=None,
                        variables_collections=net_param,
#                        outputs_collections=net_param,
                        trainable=True,
                        scope=None
                    )
                    
                with tf.variable_scope('Advantage'):
                    fc1_adv = tf.contrib.layers.fully_connected(
                        inputs = fcn_0,
                        num_outputs = fully_connected_hidden,
                        activation_fn=tf.nn.relu,
                        variables_collections=net_param,
#                        outputs_collections=net_param,
                        trainable=True,
                        scope=None
                    )
                    self.Adv = tf.contrib.layers.fully_connected(
                        inputs = fc1_adv,
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
        self.s = tf.placeholder(tf.float32, [None, 50,1,2], name='s')
        self.temp = tf.placeholder(shape=None,dtype=tf.float32, name='temp')# input
        self.target = tf.placeholder(tf.float32,[None,2],name = 'target')
#        self.laser = tf.placeholder(tf.float32,[None,4 * 4*16],name = 'laser')
        self.q_target = tf.placeholder(tf.float32, [None, self.actions], name='Q_target')  # for calculating loss
        with tf.variable_scope('eval_net'):
            self.q_eval = dqn_layers(self.s,self.target,['eval_net_params',tf.GraphKeys.GLOBAL_VARIABLES])
            self.q_dist = tf.nn.softmax(self.q_eval/self.temp)

        with tf.variable_scope('loss'):
#            self.depth_loss1 = tf.reduce_mean(tf.squared_difference(self.laser, self.regu1))
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, 50,1,2], name='s_')
        self.target_ = tf.placeholder(tf.float32,[None,2],name = 'target_')
        # input
        with tf.variable_scope('target_net'):
            self.q_next = dqn_layers(self.s_, self.target_,['target_net_params',tf.GraphKeys.GLOBAL_VARIABLES])

    def remember(self, state, target, action, reward, next_state,next_target,done):
        self.memory.append((state, target, action, reward, next_state,next_target,done))
        if len(self.memory) > self.memory_size:
            self.memory.popleft()     
    def plot_image(self,state,laser_state,kind):
        def plotNNFilter(input_image,feature,regu,kind):
            filters = input_image.shape[3]
            fig = plt.figure(1,(40,40))
#            gridspec.GridSpec(8,8)
            n_columns = 2
            n_rows = 3
            for i in range(filters):
                fig.add_subplot(n_rows, n_columns, i+1)
                plt.title('input_image ' + str(i))
                plt.imshow(input_image[0,:,:,i], interpolation="nearest", cmap="gray") 
#            fig.savefig('input_image.png')            
#            fig1 = plt.figure(1, figsize=(20,20))
            n_columns = 1
            n_rows = 3
            fig.add_subplot(n_rows, n_columns, 2)
            plt.title('feature ')
            plt.imshow(feature, interpolation="nearest", cmap="gray") 
#            fig1.savefig('conv1.png')
#            fig2 = plt.figure(2, figsize=(20,20))
            n_columns = 1
            n_rows = 3
            fig.add_subplot(n_rows, n_columns, 3)
            plt.title('regulation ')
            plt.imshow(regu, interpolation="nearest", cmap="gray") 
            fig.savefig('state'+kind+'.png')            
        input_image,regu = self.sess.run([self.input_image, self.fc2_adv],feed_dict={self.s_: state})
        depth = laser_state
        plotNNFilter(input_image,depth,regu,kind)
        
#        conv2_show = self.sess.run(self.conv2, feed_dict={self.s: state})
#        plotNNFilter(conv2_show)

    def act(self, state,target,exploration):
        if exploration == "e_greedy":
            if np.random.rand() <= self.epsilon:
                return random.randrange(self.action_size)
            act_values = self.sess.run(self.q_eval, feed_dict={self.s: state,self.target:target})
#            act_values, Val, Adv = self.sess.run([self.q_eval,self.Val,self.Adv],feed_dict={self.s: state})
#            Adv = np.max(Adv)-np.mean(Adv)
#            Val = np.mean(Val)
#            print("Value: {}, Advantage: {}" .format(Adv,Val))
            return np.argmax(act_values[0]) # returns action
        if exploration == "boltzmann":
            act_values = self.sess.run(self.q_dist, feed_dict={self.s: state, self.target:target, self.temp: self.epsilon})
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
        for state, target, action, reward, next_state,next_target,done in minibatch:
            q_next = self.sess.run(self.q_next, feed_dict={self.s_: next_state,self.target_: next_target}) # next observation
            q_eval = self.sess.run(self.q_eval, {self.s: state,self.target: target})
            max_action = np.argmax(q_eval,axis=1)
            q_target = q_eval.copy()
            # when calculate the loss, the same part as the q(s,a)will be same
            qvl_target = reward + self.gamma * q_next[0][max_action]
            q_target[0][action] = qvl_target
        
            _, self.cost = self.sess.run([self._train_op, self.loss],
                                         feed_dict={self.s: state, self.target: target,
                                                    self.q_target: q_target})
                                                    
    #modified here
            self.cost_his.append(self.cost)
        self.learn_step_counter += 1





